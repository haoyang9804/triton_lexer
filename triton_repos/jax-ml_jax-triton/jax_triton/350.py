from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
import dataclasses
import functools
import inspect
import os
import pprint
import tempfile
import types
from typing import Any, Protocol, Union
import zlib

from absl import logging
import jax
from jax import tree_util
from jax._src import core
from jax._src import state
from jax._src import util
from jax._src.lib import gpu_triton as triton_kernel_call_lib
import jax.dlpack
import jax.extend as jex
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
import numpy as np


CAN_USE_TRITON = False
try:
    import triton
    from triton.compiler import code_generator as code_gen
    from triton.compiler import compiler as tc
    import triton.language as tl
    from triton.runtime import autotuner
    import triton._C.libtriton as _triton
    import triton.backends.nvidia.compiler as cb

    CAN_USE_TRITON = True
except ModuleNotFoundError:
    pass

try:
    import triton.backends.amd.compiler as hb
except ImportError:
    hb = None
    pass


os.environ["TRITON_CACHE_DIR"] = ""
_JAX_TRITON_DUMP_DIR = os.environ.get("JAX_TRITON_DUMP_DIR")
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


_JAX_TO_TRITON_TYPE_MAP = {
    jnp.dtype("bfloat16"): "bf16",
    jnp.dtype("float64"): "fp64",
    jnp.dtype("float32"): "fp32",
    jnp.dtype("float16"): "fp16",
    jnp.dtype("float8_e4m3fn"): "fp8e4nv",
    jnp.dtype("float8_e5m2"): "fp8e5",
    jnp.dtype("float8_e4m3fnuz"): "fp8e4b8",
    jnp.dtype("float8_e5m2fnuz"): "fp8e5b16",
    jnp.dtype("int64"): "i64",
    jnp.dtype("int32"): "i32",
    jnp.dtype("int16"): "i16",
    jnp.dtype("int8"): "i8",
    jnp.dtype("uint64"): "u64",
    jnp.dtype("uint32"): "u32",
    jnp.dtype("uint16"): "u16",
    jnp.dtype("uint8"): "u8",
    jnp.dtype("bool"): "i1",
}

Grid = Union[int, tuple[int], tuple[int, int], tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[dict[str, Any]], Grid]]


def normalize_grid(grid: GridOrLambda, metaparams) -> tuple[int, int, int]:
    if callable(grid):
        grid = grid(metaparams)
    if isinstance(grid, int):
        grid = (grid,)
    elif len(grid) > 3:
        raise ValueError("`grid` should have three or fewer dimensions.")
    return tuple(grid) + (1,) * (3 - len(grid))


def avals_to_layouts(avals):
    return [list(reversed(range(aval.ndim))) for aval in avals]


def get_triton_type(obj: Any) -> str:
    if isinstance(obj, (jax.core.ShapedArray, state.AbstractRef)):
        return f"*{_JAX_TO_TRITON_TYPE_MAP[obj.dtype]}"
    if isinstance(obj, tl.constexpr):
        obj = obj.value
    if isinstance(obj, int):
        if -(2**31) <= obj < 2**31:
            return "i32"
        elif 2**31 <= obj < 2**32:
            return "u32"
        elif -(2**63) <= obj < 2**63:
            return "i64"
        elif 2**63 <= obj < 2**64:
            return "u64"
        else:
            raise ValueError(f"integer overflow representing {obj}")
    if isinstance(obj, float):
        return "fp64"
    if isinstance(obj, np.float32):
        return "fp32"
    if isinstance(obj, bool):
        return "B"
    if isinstance(obj, str):
        return "str"
    raise NotImplementedError(f"could not compute type name for {obj}: {type(obj)}")


triton_kernel_call_p = jex.core.Primitive("triton_kernel_call")
triton_kernel_call_p.multiple_results = True
triton_kernel_call_p.def_impl(
    functools.partial(xla.apply_primitive, triton_kernel_call_p)
)


@triton_kernel_call_p.def_abstract_eval
def triton_kernel_call_abstract_eval(*_, out_shapes, **__):
    return [
        core.ShapedArray(out_shape.shape, out_shape.dtype) for out_shape in out_shapes
    ]


def aval_size_bytes(aval):
    return np.dtype(aval.dtype).itemsize * aval.size


def get_cuda_backend(device, compute_capability):
    target = cb.GPUTarget("cuda", compute_capability, 32)
    backend = cb.CUDABackend(target)
    return backend


def get_hip_backend(device, compute_capability):
    arch = triton_kernel_call_lib.get_arch_details(device)
    arch = arch.split(":")[0]
    target = hb.GPUTarget("hip", arch, 64)
    backend = hb.HIPBackend(target)
    return backend


@dataclasses.dataclass
class CompilationResult:
    binary: str
    name: str
    shared_mem_bytes: int
    cluster_dims: tuple
    ttgir: str | None
    llir: str | None


def compile_ttir_inplace(
    ttir,
    backend: [cb.CUDABackend | hb.HIPBackend],
    options: [cb.CUDAOptions | hb.HIPOptions],
    compute_capability,
    platform,
):
    if platform == "cuda":
        return compile_ttir_to_ptx_inplace(
            ttir,
            backend,
            options,
            compute_capability,
        )

    elif platform == "rocm":
        return compile_ttir_to_hsaco_inplace(
            ttir,
            backend,
            options,
            compute_capability,
        )
    else:
        raise ValueError("Unsupported device.")


def compile_ttir_to_ptx_inplace(
    ttir,
    cuda_backend: cb.CUDABackend,
    cuda_options: cb.CUDAOptions,
    compute_capability,
) -> CompilationResult:
    if cuda_options.debug:
        print(ttir)
    try:
        metadata = {}
        opt_ttir = cuda_backend.make_ttir(
            ttir, metadata, cuda_options, compute_capability
        )
        ttgir = cuda_backend.make_ttgir(
            opt_ttir,
            metadata,
            cuda_options,
            compute_capability,
        )
    except RuntimeError as e:
        ttir.dump()
        raise ValueError("TTIR->TTGIR pass failed!") from e
    if cuda_options.debug:
        print(ttgir)
    try:
        llir = cuda_backend.make_llir(
            ttgir,
            metadata,
            cuda_options,
            compute_capability,
        )
    except RuntimeError as e:
        ttgir.dump()
        raise ValueError("TTGIR->LLIR pass failed!") from e
    shared_mem_bytes = metadata["shared"]
    if cuda_options.debug:
        print(llir)
    ptx = cuda_backend.make_ptx(
        llir,
        metadata,
        cuda_options,
        compute_capability,
    )
    if cuda_options.debug:
        print(ptx)
    name = metadata["name"]
    cluster_dims = metadata["cluster_dims"]
    ttgir = str(ttgir) if _JAX_TRITON_DUMP_DIR else None
    llir = str(llir) if _JAX_TRITON_DUMP_DIR else None
    return CompilationResult(
        binary=ptx,
        name=name,
        shared_mem_bytes=shared_mem_bytes,
        cluster_dims=cluster_dims,
        ttgir=ttgir,
        llir=llir,
    )


def compile_ttir_to_hsaco_inplace(
    ttir,
    hip_backend: hb.HIPBackend,
    hip_options: hb.HIPOptions,
    compute_capability,
) -> CompilationResult:
    if hip_options.debug:
        print(ttir)
    try:
        metadata = {}
        opt_ttir = hip_backend.make_ttir(ttir, metadata, hip_options)
        ttgir = hip_backend.make_ttgir(opt_ttir, metadata, hip_options)
    except RuntimeError as e:
        ttir.dump()
        raise ValueError("TTIR->TTGIR pass failed!") from e
    if hip_options.debug:
        print(ttgir)
    try:
        llir = hip_backend.make_llir(ttgir, metadata, hip_options)
    except RuntimeError as e:
        ttgir.dump()
        raise ValueError("TTGIR->LLIR pass failed!") from e
    shared_mem_bytes = metadata["shared"]
    if hip_options.debug:
        print(llir)

    amdgcn = hip_backend.make_amdgcn(llir, metadata, hip_options)
    hsaco = hip_backend.make_hsaco(amdgcn, metadata, hip_options)

    name = metadata["name"]
    ttgir = str(ttgir) if _JAX_TRITON_DUMP_DIR else None
    llir = str(llir) if _JAX_TRITON_DUMP_DIR else None

    cluster_dims = (0, 0, 0)

    fd, hsaco_path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        f.write(hsaco)
    return CompilationResult(
        binary=hsaco_path,
        name=name,
        shared_mem_bytes=shared_mem_bytes,
        cluster_dims=cluster_dims,
        ttgir=ttgir,
        llir=llir,
    )


_COMPILED_KERNEL_CACHE = {}


def get_or_create_triton_kernel(
    backend_init_func,
    platform,
    fn,
    arg_dtypes,
    scalar_args,
    *,
    num_warps,
    num_stages,
    num_ctas,
    compute_capability,
    enable_fp_fusion,
    metaparams,
    dump: bool,
) -> tuple[triton_kernel_call_lib.TritonKernel, Any]:
    if num_warps is None:
        num_warps = 4
    if num_stages is None:
        num_stages = 3

    device = 0
    if compute_capability is None:
        compute_capability = triton_kernel_call_lib.get_compute_capability(device)
    if num_ctas > 1 and compute_capability < 90:
        raise ValueError("num_ctas > 1 unsupported before Hopper.")

    backend = backend_init_func(device, compute_capability)

    signature = {fn.arg_names[i]: v for i, v in enumerate(arg_dtypes)}

    alignments = [16] * len(arg_dtypes)
    for i, _, value in scalar_args:
        alignments[i] = value
    specialize_extra = backend.get_arg_specialization
    if specialize_impl := getattr(triton.runtime.jit, "specialize_impl", None):

        specialize_impl = functools.partial(
            specialize_impl, specialize_extra=specialize_extra
        )
    else:

        create_specialize_impl = triton.runtime.jit.create_specialize_impl
        if len(inspect.signature(create_specialize_impl).parameters) == 0:

            specialize_impl = functools.partial(
                create_specialize_impl(), specialize_extra=specialize_extra
            )
        else:

            specialize_impl = create_specialize_impl(specialize_extra)
    specialization = [
        specialize_impl(
            types.SimpleNamespace(
                data_ptr=lambda: alignment, dtype=arg_dtype.removeprefix("*")
            ),
        )
        for arg_dtype, alignment in zip(arg_dtypes, alignments)
    ]
    attrs = {
        (i,): backend.parse_attr(attr) for i, (_, attr) in enumerate(specialization)
    }
    constants = dict(metaparams)
    constants.update({k: None for _, k, v in scalar_args if v is None})
    constants.update({fn.arg_names[i]: 1 for i, _, v in scalar_args if v == 1})
    for constant in constants:
        signature[constant] = "constexpr"

    cache_key = (
        fn,
        tuple(signature.items()),
        tuple(specialization),
        tuple(constants.items()),
        num_warps,
        num_stages,
        num_ctas,
        compute_capability,
        enable_fp_fusion,
    )
    kernel = _COMPILED_KERNEL_CACHE.get(cache_key)

    if kernel is None:
        opts = {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "num_ctas": num_ctas,
            "optimize_epilogue": False,
            "debug": dump,
            "enable_fp_fusion": enable_fp_fusion,
        }

        options = backend.parse_options(opts)

        kernel_hash = abs(hash(cache_key))
        if _JAX_TRITON_DUMP_DIR:
            os.makedirs(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}")
            with open(f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/config", "w") as f:
                pprint.pprint(cache_key, stream=f)
                pprint.pprint(options, stream=f)

        context = _triton.ir.context()
        _triton.ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation(options)

        module = code_gen.ast_to_ttir(
            fn,
            tc.ASTSource(fn, constexprs=constants, signature=signature, attrs=attrs),
            options=options,
            codegen_fns=codegen_fns,
            context=context,
            module_map=backend.get_module_map(),
        )
        ttir = str(module)

        compilation_result = compile_ttir_inplace(
            module, backend, options, compute_capability, platform
        )

        kernel_name = compilation_result.name
        if _JAX_TRITON_DUMP_DIR:
            with open(
                f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ttir", "w"
            ) as f:
                f.write(ttir)
            with open(
                f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ptx", "w"
            ) as f:
                f.write(compilation_result.binary)
            with open(
                f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.ttgir", "w"
            ) as f:
                f.write(compilation_result.ttgir)
            with open(
                f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.llir", "w"
            ) as f:
                f.write(compilation_result.llir)
            with open(
                f"{_JAX_TRITON_DUMP_DIR}/{kernel_hash}/{kernel_name}.compile_info",
                "w",
            ) as f:
                f.write(
                    f"{kernel_name}: shared_mem_bytes:"
                    f" {compilation_result.shared_mem_bytes}, cluster_dims:"
                    f" {compilation_result.cluster_dims}\n"
                )

        kernel = triton_kernel_call_lib.TritonKernel(
            kernel_name,
            num_warps,
            compilation_result.shared_mem_bytes,
            compilation_result.binary,
            ttir,
            compute_capability,
            *compilation_result.cluster_dims,
        )

        _COMPILED_KERNEL_CACHE[cache_key] = kernel

    return kernel, attrs


def triton_kernel_call_lowering(
    backend_init_func,
    ctx,
    *array_args,
    fn,
    scalar_args,
    name,
    custom_call_target_name,
    out_shapes,
    grid,
    num_warps,
    num_stages,
    num_ctas,
    compute_capability,
    enable_fp_fusion,
    input_output_aliases,
    zeroed_outputs,
    debug,
    serialized_metadata,
    **metaparams,
):
    kernel_call_name = name
    args = list(ctx.avals_in)
    arg_dtypes = list(map(get_triton_type, ctx.avals_in))
    for idx, dtype, v in scalar_args:
        args.insert(idx, v)
        arg_dtypes.insert(idx, dtype)
    args.extend(ctx.avals_out)
    arg_dtypes.extend(map(get_triton_type, ctx.avals_out))
    named_args = dict(unsafe_zip(fn.arg_names, args))

    if isinstance(fn, autotuner.Autotuner):
        if hasattr(fn, "key_idx"):
            key_idxs = fn.key_idx
        else:
            key_idxs = [fn.arg_names.index(k) for k in fn.keys]
        if any(idx not in key_idxs for idx, _, _ in scalar_args):
            logging.warning(
                "Auto-tuning key does not include all scalar arguments. "
                "We may perform redundant auto-tuning."
            )

        prev_early_config_prune_fn = fn.early_config_prune

        def prune_configs(configs, named_args, **kwargs):
            pruned_configs = []
            for config in configs:
                if config.pre_hook is not None:
                    raise NotImplementedError("`pre_hook` is not supported")

                if all(config.kwargs.get(k, v) == v for k, v in metaparams.items()):
                    pruned_configs.append(config)
            if prev_early_config_prune_fn is not None:
                pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
            return pruned_configs

        fn.early_config_prune = prune_configs
        fn.nargs = named_args
        configs = fn.prune_configs(metaparams)
        fn = fn.fn
    else:
        config = triton.Config(
            {},
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
        )
        configs = [config]

    if isinstance(fn, autotuner.Heuristics):
        updated_configs = []
        for config in configs:
            kwargs = config.kwargs.copy()
            for name, heuristic in fn.values.items():
                kwargs[name] = heuristic({**named_args, **metaparams, **kwargs})
            updated_config = copy.copy(config)
            updated_config.kwargs = kwargs
            updated_configs.append(updated_config)
        configs = updated_configs
        fn = fn.fn

    if not isinstance(fn, triton.JITFunction):
        raise ValueError(
            "`kernel` must be a Triton `JITFunction`, `Heuristics` or `Autotuner`."
        )

    outputs_offset = len(ctx.avals_in) + len(scalar_args)
    config_params = []
    for config in configs:
        config_metaparams = {**metaparams, **config.kwargs}
        config_grid = normalize_grid(grid, config_metaparams)

        config_zeroed_outputs = zeroed_outputs
        if callable(zeroed_outputs):
            config_zeroed_outputs = config_zeroed_outputs(config_metaparams)

        zeroed_params_with_sizes = {
            i + outputs_offset: aval_size_bytes(ctx.avals_out[i])
            for i in sorted(config_zeroed_outputs)
        }

        config_params.append(
            dict(
                metaparams=tuple(sorted(config_metaparams.items())),
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                num_ctas=config.num_ctas,
                grid=config_grid,
                zeroed_params_with_sizes=tuple(zeroed_params_with_sizes.items()),
            )
        )

    kernel_calls = []
    for params in config_params:
        kernel, specialization_attr = get_or_create_triton_kernel(
            backend_init_func,
            ctx.module_context.platforms[0],
            fn,
            arg_dtypes,
            scalar_args,
            num_warps=params["num_warps"],
            num_stages=params["num_stages"],
            num_ctas=params["num_ctas"],
            compute_capability=compute_capability,
            enable_fp_fusion=enable_fp_fusion,
            metaparams=dict(params["metaparams"]),
            dump=debug,
        )

        kernel_params = []
        zeroed_params_with_sizes = dict(params["zeroed_params_with_sizes"])
        equal_to_1 = {i for i, _, v in scalar_args if v == 1}
        for i, (arg, dtype) in enumerate(zip(args, arg_dtypes)):
            if isinstance(arg, core.ShapedArray):
                arg_attrs = specialization_attr[(i,)]
                kernel_params.append(
                    triton_kernel_call_lib.create_array_parameter(
                        zeroed_params_with_sizes.get(i, 0),
                        16 if (["tt.divisibility", 16] in arg_attrs) else 0,
                    )
                )
            elif i not in equal_to_1:
                kernel_params.append(
                    triton_kernel_call_lib.create_scalar_parameter(arg, dtype)
                )

        kernel_calls.append(
            triton_kernel_call_lib.TritonKernelCall(
                kernel,
                params["grid"][0],
                params["grid"][1],
                params["grid"][2],
                kernel_params,
            )
        )

    if len(kernel_calls) > 1:
        named_scalar_args = {fn.arg_names[i]: v for i, _, v in scalar_args}
        input_output_aliases_with_sizes = tuple(
            (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
            for input_idx, output_idx in input_output_aliases
        )
        kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
            f"{kernel_call_name} ({fn.fn.__name__}) {named_scalar_args}",
            [(call, str(config)) for call, config in zip(kernel_calls, configs)],
            input_output_aliases_with_sizes,
        )
    else:
        kernel_call = kernel_calls[0]

    call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)
    rule = jax.ffi.ffi_lowering(
        custom_call_target_name,
        api_version=2,
        backend_config=zlib.compress(call_proto),
        operand_output_aliases=dict(input_output_aliases),
    )
    return rule(ctx, *array_args)


mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, get_cuda_backend),
    platform="cuda",
)

mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, get_hip_backend),
    platform="rocm",
)


def triton_kernel_call_raise_on_jvp(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "jax_triton.triton_call does not support automatic differentiation. Use "
        "jax.custom_jvp or jax.custom_vjp to implement a custom automatic "
        "differentiation rule for your kernel."
    )


ad.primitive_jvps[triton_kernel_call_p] = triton_kernel_call_raise_on_jvp


def triton_kernel_call_raise_on_vmap(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "jax_triton.triton_call does not support batching with jax.vmap. Use "
        "jax.custom_batching.custom_vmap to implement a custom batching rule for "
        "your kernel."
    )


batching.primitive_batchers[triton_kernel_call_p] = triton_kernel_call_raise_on_vmap


class ShapeDtype(Protocol):

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...


def triton_call(
    *args: jax.Array | bool | int | float | np.float32,
    kernel: triton.JITFunction,
    out_shape: ShapeDtype | Sequence[ShapeDtype],
    grid: GridOrLambda,
    name: str = "",
    custom_call_target_name: str = "triton_kernel_call",
    num_warps: int | None = None,
    num_stages: int | None = None,
    num_ctas: int = 1,
    compute_capability: int | None = None,
    enable_fp_fusion: bool = True,
    input_output_aliases: dict[int, int] | None = None,
    zeroed_outputs: Sequence[int] | Callable[[dict[str, Any]], Sequence[int]] = (),
    debug: bool = False,
    serialized_metadata: bytes = b"",
    **metaparams: Any,
) -> Any:

    if not CAN_USE_TRITON:
        raise ValueError("`triton_call` is only available when `triton` is installed.")
    out_shape = tree_util.tree_map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape
    )
    flat_args, _ = tree_util.tree_flatten(args)

    flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)

    array_args = []
    scalar_args = []
    for i, arg in enumerate(flat_args):
        if isinstance(arg, (bool, int, float)):
            scalar_args.append((i, get_triton_type(arg), arg))
        elif isinstance(arg, np.float32):
            scalar_args.append((i, get_triton_type(arg), float(arg)))
        else:
            array_args.append(arg)

    if input_output_aliases is None:
        input_output_aliases = {}

    out_flat = triton_kernel_call_p.bind(
        *array_args,
        fn=kernel,
        scalar_args=tuple(scalar_args),
        name=name,
        custom_call_target_name=custom_call_target_name,
        out_shapes=tuple(flat_out_shapes),
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
        compute_capability=compute_capability,
        enable_fp_fusion=enable_fp_fusion,
        input_output_aliases=tuple(input_output_aliases.items()),
        zeroed_outputs=zeroed_outputs,
        debug=debug,
        serialized_metadata=serialized_metadata,
        **metaparams,
    )
    return tree_util.tree_unflatten(out_tree, out_flat)
