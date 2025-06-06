from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, nvidia
from triton.runtime.errors import PTXASError

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path
import sysconfig
from string import Template


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        if lhs_bitwidth == 8:
            return (16, 16, 32)
        else:
            return (16, 16, 16)

    return check_dot_compatibility


@functools.lru_cache()
def _path_to_binary(binary: str):
    binary += sysconfig.get_config_var("EXE")
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
    ]

    cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda/bin")

    paths += [f"{cuda_home}/{binary}"]

    for path in paths:
        if os.path.exists(path) and os.path.isfile(path):
            result = subprocess.check_output(
                [path, "--version"], stderr=subprocess.STDOUT
            )
            if result is not None:
                version = re.search(
                    r".*release (\d+\.\d+).*",
                    result.decode("utf-8"),
                    flags=re.MULTILINE,
                )
                if version is not None:
                    return path, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_ptxas():
    name = "ptxas"
    return _path_to_binary(name)


@functools.lru_cache()
def get_nvlink():
    return _path_to_binary("nvlink")


@functools.lru_cache()
def get_nvcc():
    return _path_to_binary("nvcc")


@functools.lru_cache()
def get_ptxas_version():
    mock_ver = os.environ.get("TRITON_MOCK_PTX_VERSION")
    if mock_ver is not None:
        return mock_ver
    version = subprocess.check_output([get_ptxas()[0], "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:

    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split("."))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError(
        "Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version
    )


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        _, cuda_version = get_ptxas()
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    llvm_ptx_version = min(86, ptx_version)
    features = f"+ptx{llvm_ptx_version}"
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):

    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"


class NVSHMEMHelper:

    @staticmethod
    @functools.lru_cache()
    def get_nvshmem_home():
        return Path(
            os.environ.get(
                "NVSHMEM_HOME",
                Path(os.path.realpath(__file__)).parent.parent.parent
                / "nvshmem"
                / "build"
                / "install",
            )
        )

    @staticmethod
    @functools.lru_cache()
    def get_nvshmem_lib():
        return NVSHMEMHelper.get_nvshmem_home() / "lib"

    @staticmethod
    @functools.lru_cache()
    def get_aot_nvshmem_cubin(capability):
        return Path(__file__).parent / "lib" / f"nvshmem_wrapper.sm{capability}.cubin"

    @staticmethod
    @functools.lru_cache()
    def get_nvshmem_wrapper_src():
        return (
            Path(os.path.realpath(__file__)).parent.parent.parent
            / "nvshmem_bind"
            / "runtime"
            / "nvshmem_wrapper.cu"
        )

    @staticmethod
    @functools.lru_cache()
    def extract_nvshmem_functions() -> dict:
        file_path = NVSHMEMHelper.get_nvshmem_wrapper_src()
        functions = {}
        with open(file_path, "r") as f:
            content = f.read()

        extern_block_pattern = re.compile(
            r'extern "C" {\s*' r"((?:__device__.*?}\s*)+)" r"}", re.DOTALL
        )

        device_func_pattern = re.compile(
            r"__device__\s+"
            r"(?:[\w\*]+\s+)+?"
            r"(\w+)\s*\([^\)]*\)\s*"
            r"\{.*?\}(?=\s*__device__|\s*$)",
            re.DOTALL,
        )

        for extern_block in extern_block_pattern.finditer(content):
            block_content = extern_block.group(1)
            for match in device_func_pattern.finditer(block_content):
                func_name = match.group(1)
                full_code = match.group(0).strip()
                functions[func_name] = full_code

        return functions

    @staticmethod
    def generate_sub_cu(user_ptx):
        functions = NVSHMEMHelper.extract_nvshmem_functions()
        symbols = []
        jit_funcs = []
        for k, v in functions.items():
            if k in user_ptx:
                symbols.append(k)
                jit_funcs.append(v)
        content = "\n".join(jit_funcs)
        code_template = Template()
        code = code_template.substitute(content=content)
        return code

    @staticmethod
    def get_jit_nvshmem_cubin(user_ptx: str, capability: int):
        jit_code = NVSHMEMHelper.generate_sub_cu(user_ptx)
        NVSHMEM_HOME = NVSHMEMHelper.get_nvshmem_home()
        arch = sm_arch_from_capability(capability)
        suffix = "a" if capability >= 90 else ""
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".cu"
        ) as fsrc, tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".ptx"
        ) as fptx, tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".cubin"
        ) as fbin:
            fsrc.write(jit_code)
            fsrc.flush()

            NVCC_GENCODE = f"-gencode=arch=compute_{capability}{suffix},code={arch}"
            nvcc, _ = get_nvcc()

            nvcc_cmd = [
                nvcc,
                "-rdc=true",
                "-ccbin",
                "g++",
                NVCC_GENCODE,
                "-I",
                os.path.join(NVSHMEM_HOME, "include"),
                fsrc.name,
                "-ptx",
                "-c",
                "-o",
                fptx.name,
            ]

            try:
                subprocess.run(nvcc_cmd, check=True, close_fds=False)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"PTX generation failed: {e}")
            fptx.flush()
            ptxas, _ = get_ptxas()

            ptxas_cmd = [ptxas, "-c", fptx.name, f"--gpu-name={arch}", "-o", fbin.name]

            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"PTX assembly failed for {arch}: {e}")

            return fbin.name

    @staticmethod
    def get_nvshmem_cubin(user_ptx, capability):
        aot_cubin_file = NVSHMEMHelper.get_aot_nvshmem_cubin(capability=capability)
        if os.path.exists(aot_cubin_file):
            return aot_cubin_file
        else:
            cubin = NVSHMEMHelper.get_jit_nvshmem_cubin(user_ptx, capability)
            return cubin


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3

    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    nvshmem_device_lib: str = ""
    debug: bool = False
    backend_name: str = "cuda"
    sanitize_overflow: bool = True
    arch: str = None

    def __post_init__(self):
        default_libdir = Path(__file__).parent / "lib"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get("libdevice", None):
            extern_libs["libdevice"] = os.getenv(
                "TRITON_LIBDEVICE_PATH", str(default_libdir / "libdevice.10.bc")
            )
        nvshmem_device_lib = os.getenv(
            "TRITON_LIBDEVICE_PATH", str(default_libdir / "libnvshmem_device.bc")
        )
        nvshmemi_device_lib = os.getenv(
            "TRITON_LIBDEVICE_PATH", str(default_libdir / "libnvshmemi_device.bc")
        )

        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))
        object.__setattr__(self, "nvshmem_device_lib", nvshmem_device_lib)
        object.__setattr__(self, "nvshmemi_device_lib", nvshmemi_device_lib)
        assert (
            self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0
        ), "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple(
            (k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"])
        )
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cuda"

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        args = {"arch": os.getenv("TRITON_OVERRIDE_ARCH", f"sm{self.target.arch}")}
        args.update(
            {
                k: opts[k]
                for k in CUDAOptions.__dataclass_fields__.keys()
                if k in opts
                if opts[k] is not None
            }
        )
        capability = int(self._parse_arch(args["arch"]))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dtypes"] = ("fp8e4b15",)

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = os.getenv("TRITON_DEFAULT_FP_FUSION", "1") == "1"

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda

        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types": (
                cuda.convert_custom_float8_sm80
                if capability >= 80
                else cuda.convert_custom_float8_sm70
            ),
            "min_dot_size": min_dot_size(self.target),
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        from triton.language.extra.cuda import libnvshmem_device

        return {
            "triton.language.extra.libdevice": libdevice,
            "triton.language.extra.libshmem_device": libnvshmem_device,
        }

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        cluster_info = nvidia.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas
        )

        passes.ttgpuir.add_coalesce(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_f32_dot_tc(pm)

        nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        passes.common.add_cse(pm)
        if capability // 10 in [8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        elif capability // 10 >= 10:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
            passes.ttgpuir.add_hoist_tmem_alloc(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
            passes.common.add_canonicalizer(pm)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (
            cluster_info.clusterDimX,
            cluster_info.clusterDimY,
            cluster_info.clusterDimZ,
        )
        return mod

    def make_llir(self, src, metadata, options, capability):
        ptx_version = get_ptx_version_from_options(options, self.target.arch)

        mod = src

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        nvidia.passes.ttnvgpuir.add_lower_mma(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)

        llvm.init_targets()
        context = llvm.context()
        if os.environ.get("TRITON_ENABLE_ASAN", "0") == "1":
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend"
            )
        llvm_mod = llvm.to_module(mod, context)
        proc = sm_arch_from_capability(capability)
        features = get_features(options, self.target.arch)
        triple = "nvptx64-nvidia-cuda"
        nvidia.set_short_ptr()
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        nvidia.set_nvvm_reflect_ftz(llvm_mod)

        if options.maxnreg is not None:
            for k in llvm_mod.get_functions():
                if not k.is_declaration() and k.is_external_linkage():
                    k.set_nvvm_maxnreg(options.maxnreg)
        metadata["use_nvshmem"] = False
        metadata["use_nvshmem_wrapper"] = False
        for k in llvm_mod.get_functions():

            if "nvshmem" in k.name and k.is_declaration():
                metadata["use_nvshmem"] = True
                if k.name.startswith("nvshmem") and k.name.endswith("wrapper"):
                    metadata["use_nvshmem_wrapper"] = True

        if "nvshmemi_device_state_d" in str(llvm_mod):
            metadata["use_nvshmem"] = True

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)
        if (
            options.nvshmem_device_lib
            and metadata["use_nvshmem"]
            and not metadata["use_nvshmem_wrapper"]
        ):
            llvm.link_extern_libs(llvm_mod, [options.nvshmem_device_lib])
            if os.path.exists(options.nvshmemi_device_lib):

                llvm.link_extern_libs(llvm_mod, [options.nvshmemi_device_lib])
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = src.get_int_attr(
            "ttg.global_scratch_memory_size"
        )
        metadata["global_scratch_align"] = src.get_int_attr(
            "ttg.global_scratch_memory_alignment"
        )
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_ptx(self, src, metadata, opt, capability):
        ptx_version = get_ptx_version_from_options(opt, self.target.arch)

        triple = "nvptx64-nvidia-cuda"
        proc = sm_arch_from_capability(capability)
        features = get_features(opt, self.target.arch)
        ret = llvm.translate_to_asm(
            src, triple, proc, features, [], opt.enable_fp_fusion, False
        )

        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]

        ptx_version = f"{ptx_version//10}.{ptx_version%10}"
        ret = re.sub(
            r"\.version \d+\.\d+", f".version {ptx_version}", ret, flags=re.MULTILINE
        )
        ret = re.sub(
            r"\.target sm_\d+", f".target sm_{capability}", ret, flags=re.MULTILINE
        )

        ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if os.environ.get("NVPTX_ENABLE_DUMP", "0") == "1":
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    def make_cubin(self, src, metadata, opt, capability):
        ptxas, _ = get_ptxas()
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".ptx"
        ) as fsrc, tempfile.NamedTemporaryFile(
            delete=False, mode="r", suffix=".log"
        ) as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + ".o"

            fbin_combined = fbin + ".combined.cubin"
            has_nvshmem_wrapper = metadata["use_nvshmem_wrapper"]
            compile_only_cmds = ["-c"] if has_nvshmem_wrapper else []

            line_info = (
                ["-lineinfo", "-suppress-debug-info"]
                if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "1"
                else ["-lineinfo"]
            )
            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
            arch = sm_arch_from_capability(capability)
            opt_level = (
                ["--opt-level", "0"]
                if os.environ.get("DISABLE_PTXAS_OPT", "0") == "1"
                else []
            )
            ptxas_cmd = [
                ptxas,
                *compile_only_cmds,
                *line_info,
                *fmad,
                "-v",
                *opt_level,
                f"--gpu-name={arch}",
                fsrc.name,
                "-o",
                fbin,
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = "Internal Triton PTX codegen error"
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = "`ptxas` raised SIGSEGV"
                else:
                    error = f"`ptxas` failed with error code {e.returncode}"

                raise PTXASError(
                    f"{error}\n"
                    f"`ptxas` stderr:\n{log}\n"
                    f'Repro command: {" ".join(ptxas_cmd)}\n'
                )

            if has_nvshmem_wrapper:

                nvlink, _ = get_nvlink()
                nvlink_cmds = [
                    nvlink,
                    f"-arch={arch}",
                    f"-L{NVSHMEMHelper.get_nvshmem_lib()}",
                    "-lnvshmem_device",
                    fbin,
                    NVSHMEMHelper.get_nvshmem_cubin(src, capability).__str__(),
                    "-o",
                    fbin_combined,
                ]
                try:
                    subprocess.run(
                        nvlink_cmds, check=True, close_fds=False, stderr=flog
                    )
                except Exception as e:
                    import logging

                    logging.error(f"error runing nvlink: {nvlink_cmds}")
                    logging.exception(e)
            if has_nvshmem_wrapper:
                with open(fbin_combined, "rb") as f:
                    cubin = f.read()
            else:
                with open(fbin, "rb") as f:
                    cubin = f.read()
            if os.path.exists(fbin_combined):
                os.remove(fbin_combined)

            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options):
        capability = self._parse_arch(options.arch)
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(
            src, metadata, options, capability
        )
        stages["llir"] = lambda src, metadata: self.make_llir(
            src, metadata, options, capability
        )
        stages["ptx"] = lambda src, metadata: self.make_ptx(
            src, metadata, options, self.target.arch
        )
        stages["cubin"] = lambda src, metadata: self.make_cubin(
            src, metadata, options, self.target.arch
        )

    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version()
        return f"{version}-{self.target.arch}"
