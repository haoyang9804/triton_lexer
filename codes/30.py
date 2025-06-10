
























import triton
import triton.language as tl
from triton.language import core
from triton.language.extra.cuda.libdevice import ffs


@core.extern
def __syncthreads(_builder=None):
    return tl.tensor(_builder.create_barrier(), tl.void)


@core.extern
def __fence(scope: core.constexpr = core.constexpr("gpu"), _builder=None):
    return core.inline_asm_elementwise(
        asm=f,
        constraints="=r",  
        args=[],
        dtype=tl.uint32,
        is_pure=False,  
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def load_v4_u32(ptr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=,
        constraints=("=r,=r,=r,=r,l"),  
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def load_v2_b64(ptr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=,
        constraints=("=l,=l,l"),  
        args=[ptr],
        dtype=(tl.int64, tl.int64),
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def store_v2_u32(ptr, val0, val1, _builder=None):
    return tl.inline_asm_elementwise(
        asm=,
        constraints=("=r,l,r,r"),  
        args=[ptr, val0, val1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def multimem_st_b64(ptr, val0, _builder=None):
    return tl.inline_asm_elementwise(
        asm=,
        constraints=("=r,l,l"),  
        args=[ptr, val0],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def multimem_st_b32(ptr, val0, _builder=None):
    return tl.inline_asm_elementwise(
        asm=,
        constraints=("=r,l,r"),  
        args=[ptr, val0],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def multimem_st_v2_b32(ptr, val0, val1, _builder=None):
    return tl.inline_asm_elementwise(
        asm=,
        constraints=("=r,l,r,r"),  
        args=[ptr, val0, val1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )



@tl.core.extern
def multimem_ld_reduce(ptr, op, _builder=None):
    tl.static_assert(ptr.is_ptr(), "multimem_ld_reduce(ptr) expect ptr is a pointer_type")
    if ptr.dtype == tl.int32:
        return tl.inline_asm_elementwise(
            asm=,
            constraints=("=r,l,r"),  
            args=[ptr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
            _builder=_builder,
        )


@core.extern
def _tid_wrapper(axis: core.constexpr, _builder=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                f"llvm.nvvm.read.ptx.sreg.tid.{axis.value}",
                core.dtype("int32"),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def tid(axis: core.constexpr, _builder=None):
    if axis == 0:
        return _tid_wrapper(core.constexpr("x"), _builder=_builder)
    elif axis == 1:
        return _tid_wrapper(core.constexpr("y"), _builder=_builder)
    elif axis == 2:
        return _tid_wrapper(core.constexpr("z"), _builder=_builder)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")


@core.extern
def laneid(_builder=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                "llvm.nvvm.read.ptx.sreg.laneid",
                core.dtype("int32"),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def _ntid_wrapper(axis: core.constexpr, _builder=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                f"llvm.nvvm.read.ptx.sreg.ntid.{axis.value}",
                core.dtype("int32"),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def ntid(axis: core.constexpr, _builder=None):
    if axis == 0:
        return _ntid_wrapper(core.constexpr("x"), _builder=_builder)
    elif axis == 1:
        return _ntid_wrapper(core.constexpr("y"), _builder=_builder)
    elif axis == 2:
        return _ntid_wrapper(core.constexpr("z"), _builder=_builder)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")



@tl.core.extern
def red_release(barrier_ptr, value, scope: core.constexpr = core.constexpr("gpu"), _builder=None):
    tl.inline_asm_elementwise(
        asm=f,
        constraints=("=r,"
                     "l,r"),  
        args=[barrier_ptr, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def arrive_inc(barrier_ptr, thread_idx, value, scope: core.constexpr):
    __syncthreads()
    if thread_idx == 0:
        red_release(barrier_ptr, value, scope)



@tl.core.extern
def arrive_inc_asm(barrier_ptr, thread_idx, value, scope: core.constexpr = "gpu", _builder=None):
    tl.inline_asm_elementwise(
        asm=f,
        constraints=("=r,"
                     "l,r,r"),  
        args=[barrier_ptr, thread_idx, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@core.extern
def _int_constaint(bitwidth: core.constexpr, _builder=None):
    
    
    if bitwidth.value == 128:
        return core.constexpr("q")
    if bitwidth.value == 64:
        return core.constexpr("l")
    elif bitwidth.value == 32:
        return core.constexpr("r")
    elif bitwidth.value == 16:
        return core.constexpr("h")
    elif bitwidth.value == 8:
        return core.constexpr("r")
    else:
        tl.static_assert(False, "unsupported dtype")


@core.extern
def _float_constraint(bitwidth: core.constexpr, _builder=None):
    if bitwidth.value == 64:
        return core.constexpr("d")
    elif bitwidth.value == 32:
        return core.constexpr("f")
    else:
        tl.static_assert(False, "unsupported dtype")


@core.extern
def ld(
    ptr,
    scope: core.constexpr = "gpu",
    semantic: core.constexpr = "relaxed",
    _builder=None,
):
    tl.static_assert(ptr.dtype.is_ptr(), "ld(ptr, scope) should be a pointer", _builder=_builder)
    if isinstance(scope, core.constexpr):
        scope = scope.value
    tl.static_assert(scope in ["gpu", "sys"], "scope should be gpu or sys", _builder=_builder)
    if isinstance(semantic, core.constexpr):
        semantic = semantic.value
    tl.static_assert(
        semantic in ["relaxed", "acquire"],
        "semantic should be relaxed or acquire",
        _builder=_builder,
    )
    element_ty: tl.dtype = ptr.dtype.element_ty
    constraint = _int_constaint(core.constexpr(element_ty.primitive_bitwidth), _builder=_builder)
    
    return tl.inline_asm_elementwise(
        asm=f"ld.global.{semantic}.{scope}.b{element_ty.primitive_bitwidth} $0, [$1];",
        constraints=f"={constraint.value},l",
        args=[ptr],
        dtype=ptr.dtype.element_ty,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def ld_b32(ptr, _builder=None):
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int32(),
        "ld_b32(ptr) argument 0 `ptr` should be a pointer of int type",
        _builder=_builder,
    )
    return ld(ptr, scope="gpu", semantic="relaxed", _builder=_builder)


@tl.core.extern
def ld_acquire(ptr, scope: core.constexpr = "gpu", _builder=None):
    return ld(ptr, scope, "acquire", _builder=_builder)


@tl.core.extern
def st(
        ptr,
        val,
        scope: core.constexpr = core.constexpr("gpu"),
        semantic: core.constexpr = core.constexpr("relaxed"),
        _builder=None,
):
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int(),
        "st(ptr, val) argument 0 `ptr` should be a pointer of int type",
        _builder=_builder,
    )
    dtype = ptr.dtype.element_ty
    if isinstance(val, core.constexpr):
        val = tl.cast(val.value, dtype, _builder=_builder)
    else:
        val = tl.cast(val, dtype, _builder=_builder)

    tl.static_assert(val.dtype.is_int(), "st(ptr, val) argument `val` should be of int type", _builder=_builder)

    if isinstance(scope, core.constexpr):
        scope = scope.value
    if isinstance(semantic, core.constexpr):
        semantic = semantic.value
    tl.static_assert(scope in ["gpu", "sys"], "scope should be gpu or sys", _builder=_builder)
    tl.static_assert(
        semantic in ["relaxed", "release"],
        "semantic should be relaxed or release",
        _builder=_builder,
    )
    constraint = _int_constaint(core.constexpr(dtype.primitive_bitwidth), _builder=_builder)
    
    
    return tl.inline_asm_elementwise(
        asm=f,
        constraints=(f"=r,l,{constraint.value}"),  
        args=[ptr, val],
        dtype=tl.int32,  
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def st_b32(ptr, val0, _builder=None):
    return st(ptr, val0, scope="gpu", semantic="relaxed", _builder=_builder)


@tl.core.extern
def atomic_add(
    ptr,
    value,
    scope: core.constexpr = "gpu",
    semantic: core.constexpr = "relaxed",
    _builder=None,
):
    
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int(),
        "ptr must be a pointer of int",  
        _builder=_builder,
    )
    if isinstance(scope, core.constexpr):
        scope = scope.value
    tl.static_assert(scope in ["gpu", "sys"], "scope should be gpu or sys", _builder=_builder)
    if isinstance(semantic, core.constexpr):
        semantic = semantic.value
    tl.static_assert(
        semantic in ["release", "acquire", "relaxed", "acq_rel"],
        "semantic should be release, acquire, relaxed or acq_rel",
        _builder=_builder,
    )
    constraint = _int_constaint(core.constexpr(ptr.dtype.element_ty.primitive_bitwidth), _builder=_builder).value

    
    
    return tl.inline_asm_elementwise(
        asm=
        f"atom.{semantic}.{scope}.global.add.{'s' if ptr.dtype.element_ty.is_int_signed() else 'u'}{ptr.dtype.element_ty.primitive_bitwidth} $0, [$1], $2;",
        constraints=(f"={constraint},l,{constraint}"),
        args=[
            ptr,
            value,
        ],
        is_pure=False,
        pack=1,
        dtype=ptr.dtype.element_ty,
        _builder=_builder,
    )


@triton.jit
def atomic_add_per_warp(barrier_ptr, value, scope: core.constexpr, semantic: core.constexpr):
    _laneid = laneid()
    x = tl.cast(0, barrier_ptr.dtype.element_ty)
    if _laneid == 0:
        x = atomic_add(barrier_ptr, value, scope, semantic)
    return __shfl_sync_i32(0xFFFFFFFF, x, 0)


@triton.jit
def wait_eq(barrier_ptr, thread_idx, value, scope: core.constexpr):
    if thread_idx == 0:
        while ld_acquire(barrier_ptr, scope) != value:
            pass
    __syncthreads()


@tl.core.extern
def __shfl_sync_with_mode_i32(
    mask,
    value,
    delta,
    mode: core.constexpr = "up",
    c: core.constexpr = 31,
    _builder=None,
):
    tl.static_assert(value.dtype == tl.int32 or value.dtype == tl.uint32,
                     "__shfl_sync_i32 only support int32 or uint32", _builder=_builder)
    
    return tl.inline_asm_elementwise(
        asm=f"shfl.sync.{mode.value}.b32 $0, $1, $2, {c.value}, $3;",
        constraints="=r,r,r,r",
        args=[value, delta, mask],
        dtype=value.dtype,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def __shfl_sync_i32(mask, value, laneid):
    return __shfl_sync_with_mode_i32(mask, value, laneid, "idx", 31)


@triton.jit
def __shfl_up_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "up", 0)


@triton.jit
def __shfl_down_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "down", 31)


@triton.jit
def __shfl_xor_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "bfly", 31)



@tl.core.extern
def __ballot_sync(
    mask,
    predicate,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm="{.reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2;}",
        constraints="=r,r,r",
        args=[predicate, mask],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope: core.constexpr,
    semantic: core.constexpr,
    _builder=None,
):
    constraint = _int_constaint(core.constexpr(ptr.dtype.element_ty.primitive_bitwidth), _builder=_builder).value
    return tl.inline_asm_elementwise(
        asm=
        f"atom.{semantic.value}.{scope.value}.global.cas.b{ptr.dtype.element_ty.primitive_bitwidth} $0, [$1], $2, $3;",
        constraints=(f"={constraint},l,{constraint},{constraint}"),
        args=[
            ptr,
            value,
            target_value,
        ],
        dtype=ptr.dtype.element_ty,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


__all__ = [
    "__syncthreads",
    "__fence",
    "tid",
    "ntid",
    "laneid",
    "wait_eq",
    "arrive_inc",
    "red_release",
    "ld_acquire",
    "atomic_add",
    "atomic_add_per_warp",
    "__shfl_sync_i32",
    "__shfl_up_sync_i32",
    "__shfl_down_sync_i32",
    "__shfl_xor_sync_i32",
    "__ballot_sync",
    "ld",
    "ffs",
    "atomic_cas",
    "ld_b32",
    "st_b32",
]
