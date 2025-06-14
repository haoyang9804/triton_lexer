import os
import triton
import triton.language as tl


def _test_pid_conds(conds, pid0=0, pid1=0, pid2=0):

    pids = pid0, pid1, pid2
    conds = conds.replace(" ", "").split(",")
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond == "":
            continue
        if cond[:2] in ["<=", ">=", "!="]:
            op, threshold = cond[:2], int(cond[2:])
        elif cond[:1] in ["<", ">", "="]:
            op, threshold = cond[:1], int(cond[1:])
        else:
            raise ValueError(
                f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule in '{cond}'."
            )
        op = "==" if op == "=" else op
        if not eval(f"{pid} {op} {threshold}"):
            return False
    return True


@triton.jit
def test_pid_conds(conds):

    return _test_pid_conds(
        conds,
        tl.program_id(0).handle.data[0],
        tl.program_id(1).handle.data[0],
        tl.program_id(2).handle.data[0],
    )


@triton.jit
def breakpoint_if(conds):

    from IPython.core.debugger import set_trace

    if test_pid_conds(conds):
        set_trace()


@triton.jit
def print_if(*txt, conds):

    if test_pid_conds(conds):
        print(*txt)


@triton.jit
def breakpoint_once():
    breakpoint_if("=0,=0,=0")


@triton.jit
def print_once(*txt):
    print_if(*txt, conds="=0,=0,=0")


def assert_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous(), "A tensor is not contiguous"
        if not os.environ.get("TRITON_INTERPRET") == "1":
            assert t.is_cuda, "A tensor is not on cuda"


@triton.jit
def offsets_from_base(ptrs, base_ptr):

    return ptrs.to(tl.uint64) - base_ptr.to(tl.uint64)
