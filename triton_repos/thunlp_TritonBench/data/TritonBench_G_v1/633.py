import torch
import triton
import triton.language as tl


@triton.jit
def spinning_lock_kernel(
    P,
    C,
    locks,
    num_sms,
    k,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // num_sms
    pid_n = pid % num_sms

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for iters in range(1, 10):
        if pid % k == 0:
            next_pid = pid + 1

            while next_pid < pid + k and next_pid < num_sms:
                while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
                    pass

                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                P_ = (
                    P
                    + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N
                    + rm1[:, None] * BLOCK_SIZE_N
                    + rn1[None, :]
                )
                acc1 = tl.load(P_)
                acc += acc1

                next_pid += 1

        else:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            P_ = (
                P
                + pid * BLOCK_SIZE_M * BLOCK_SIZE_N
                + rm1[:, None] * BLOCK_SIZE_N
                + rn1[None, :]
            )
            tl.store(P_, acc)
            tl.atomic_xchg(locks + pid, 1)

        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, acc, mask=mask)


def spinning_lock(
    P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N
):
    grid = (num_sms,)
    spinning_lock_kernel[grid](
        P,
        C,
        locks,
        num_sms,
        k,
        M,
        N,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )


def test_spinning_lock():

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    M = 1024
    N = 1024
    num_sms = 304
    k = 3

    P = torch.zeros(
        (num_sms * BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=torch.float32, device="cuda"
    )
    C = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    locks = torch.zeros(num_sms, dtype=torch.int32, device="cuda")

    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    result = {}

    spinning_lock(
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    result["test_case_1"] = C.clone()

    k = 2
    spinning_lock(
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    result["test_case_2"] = C.clone()

    num_sms = 2
    spinning_lock(
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    result["test_case_3"] = C.clone()

    num_sms = 5
    spinning_lock(
        P, C, locks, num_sms, k, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    result["test_case_4"] = C.clone()

    return result


result_gold = test_spinning_lock()
