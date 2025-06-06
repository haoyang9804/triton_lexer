import argparse
from typing import List
import os

import torch
import triton
import triton.language as tl


from display import print_end_line
from tensor_type import Float32, Int32
from test_puzzle import test





r





@triton.jit
def demo1(x_ptr):
    range = tl.arange(0, 8)
    
    print(range)
    x = tl.load(x_ptr + range, range < 5, 0)
    print(x)


def run_demo1():
    print("Demo1 Output: ")
    demo1[(1, 1, 1)](torch.ones(4, 3))
    print_end_line()





@triton.jit
def demo2(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range * 4 + j_range
    
    print(range)
    x = tl.load(x_ptr + range, (i_range < 4) & (j_range < 3), 0)
    print(x)


def run_demo2():
    print("Demo2 Output: ")
    demo2[(1, 1, 1)](torch.ones(4, 4))
    print_end_line()





@triton.jit
def demo3(z_ptr):
    range = tl.arange(0, 8)
    z = tl.store(z_ptr + range, 10, range < 5)


def run_demo3():
    print("Demo3 Output: ")
    z = torch.ones(4, 3)
    demo3[(1, 1, 1)](z)
    print(z)
    print_end_line()





@triton.jit
def demo4(x_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    x = tl.load(x_ptr + range, range < 20)
    print("Print for each", pid, x)


def run_demo4():
    print("Demo4 Output: ")
    x = torch.ones(2, 4, 4)
    demo4[(3, 1, 1)](x)
    print_end_line()


r


def add_spec(x: Float32[32,]) -> Float32[32,]:
    "This is the spec that you should implement. Uses typing to define sizes."
    return x + 10.0


@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    
    off_x = tl.arange(0, B0)
    x = tl.load(x_ptr + off_x)
    
    x = x + 10.0
    tl.store(z_ptr + off_x, x)
    return


r


def add2_spec(x: Float32[200,]) -> Float32[200,]:
    return x + 10.0


@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    
    block_id = tl.program_id(0)
    off_x = block_id * B0 + tl.arange(0, B0)
    mask = off_x < N0
    x = tl.load(x_ptr + off_x, mask=mask)
    x = x + 10.0
    tl.store(z_ptr + off_x, x, mask=mask)
    return


r


def add_vec_spec(x: Float32[32,], y: Float32[32,]) -> Float32[32, 32]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    
    off_x = tl.arange(0, B0)
    off_y = tl.arange(0, B1)
    off_z = off_y[:, None] * B0 + off_x[None, :]
    
    x = tl.load(x_ptr + off_x)
    y = tl.load(y_ptr + off_y)
    z = y[:, None] + x[None, :]
    
    tl.store(z_ptr + off_z, z)
    return


r


def add_vec_block_spec(x: Float32[100,], y: Float32[90,]) -> Float32[90, 100]:
    return x[None, :] + y[:, None]


@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    
    off_x = block_id_x * B0 + tl.arange(0, B0)
    off_y = block_id_y * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]
    
    
    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)
    z = y[:, None] + x[None, :]
    
    tl.store(z_ptr + off_z, z, mask=mask_z)
    return


r


def mul_relu_block_spec(x: Float32[100,], y: Float32[90,]) -> Float32[90, 100]:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    
    off_x = block_id_x * B0 + tl.arange(0, B0)
    off_y = block_id_y * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]
    
    
    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)
    z = x[None, :] * y[:, None]
    relu_z = tl.where(z > 0, z, 0.0)
    
    tl.store(z_ptr + off_z, relu_z, mask=mask_z)
    return


r


def mul_relu_block_back_spec(
    x: Float32[90, 100], y: Float32[90,], dz: Float32[90, 100]
) -> Float32[90, 100]:
    x = x.clone()
    y = y.clone()
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = torch.relu(x * y[:, None])
    z.backward(dz)
    dx = x.grad
    return dx


@triton.jit
def mul_relu_block_back_kernel(
    x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    block_id_j = tl.program_id(1)
    
    
    
    
    off_i = block_id_i * B0 + tl.arange(0, B0)
    off_j = block_id_j * B1 + tl.arange(0, B1)
    off_ji = off_j[:, None] * N0 + off_i[None, :]

    mask_i = off_i < N0
    mask_j = off_j < N1
    mask_ji = mask_j[:, None] & mask_i[None, :]

    x = tl.load(x_ptr + off_ji, mask=mask_ji)
    y = tl.load(y_ptr + off_j, mask=mask_j)
    dz = tl.load(dz_ptr + off_ji, mask=mask_ji)

    
    df = tl.where(x * y[:, None] > 0, 1.0, 0.0)
    dxy_x = y[:, None]
    
    dx = df * dxy_x * dz

    tl.store(dx_ptr + off_ji, dx, mask=mask_ji)

    return


r


def sum_spec(x: Float32[4, 200]) -> Float32[4,]:
    return x.sum(1)


@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    
    block_id_i = tl.program_id(0)
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    z = tl.zeros([B0], dtype=tl.float32)

    
    
    
    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        z += tl.sum(x, axis=1)

    tl.store(z_ptr + off_i, z, mask=mask_i)

    return


r


def softmax_spec(x: Float32[4, 200]) -> Float32[4, 200]:
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def softmax_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    exp_sum = tl.zeros([B0], dtype=tl.float32)
    x_max = tl.full([B0], -float("inf"), dtype=tl.float32)
    new_x_max = tl.full((B0,), -float("inf"), dtype=tl.float32)

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)

        
        
        new_x_max = tl.maximum(x_max, tl.max(x, axis=1))
        new_exp_x = tl.exp2(log2_e * (x - new_x_max[:, None]))
        factor = tl.exp2(log2_e * (x_max - new_x_max))
        exp_sum = exp_sum * factor + tl.sum(new_exp_x, axis=1)
        x_max = new_x_max

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        exp_x = tl.exp2(log2_e * (x - x_max[:, None]))
        z = exp_x / exp_sum[:, None]
        tl.store(z_ptr + off_ij, z, mask=mask_ij)
    return


@triton.jit
def softmax_kernel_brute_force(
    x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr
):
    
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    exp_sum = tl.zeros([B0], dtype=tl.float32)
    x_max = tl.zeros([B0], dtype=tl.float32)

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        x_max = tl.maximum(x_max, tl.max(x, axis=1))

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        exp_x = tl.exp2(log2_e * (x - x_max[:, None]))
        exp_sum += tl.sum(exp_x, axis=1)

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None] * T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        exp_x = tl.exp2(log2_e * (x - x_max[:, None]))
        z = exp_x / exp_sum[:, None]
        tl.store(z_ptr + off_ij, z, mask=mask_ij)

    return


r


def flashatt_spec(
    q: Float32[200,], k: Float32[200,], v: Float32[200,]
) -> Float32[200,]:
    x = q[:, None] * k[None, :]
    x_max = x.max(1, keepdim=True)[0]
    x = x - x_max
    x_exp = x.exp()
    soft = x_exp / x_exp.sum(1, keepdim=True)
    return (v[None, :] * soft).sum(1)


@triton.jit
def flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    
    

    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    inf = 1.0e6

    
    q = tl.load(q_ptr + off_i, mask=mask_i)

    
    
    

    
    exp_sum = tl.zeros((B0,), dtype=tl.float32)
    
    qk_max = tl.full((B0,), -inf, dtype=tl.float32)
    z = tl.zeros((B0,), dtype=tl.float32)

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]

        k = tl.load(k_ptr + off_j, mask=mask_j)
        qk = q[:, None] * k[None, :] + tl.where(mask_ij, 0, -1.0e6)
        

        
        new_max = tl.maximum(tl.max(qk, axis=1), qk_max)
        qk_exp = tl.exp2(log2_e * (qk - new_max[:, None]))
        
        factor = tl.exp2(log2_e * (qk_max - new_max))
        
        new_exp_sum = exp_sum * factor + tl.sum(qk_exp, axis=1)
        v = tl.load(v_ptr + off_j, mask=mask_j, other=0.0)
        z = z * factor + tl.sum(qk_exp * v[None, :], axis=1)

        qk_max = new_max
        exp_sum = new_exp_sum

    z = z / exp_sum
    tl.store(z_ptr + off_i, z, mask=mask_i)
    return


r


def conv2d_spec(x: Float32[4, 8, 8], k: Float32[4, 4]) -> Float32[4, 8, 8]:
    z = torch.zeros(4, 8, 8)
    x = torch.nn.functional.pad(x, (0, 4, 0, 4, 0, 0), value=0.0)
    
    for i in range(8):
        for j in range(8):
            z[:, i, j] = (k[None, :, :] * x[:, i : i + 4, j : j + 4]).sum(1).sum(1)
    return z


@triton.jit
def conv2d_kernel(
    x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr
):
    block_id_i = tl.program_id(0)
    
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0

    off_h = tl.arange(0, KH)
    off_w = tl.arange(0, KW)
    off_hw = off_h[:, None] * KW + off_w[None, :]

    k = tl.load(k_ptr + off_hw)

    for j in tl.range(0, H):
        for l in tl.range(0, W):
            off_j_oj = j + off_h[None, :, None]
            off_l_ol = l + off_w[None, None, :]
            off_x = off_i * H * W + off_j_oj * W + off_l_ol
            mask_x = (off_j_oj < H) & (off_l_ol < W)
            x = tl.load(x_ptr + off_x, mask=mask_x)

            z = tl.sum(x * k[None, :])
            off_z = off_i * H * W + j * W + l
            tl.store(z_ptr + off_z, z)

    return


r


def dot_spec(x: Float32[4, 32, 64], y: Float32[4, 64, 32]) -> Float32[4, 32, 32]:
    return x @ y


@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    block_id_i = tl.program_id(2)
    
    off_i = block_id_i * B2 + tl.arange(0, B2)
    off_j = block_id_j * B0 + tl.arange(0, B0)
    off_k = block_id_k * B1 + tl.arange(0, B1)

    mask_i = off_i < N2
    mask_j = off_j < N0
    mask_k = off_k < N1

    z = tl.zeros((B2, B0, B1), dtype=tl.float32)
    off_z = (
        off_i[:, None, None] * N0 * N1
        + off_j[None, :, None] * N1
        + off_k[None, None, :]
    )
    mask_z = mask_i[:, None, None] & mask_j[None, :, None] & mask_k[None, None, :]

    for l in tl.range(0, MID, B_MID):
        off_l = l + tl.arange(0, B_MID)
        mask_l = off_l < MID
        off_x = (
            off_i[:, None, None] * N0 * MID
            + off_j[None, :, None] * MID
            + off_l[None, None, :]
        )
        off_y = (
            off_i[:, None, None] * MID * N1
            + off_l[None, :, None] * N1
            + off_k[None, None, :]
        )
        mask_x = mask_i[:, None, None] & mask_j[None, :, None] & mask_l[None, None, :]
        mask_y = mask_i[:, None, None] & mask_l[None, :, None] & mask_k[None, None, :]
        x = tl.load(x_ptr + off_x, mask=mask_x)
        y = tl.load(y_ptr + off_y, mask=mask_y)
        z += tl.dot(x, y)

    tl.store(z_ptr + off_z, z, mask=mask_z)
    return


r

FPINT = 32 // 4
GROUP = 8


def quant_dot_spec(
    scale: Float32[32, 8],
    offset: Int32[32,],
    weight: Int32[32, 8],
    activation: Float32[64, 32],
) -> Float32[32, 32]:
    offset = offset.view(32, 1)

    def extract(x):
        over = torch.arange(8) * 4
        mask = 2**4 - 1
        return (x[..., None] >> over) & mask

    scale = scale[..., None].expand(-1, 8, GROUP).contiguous().view(-1, 64)
    offset = (
        extract(offset)[..., None].expand(-1, 1, 8, GROUP).contiguous().view(-1, 64)
    )
    return (scale * (extract(weight).view(-1, 64) - offset)) @ activation


@triton.jit
def quant_dot_kernel(
    scale_ptr,
    offset_ptr,
    weight_ptr,
    activation_ptr,
    z_ptr,
    N0,
    N1,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    
    off_j = block_id_j * B0 + tl.arange(0, B0)
    off_k = block_id_k * B1 + tl.arange(0, B1)

    mask_j = off_j < N0
    mask_k = off_k < N1

    z = tl.zeros((B0, B1), dtype=tl.float32)
    off_z = off_j[:, None] * N1 + off_k[None, :]
    mask_z = mask_j[:, None] & mask_k[None, :]

    for l in tl.range(0, MID, B_MID):
        
        off_l_div_g = tl.arange(0, B_MID // GROUP) + (l // GROUP)
        mask_l_div_g = off_l_div_g < (MID // GROUP)
        off_scale = off_j[:, None] * (MID // GROUP) + off_l_div_g[None, :]
        
        mask_scale = mask_j[:, None] & mask_l_div_g[None, :]
        scale = tl.load(scale_ptr + off_scale, mask=mask_scale)

        
        
        shift = tl.load(offset_ptr + off_j, mask=mask_j)

        
        
        off_weight_l = l + tl.arange(0, B_MID // FPINT)
        mask_weight_l = off_weight_l < (MID // FPINT)
        off_weight = off_j[:, None] * (MID // FPINT) + off_weight_l[None, :]
        mask_weight = mask_j[:, None] & mask_weight_l[None, :]
        weight = tl.load(weight_ptr + off_weight, mask=mask_weight)

        
        off_l = l + tl.arange(0, B_MID)
        mask_l = off_l < MID
        off_activation = off_l[:, None] * N1 + off_k[None, :]
        mask_activation = mask_l[:, None] & mask_k[None, :]
        activation = tl.load(activation_ptr + off_activation, mask=mask_activation)

        
        BITS = 32 // FPINT
        unpack_offs = tl.arange(0, FPINT) * BITS
        unpack_upperbound_mask = (1 << BITS) - 1
        unpacked_shift = (shift[:, None] >> unpack_offs) & unpack_upperbound_mask
        unpacked_weight = (weight[:, :, None] >> unpack_offs) & unpack_upperbound_mask
        
        
        transformed_weight = scale[:, :, None] * (
            unpacked_weight - unpacked_shift[:, :, None]
        )
        
        transformed_weight = transformed_weight.reshape(
            unpacked_shift.shape[0], unpacked_shift.shape[-1] * FPINT
        )

        
        z += tl.dot(transformed_weight, activation)

    tl.store(z_ptr + off_z, z, mask=mask_z)

    return


def run_demos():
    run_demo1()
    run_demo2()
    run_demo3()
    run_demo4()


def run_puzzles(args, puzzles: List[int]):
    print_log = args.log
    device = args.device

    if 1 in puzzles:
        print("Puzzle 
        ok = test(
            add_kernel,
            add_spec,
            nelem={"N0": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 2 in puzzles:
        print("Puzzle 
        ok = test(
            add_mask2_kernel,
            add2_spec,
            nelem={"N0": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 3 in puzzles:
        print("Puzzle 
        ok = test(
            add_vec_kernel,
            add_vec_spec,
            nelem={"N0": 32, "N1": 32},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 4 in puzzles:
        print("Puzzle 
        ok = test(
            add_vec_block_kernel,
            add_vec_block_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 5 in puzzles:
        print("Puzzle 
        ok = test(
            mul_relu_block_kernel,
            mul_relu_block_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 6 in puzzles:
        print("Puzzle 
        ok = test(
            mul_relu_block_back_kernel,
            mul_relu_block_back_spec,
            nelem={"N0": 100, "N1": 90},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 7 in puzzles:
        print("Puzzle 
        ok = test(
            sum_kernel,
            sum_spec,
            B={"B0": 1, "B1": 32},
            nelem={"N0": 4, "N1": 32, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 8 in puzzles:
        print("Puzzle 
        ok = test(
            softmax_kernel,
            softmax_spec,
            B={"B0": 1, "B1": 32},
            nelem={"N0": 4, "N1": 32, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 9 in puzzles:
        print("Puzzle 
        ok = test(
            flashatt_kernel,
            flashatt_spec,
            B={"B0": 64, "B1": 32},
            nelem={"N0": 200, "T": 200},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 10 in puzzles:
        print("Puzzle 
        ok = test(
            conv2d_kernel,
            conv2d_spec,
            B={"B0": 1},
            nelem={"N0": 4, "H": 8, "W": 8, "KH": 4, "KW": 4},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 11 in puzzles:
        print("Puzzle 
        ok = test(
            dot_kernel,
            dot_spec,
            B={"B0": 16, "B1": 16, "B2": 1, "B_MID": 16},
            nelem={"N0": 32, "N1": 32, "N2": 4, "MID": 64},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    if 12 in puzzles:
        print("Puzzle 
        ok = test(
            quant_dot_kernel,
            quant_dot_spec,
            B={"B0": 16, "B1": 16, "B_MID": 64},
            nelem={"N0": 32, "N1": 32, "MID": 64},
            print_log=print_log,
            device=device,
        )
        print_end_line()
        if not ok:
            return
    print("All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--puzzle", type=int, metavar="N", help="Run Puzzle 
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all Puzzles. Stop at first failure.",
    )
    parser.add_argument("-l", "--log", action="store_true", help="Print log messages.")
    parser.add_argument(
        "-i",
        "--intro",
        action="store_true",
        help="Run all demos in the introduction part.",
    )

    args = parser.parse_args()

    if os.getenv("TRITON_INTERPRET", "0") == "1":
        torch.set_default_device("cpu")
        args.device = "cpu"
    else:  
        torch.set_default_device("cuda")
        args.device = "cuda"

    if args.intro:
        run_demos()
    elif args.all:
        run_puzzles(args, list(range(1, 13)))
    elif args.puzzle:
        run_puzzles(args, [int(args.puzzle)])
    else:
        parser.print_help()
