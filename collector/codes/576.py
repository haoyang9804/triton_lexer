import torch
import triton.language as tl
import triton


def cdiv(x, y):
    return (x + y - 1) // y


res = cdiv(11, 5)
print(res)

tres = cdiv(22, 5)
print(tres)


M = 1024
N = 768
K = 128

print(f"C [{M}x{N}] = A [{M}x{K}] * B [{K}x{N}]")

a = torch.rand(M, K)
b = torch.rand(K, N)
c = torch.rand(M, N)


BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32


GROUP_SIZE_M = 2


num_pid_m = cdiv(M, BLOCK_SIZE_M)
num_pid_n = cdiv(N, BLOCK_SIZE_N)

print("num_pid_m:", num_pid_m)
print("num_pid_n:", num_pid_n)
print("num_pid_n * GROUP_SIZE_M =", num_pid_n * GROUP_SIZE_M)

num_pid_m = cdiv(M, BLOCK_SIZE_M)
num_pid_n = cdiv(N, BLOCK_SIZE_N)
print(f"{num_pid_m=}, {num_pid_n=}")

nb_programs = num_pid_m * num_pid_n
print("nb programs to launch:", nb_programs)

num_pid_in_group = GROUP_SIZE_M * num_pid_n
assert num_pid_n * GROUP_SIZE_M <= M
print(num_pid_in_group)
