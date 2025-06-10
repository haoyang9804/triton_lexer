import torch
import triton
import triton.language as tl


torch.manual_seed(456)

N, d = 16, 8

Q = torch.rand((N, d))
K = torch.rand((N, d))
V = torch.rand((N, d))


Br = 4
Bc = d

expected_softmax = torch.softmax(Q @ K.T, dim=1)
expected_attention = expected_softmax @ V


S_mat = Q @ K.T
row_max = torch.max(S_mat, dim=1).values[:, None]

input_safe = S_mat - row_max
softmax_numerator = torch.exp(input_safe)

softmax_denominator = torch.sum(softmax_numerator, dim=1)[:, None]

naive_softmax = softmax_numerator / softmax_denominator

matmul_result = naive_softmax @ V

assert torch.allclose(naive_softmax, expected_softmax)
assert torch.allclose(matmul_result, expected_attention)

S_mat_for_check = torch.zeros((N, N))

Br = 4
Bc = d


attn_score = torch.zeros((N, N))

for block_start_Bc in range(0, N, d):
    block_end_Bc = block_start_Bc + d
    Kj = K[block_start_Bc:block_end_Bc, :]
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        Qi = Q[block_start_Br:block_end_Br, :]

        Sij = Qi @ Kj.T
        attn_score[block_start_Br:block_end_Br, block_start_Bc:block_end_Bc] += Sij


for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc
    Kj = K[block_start_Bc:block_end_Bc, :]
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        Qi = Q[block_start_Br:block_end_Br, :]

        Sij = Qi @ Kj.T
        S_mat_for_check[block_start_Br:block_end_Br, block_start_Bc:block_end_Bc] += Sij

print(f"testing attn_score vs ref")
assert torch.allclose(attn_score, Q @ K.T)


O = torch.zeros((N, d))

for block_start_Bc in range(0, N, d):
    block_end_Bc = block_start_Bc + d
    Kj = K[block_start_Bc:block_end_Bc, :]
    Vj = V[block_start_Bc:block_end_Bc, :]
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        Qi = Q[block_start_Br:block_end_Br, :]

        Sij = Qi @ Kj.T
        Oi = Sij @ Vj
        O[block_start_Br:block_end_Br, :] += Oi

assert torch.allclose(O, (Q @ K.T) @ V)


O = torch.zeros((N, d))
l = torch.zeros((N, 1))
m = torch.full((N, 1), -torch.inf)


for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc

    Kj = K[block_start_Bc:block_end_Bc, :]
    Vj = V[block_start_Bc:block_end_Bc, :]
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br

        mi = m[block_start_Br:block_end_Br, :]
        li = l[block_start_Br:block_end_Br, :]
        Oi = O[block_start_Br:block_end_Br, :]
        Qi = Q[block_start_Br:block_end_Br, :]

        Sij = Qi @ Kj.T

        mij_hat = torch.max(Sij, dim=1).values[:, None]

        pij_hat = torch.exp(Sij - mij_hat)
        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (
            torch.exp(mij_hat - mi_new) * pij_hat / li_new
        ) @ Vj

        m[block_start_Br:block_end_Br, :] = mi_new
        l[block_start_Br:block_end_Br, :] = li_new
        O[block_start_Br:block_end_Br, :] = Oi


assert torch.allclose(O, expected_attention)
print(f"testing first flash attn!")
assert torch.allclose(O, expected_attention)
print(f"Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


O = torch.zeros((N, d))


l = torch.zeros((N, 1))

m = torch.full((N, 1), -torch.inf)

for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc

    Kj = K[block_start_Bc:block_end_Bc, :]
    Vj = V[block_start_Bc:block_end_Bc, :]
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br

        mi = m[block_start_Br:block_end_Br, :]
        li = l[block_start_Br:block_end_Br, :]
        Oi = O[block_start_Br:block_end_Br, :]
        Qi = Q[block_start_Br:block_end_Br, :]

        Sij = Qi @ Kj.T

        mij_hat = torch.max(Sij, dim=1).values[:, None]

        pij_hat = torch.exp(Sij - mij_hat)

        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (
            torch.exp(mij_hat - mi_new) * pij_hat / li_new
        ) @ Vj

        m[block_start_Br:block_end_Br, :] = mi_new
        l[block_start_Br:block_end_Br, :] = li_new

        O[block_start_Br:block_end_Br, :] = Oi

assert torch.allclose(O, expected_attention)
print(f"testing first flash attn!")
assert torch.allclose(O, expected_attention)
print(f"Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
