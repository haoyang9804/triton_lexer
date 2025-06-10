import torch

import triton_kernels as tk
from triton_kernels.flux import DoubleStreamBlock

hidden_size = 3072
num_heads = 24
mlp_ratio = 4.0
head_dim = hidden_size // num_heads

batch_size = 3
img_seq_len = 4080
txt_seq_len = 256
seq_len = img_seq_len + txt_seq_len

device = "cuda"

block = DoubleStreamBlock(
    hidden_size=hidden_size,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
)
block_triton = tk.DoubleStreamBlock(
    hidden_size=hidden_size,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
)
block_triton.load_state_dict(block.state_dict())
block = block.to(device)
block_triton = block_triton.to(device)

img = torch.randn([batch_size, img_seq_len, hidden_size], device=device)
txt = torch.randn([batch_size, txt_seq_len, hidden_size], device=device)
vec = torch.randn([batch_size, hidden_size], device=device)
pe = torch.randn([1, 1, seq_len, head_dim // 2, 2, 2], device=device)

out = block(img=img, txt=txt, vec=vec, pe=pe)
out_triton = block_triton(img=img, txt=txt, vec=vec, pe=pe)

torch.testing.assert_close(out, out_triton, atol=1e-5, rtol=0)


warmup_count = 5

for i in range(warmup_count):
    out = block(img=img, txt=txt, vec=vec, pe=pe)
for i in range(warmup_count):
    out_triton = block_triton(img=img, txt=txt, vec=vec, pe=pe)


run_count = 100

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for i in range(run_count):
    out = block(img=img, txt=txt, vec=vec, pe=pe)
end.record()
torch.cuda.synchronize()
print(f"baseline block time: {start.elapsed_time(end):.2f} ms")

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for i in range(run_count):
    out_triton = block_triton(img=img, txt=txt, vec=vec, pe=pe)
end.record()
torch.cuda.synchronize()
print(f"  triton block time: {start.elapsed_time(end):.2f} ms")
