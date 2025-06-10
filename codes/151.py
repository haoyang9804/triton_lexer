from __future__ import annotations

from collections import Counter
import math

import torch
import triton
import numpy as np
from e3nn import o3

__all__ = [
    "pad_tensor_to_power",
    "calculate_lastdim_num_blocks",
    "spherical_harmonics_irreps",
    "num_irreps_projections",
    "separate_embedding_irreps",
]


def num_irreps_projections(l: int) -> int:

    return 2 * l + 1


def pad_tensor_to_power(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    num_nodes = input_tensor.size(0)
    pad_size = triton.next_power_of_2(num_nodes)
    num_pad = pad_size - num_nodes

    zero_pad = torch.zeros(
        (num_pad, *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    joint_tensor = torch.cat([input_tensor, zero_pad], dim=0)
    mask = torch.ones(pad_size, device=joint_tensor.device, dtype=torch.bool)
    mask[num_nodes:] = False
    return (joint_tensor, mask)


def calculate_lastdim_num_blocks(input_tensor: torch.Tensor, block_size: int) -> int:

    stride = input_tensor.stride(-2)
    numel = input_tensor.numel()
    total_blocks = math.ceil(numel / stride)
    return total_blocks


def unravel_index(tensor: torch.Tensor, index: int) -> tuple[int, ...]:

    assert 0 <= index < tensor.numel()
    indices = []
    for size in reversed(tensor.shape):
        indices.append(index % size)
        index //= size
    return tuple(reversed(indices))


def spherical_harmonics_irreps(l_values: list[int], num_feat: int = 1) -> o3.Irreps:

    assert num_feat > 0, "Number of features must be positive!"
    joint = []
    for l in sorted(l_values):
        parity = "e" if (-1) ** l > 0 else "o"
        joint.append(f"{num_feat}x{l}{parity}")
    return o3.Irreps("+".join(joint))


def separate_embedding_irreps(
    embeddings: torch.Tensor | np.ndarray, irreps: o3.Irreps, return_numpy: bool = True
) -> dict[int, torch.Tensor]:

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu()
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    irrep_dims = dict(Counter(irreps.ls))
    splits = np.cumsum(list(irrep_dims.values())).tolist()
    return_dict = {}
    chunks = torch.tensor_split(embeddings, splits, dim=-1)

    for key, chunk in zip(irrep_dims.keys(), chunks):
        if return_numpy:
            chunk = chunk.numpy()
        return_dict[key] = chunk
    return return_dict
