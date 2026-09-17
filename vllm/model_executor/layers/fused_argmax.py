# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _argmax_combine(left_value, left_index, right_value, right_index):
    # Match torch.max: the first NaN wins, otherwise the first maximum wins.
    take_left = (
        (left_value > right_value)
        | ((left_value == right_value) & (left_index < right_index))
        | (
            (left_value != left_value)
            & ((right_value == right_value) | (left_index < right_index))
        )
    )
    return (
        tl.where(take_left, left_value, right_value),
        tl.where(take_left, left_index, right_index),
    )


@triton.jit
def _argmax_and_pack_kernel(
    logits_ptr,
    packed_ptr,
    vocab_start,
    vocab_size: tl.constexpr,
    row_stride: tl.constexpr,
    col_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    best_values = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    best_indices = tl.full((BLOCK_SIZE,), 0x7FFFFFFF, tl.int32)
    for start in range(tl.cdiv(vocab_size, BLOCK_SIZE)):
        cols = start * BLOCK_SIZE + offsets
        values = tl.load(
            logits_ptr + row * row_stride + cols.to(tl.int64) * col_stride,
            mask=cols < vocab_size,
            other=-float("inf"),
        ).to(tl.float32)
        indices = tl.where(cols < vocab_size, cols, 0x7FFFFFFF)
        best_values, best_indices = _argmax_combine(
            best_values, best_indices, values, indices
        )
    value, index = tl.reduce((best_values, best_indices), 0, _argmax_combine)
    global_index = index.to(tl.int64) + vocab_start
    tl.store(packed_ptr + row * 2, value)
    tl.store(packed_ptr + row * 2 + 1, global_index.to(tl.float32))


def argmax_and_pack(logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    """Pack each row's maximum and global token ID into an FP32 TP candidate.

    Input logits must be a 2D FP16, BF16 or FP32 accelerator tensor with a
    nonempty vocabulary. Padding must already be masked. As in the unfused
    packing, token IDs are converted to FP32 after adding the shard offset.
    """
    num_rows, vocab_size = logits.shape
    packed = torch.empty((num_rows, 2), dtype=torch.float32, device=logits.device)
    if num_rows == 0:
        return packed
    # Use more threads per row when there are few rows to process in parallel.
    max_block_size = 32768 if num_rows <= 32 else 8192
    block_size = min(triton.next_power_of_2(vocab_size), max_block_size)
    _argmax_and_pack_kernel[(num_rows,)](
        logits,
        packed,
        vocab_start,
        vocab_size,
        logits.stride(0),
        logits.stride(1),
        block_size,
        num_warps=max(4, block_size // 2048),
    )
    return packed
