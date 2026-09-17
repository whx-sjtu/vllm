# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_argmax import argmax_and_pack
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)


def _reference(logits, vocab_start):
    values, indices = logits.max(dim=-1)
    return torch.stack((values.float(), (indices + vocab_start).float()), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(0, 31), (1, 1), (7, 1003), (17, 32768), (33, 32769), (3, 131089)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_argmax_and_pack_matches_torch(dtype, shape, strided):
    """Tail lanes, row/column strides and multi-tile vocabularies preserve IDs."""
    rows, vocab = shape
    stride = 2 if strided else 1
    logits = torch.randn(rows, vocab * stride, device="cuda", dtype=dtype)[:, ::stride]
    torch.testing.assert_close(
        argmax_and_pack(logits, 128_256),
        _reference(logits, 128_256),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("vocab_start", [0, 2**24 - 1, 2**31 + 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_argmax_and_pack_ties_and_nonfinite_values(vocab_start, dtype):
    """First NaN/maximum wins, including ties across tiles and all -inf rows."""
    logits = torch.full((5, 65541), -float("inf"), device="cuda", dtype=dtype)
    logits[0, [3, 32771]] = 7
    logits[1, [2, 32770]] = float("inf")
    logits[2, [5, 32769]] = float("nan")
    logits[3, 1] = float("inf")
    logits[3, 32773] = float("nan")
    torch.testing.assert_close(
        argmax_and_pack(logits, vocab_start),
        _reference(logits, vocab_start),
        atol=0,
        rtol=0,
        equal_nan=True,
    )


def test_argmax_and_pack_compile_and_graph_replay():
    """Replay must read updated logits, including a changed winning token."""
    logits = torch.randn(3, 8197, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(argmax_and_pack, fullgraph=True)
    compiled(logits, 1000)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(logits, 1000)
    for column in (1, 8196):
        logits.zero_()
        logits[:, column] = 1
        graph.replay()
        torch.testing.assert_close(actual, _reference(logits, 1000), atol=0, rtol=0)
