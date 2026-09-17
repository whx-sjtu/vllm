# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TP candidate packing on CUDA or ROCm, excluding GEMM/communication."""

import argparse
import itertools
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_argmax import argmax_and_pack
from vllm.triton_utils import triton


def torch_argmax_and_pack(logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    values, indices = logits.max(dim=-1)
    return torch.stack((values.float(), (indices + vocab_start).float()), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256, 1024]
    )
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[8192, 32768, 65536, 131072]
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    results = []
    for batch, vocab in itertools.product(args.batch_sizes, args.vocab_sizes):
        logits = torch.randn(
            batch, vocab, dtype=getattr(torch, args.dtype), device="cuda"
        )
        baseline = partial(torch_argmax_and_pack, logits, vocab)
        fused = partial(argmax_and_pack, logits, vocab)
        torch.testing.assert_close(fused(), baseline(), atol=0, rtol=0)
        for mode in args.modes:
            bench = (
                partial(triton.testing.do_bench_cudagraph, rep=20)
                if mode == "graph"
                else partial(triton.testing.do_bench, rep=100)
            )
            timings = {"baseline": [], "fused": []}
            implementations = [("baseline", baseline), ("fused", fused)]
            for repeat in range(args.repeats):
                order = implementations if repeat % 2 == 0 else implementations[::-1]
                for name, fn in order:
                    timings[name].append(1000 * bench(fn))
            baseline_us = statistics.median(timings["baseline"])
            fused_us = statistics.median(timings["fused"])
            result = dict(
                batch=batch,
                local_vocab=vocab,
                dtype=args.dtype,
                mode=mode,
                baseline_us=baseline_us,
                fused_us=fused_us,
                speedup=baseline_us / fused_us,
                timings_us=timings,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(
            json.dumps(
                dict(
                    device=torch.cuda.get_device_name(),
                    torch=torch.__version__,
                    triton=triton.__version__,
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
