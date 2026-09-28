"""Microbenchmark: cost of a *new* sequence length vs a repeated one in TE DotProductAttention (cuDNN backend).

Mirrors the Nemotron-Nano-V3 attention layer (32 q heads, 2 kv heads, head_dim 128, causal + padding mask,
bshd, bf16, fwd+bwd) at the baseline micro-batch size (8) and SQuAD-like padded lengths. For each length it
times the first call (cache miss) and the mean of the next calls (cache hit).

Usage: mfu/docker.sh -- env CUDA_VISIBLE_DEVICES=0 python mfu/bench_te_attn_shapes.py
"""

import json
import time

import torch
import transformer_engine.pytorch as te

B, HQ, HKV, D = 8, 32, 2, 128


def run(attn, s: int) -> None:
    q = torch.randn(B, s, HQ, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, s, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, s, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Right-padding mask like default_collater produces: True = masked-out position.
    lens = torch.randint(s // 2, s + 1, (B,), device="cuda")
    lens[0] = s
    pad = torch.arange(s, device="cuda")[None, :] >= lens[:, None]
    mask = pad[:, None, None, :]
    out = attn(q, k, v, attention_mask=mask, attn_mask_type="padding_causal")
    out.sum().backward()


def timed(fn) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


if __name__ == "__main__":
    attn = te.DotProductAttention(
        num_attention_heads=HQ, kv_channels=D, num_gqa_groups=HKV, attn_mask_type="padding_causal", qkv_format="bshd"
    )
    run(attn, 128)  # global warmup (library init)
    results = []
    for s in [193, 257, 311, 384, 402, 449, 512]:
        first = timed(lambda: run(attn, s))
        repeat = sum(timed(lambda: run(attn, s)) for _ in range(5)) / 5
        results.append({"seq_len": s, "first_call_ms": round(first, 1), "repeat_call_ms": round(repeat, 2)})
        print(results[-1], flush=True)
    print(json.dumps(results))
