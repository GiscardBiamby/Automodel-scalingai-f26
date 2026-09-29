# 8-GPU round 2: re-testing the 6-GPU findings on all 8 GPUs

All 8 GPUs free (2026-09-28 16:15 UTC). EP=8, so experts are *not* FSDP-sharded (unlike 6 GPUs), and the
round-1 8-GPU profile was host/DeepEP-bound rather than FSDP-bound. Cumulative ladder: every row adds one change to
the row above; 30-step runs, medians of steps 5+. Useful MFU = real tokens/s/GPU × 19.36 GFLOP / 989 TFLOP/s.

| run | change | tok/s/GPU | useful MFU | step (s) | peak mem | Δ |
|---|---|---|---|---|---|---|
| round-1 best (E) | packing + fused CE + 2 packs/GPU + GC (40 steps) | 11,836 | 23.2% | 0.672 | 56.8 GiB | reference |
| e8_r0_best | + fused Adam (current `configs/nemotron_nano_v3_squad_best.yaml`) | 12,425 | 24.3% | 0.641 | 56.8 GiB | +5.0% |
| e8_r1_bf16rs | + bf16 gradient reduce-scatter | 12,666 | 24.8% | 0.627 | 56.8 GiB | +1.9% |
| e8_r2_rmsnorm | + TE RMSNorm | 12,852 | 25.2% | 0.618 | 56.8 GiB | +1.5% |
| e8_r3_sms64 | + DeepEP 64 SMs | 14,238 | 27.9% | 0.559 | 56.8 GiB | +10.8% (DeepEP ~22% of the step at EP=8) |
| e8_r3b_sms96 | DeepEP 96 SMs instead | 14,318 | 28.0% | 0.556 | 56.8 GiB | +0.6% vs 64 |
| e8_r4_fp8 | + FP8 on TE linears (attention + shared experts, ~18% of GEMM FLOPs; at 64 SMs) | 14,330 | 28.1% | 0.555 | 56.1 GiB | +0.6% vs r3 (near noise: GEMMs are a smaller share at 8 GPUs) |

Batch size / activation checkpointing (base = e8_r2 + DeepEP 96 SMs + FP8 dense linears):

| run | change | tok/s/GPU | useful MFU | step (s) | peak mem | Δ vs e8_b1 |
|---|---|---|---|---|---|---|
| e8_b1_base | base, 2 packs/GPU (GBS 16) | 14,395 | 28.2% | 0.553 | 56.1 GiB | reference |
| e8_b2_lbs4 | 4 packs/GPU, no AC | OOM | | | | |
| e8_b3_lbs4_ac | 4 packs + full AC | 12,698 | 24.9% | 1.255 | 37.9 GiB | −11.8% |
| e8_b4_lbs8_ac | 8 packs + full AC | 13,427 | 26.3% | 2.370 | 47.8 GiB | −6.7% |
| e8_b5_lbs3 | 3 packs/GPU, no AC (GBS 24) | **15,172** | **29.7%** | 0.788 | 70.5 GiB | **+5.4%, adopted** |
| e8_b6_lbs4_acmoe | 4 packs + AC on the 23 MoE blocks only | 13,554 | 26.5% | 1.174 | 51.7 GiB | −5.8% |

Opposite of 6 GPUs: at EP=8 there is no per-micro-batch expert all-gather to amortise, so the checkpointing
recompute is pure overhead; the best batch is the largest that fits *without* AC (3 packs).

Adopted in `configs/nemotron_nano_v3_squad_best.yaml`: e8_b5 = 15,172 tok/s/GPU, 29.7% useful MFU (13.5× the
shipped recipe, +28% over round-1 config E).

Row length (packing size), on the final config (bf16 RS, TE norm, 96 DeepEP SMs, FP8 TE linears):

| run | rows x length | tokens/GPU/micro-batch | tok/s/GPU | useful MFU | pad eff. | peak mem | note |
|---|---|---|---|---|---|---|---|
| e8_b1_base | 2 x 4096 | 8,192 | 14,395 | 28.2% | 0.970 | 56.1 GiB | |
| e8_pack8192_x1 | 1 x 8192 | 8,192 | 14,378 | 28.1% | 0.985 | 56.1 GiB | same tokens/GPU as 2x4096: same speed despite +1.5 pt fill |
| e8_b5_lbs3 | 3 x 4096 | 12,288 | 15,172 | 29.7% | 0.971 | 70.5 GiB | best; not expressible with 8192-token rows |
| e8_pack8192_x2 | 2 x 8192 | 16,384 | OOM | | | | same as 4 x 4096 (OOM) |

Throughput is set by tokens per GPU per micro-batch (bounded by memory), not by row length; 4096-token rows give the
finer granularity needed to reach the 12,288-token optimum.
