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
| e8_r4_fp8 | + FP8 dense linears (at 64 SMs) | 14,330 | 28.1% | 0.555 | 56.1 GiB | +0.6% vs r3 (near noise: GEMMs are a smaller share at 8 GPUs) |
