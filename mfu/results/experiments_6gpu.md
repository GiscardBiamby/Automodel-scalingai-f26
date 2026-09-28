# 6-GPU experiment log (host GPUs 2-7)

GPUs 0-1 left to groupmates. 30-step runs unless noted; tokens/s/GPU and useful MFU are medians of steps 5+.
Useful MFU = real tokens/s/GPU × 19.36 GFLOP / 989 TFLOP/s.

## Enablers (framework fixes needed to run on 6 GPUs at all)

* `ep_size` must divide both 6 and 128 → 2, so experts are also FSDP-sharded 3 ways (`ep_shard=3`).
* FSDP2 cannot shard unevenly on dims ≠ 0, and the expert FFN width 1856 = 2⁶·29 is not divisible by 3:
  `_moe_shard_placement` now picks the first evenly divisible dim (2688), and the HF checkpoint loader labels the
  loaded expert DTensors with the same dim (commits `475923eb`, `a11eba86`).
* `reshard_after_forward: true`: with FSDP-sharded experts the MoE default (keep gathered after forward) holds
  ~1.3 GB × 23 layers of expert weights and OOMs.

## Results

| run | change vs row above / reference | tok/s/GPU | useful MFU | step (s) | peak mem | verdict |
|---|---|---|---|---|---|---|
| baseline_6gpu | shipped recipe (pad-to-longest), local batch 4 (8 OOMs), GBS 240 | 476 | 0.93% | 17.57 | 58.6 GiB | reference |
| best_6gpu | packing + fused CE + 2 packs/GPU + GC (8-GPU best, re-sized) | 6,668 | 13.05% | 1.19 | 64.1 GiB | 14.0× baseline |
| best6_asyncep | + DeepEP `dispatcher_async_dispatch` | 6,286 | 12.31% | 1.26 | 64.1 GiB | −5.7%, rejected |
| best6_prefetch | + explicit FSDP2 prefetch (new, `36e68a48`) | 6,665 | 13.05% | 1.20 | 65.3 GiB | ±0 (overlap 3.7%→23% but GPU-bound elsewhere) |
| best6_fusedadam | + `optimizer.fused: true` | 6,896 | 13.50% | 1.15 | 64.1 GiB | +3.4%, adopted |
| best6_lbs3 / lbs4 | 3 / 4 packs per GPU | OOM | | | | need activation memory |
| best6_lbs3_sac | 3 packs + selective AC | 9,222 | 18.05% | 1.29 | 53.9 GiB | +34% |
| best6_lbs4_sac | 4 packs + selective AC | 9,333 | 18.27% | 1.70 | 60.3 GiB | +35% |
| best6_lbs4_fullac | 4 packs + full AC | 10,245 | 20.06% | 1.55 | 46.9 GiB | +49% |
| best6_lbs6_fullac | 6 packs + full AC (GBS 36) | 11,696 | 22.90% | 2.04 | 49.1 GiB | +70% |
| best6_lbs8_fullac | 8 packs + full AC (GBS 48) | 12,510 | 24.49% | 2.54 | 52.0 GiB | +81%, adopted (> 8-GPU best per GPU) |

## Profile of best6 + fused Adam + prefetch (1,172 ms/step, nsys steps 20-23)

GPU busy 90% (host no longer the limiter on 6 GPUs). Communication 52% of the step (23% overlapped, 29% exposed):
all-gather 299 ms + reduce-scatter 275 ms, i.e. FSDP traffic for the 3-way-sharded experts (~59 GB gathered and
~59 GB of fp32 gradients reduced per GPU per step). FSDP2 copy-in/out kernels 241 ms (21%). GEMM 271 ms (23%),
DeepEP 87 ms (7%, down from 22% at EP=8), optimizer 77 ms. Standalone NCCL on the same GPUs reaches 204 GB/s
(bf16 AG) / 336 GB/s (fp32 RS) bus bandwidth, so the collectives are volume-bound, not misconfigured.

Diagnosis: expert weight traffic is paid per micro-batch and does not depend on the number of tokens in it, so
more tokens per micro-batch amortise it; activation memory was the blocker → activation checkpointing trades a
recompute (~1/3 extra forward) for 2× tokens per all-gather, a net +49%.
