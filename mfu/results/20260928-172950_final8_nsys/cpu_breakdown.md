# CPU-side breakdown (8 ranks x 3 captured steps; values per rank per step)

## CUDA runtime API

| call | calls | ms |
|---|---|---|
| cudaStreamSynchronize_v3020 | 40 | 151.7 |
| cudaLaunchKernel_v7000 | 4345 | 117.9 |
| cuLaunchKernelEx | 1209 | 34.4 |
| cuLaunchKernel | 486 | 10.8 |
| cudaMemcpyAsync_v3020 | 420 | 6.8 |
| cudaMemsetAsync_v3020 | 375 | 5.9 |
| cudaFree_v3020 | 4 | 5.2 |
| cudaProfilerStop_v4000 | 0 | 5.1 |
| cudaLaunchKernelExC_v11060 | 391 | 3.0 |
| cudaMalloc_v3020 | 8 | 2.7 |
| cudaEventQuery_v3020 | 689 | 1.6 |
| cudaStreamWaitEvent_v3020 | 1106 | 1.4 |

Mean train_step CPU range: 619 ms

## Host time inside NVTX ranges

| range | calls | CPU ms | median call ms | calls > 20 ms | CPU ms in those |
|---|---|---|---|---|---|
| fwd_bwd_mb0 | 1.0 | 606.5 | 613.62 | 1.0 | 606.5 |
| grad_clip | 1.0 | 116.8 | 117.01 | 1.0 | 116.8 |
| optimizer_step | 1.0 | 8.0 | 7.82 | 0.0 | 0.0 |
| mixer: NemotronV3Attention | 12.0 | 63.1 | 2.20 | 0.1 | 6.4 |
| fused_attention: FusedAttention | 12.0 | 13.7 | 0.28 | 0.1 | 6.3 |
| mixer: NemotronV3Mamba2Mixer | 46.0 | 216.0 | 2.92 | 0.6 | 18.8 |
| mixer: MoE | 46.0 | 189.7 | 3.51 | 0.0 | 0.0 |
| experts: GroupedExpertsDeepEP | 46.0 | 116.0 | 1.58 | 0.0 | 0.0 |
| shared_experts: MLP | 46.0 | 40.2 | 0.85 | 0.0 | 0.0 |

## Ranges with calls > 20 ms (count per rank per step)

- `FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM`: 1.0
- `model: FSDPNemotronV3Model`: 1.0
- `grad_clip`: 1.0
- `25: FSDPNemotronV3Block`: 0.6
- `mixer: NemotronV3Mamba2Mixer`: 0.6
- `42: FSDPNemotronV3Block`: 0.1
- `mixer: NemotronV3Attention`: 0.1
- `attn_module: DotProductAttention`: 0.1
- `fused_attention: FusedAttention`: 0.1
