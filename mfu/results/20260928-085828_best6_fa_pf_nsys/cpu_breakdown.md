# CPU-side breakdown (6 ranks x 4 captured steps; values per rank per step)

## CUDA runtime API

| call | calls | ms |
|---|---|---|
| cudaFree_v3020 | 46 | 224.3 |
| cudaStreamSynchronize_v3020 | 41 | 222.7 |
| cudaMalloc_v3020 | 48 | 80.3 |
| cudaLaunchKernel_v7000 | 4664 | 40.6 |
| cuLaunchKernelEx | 1360 | 14.7 |
| cudaMemcpyAsync_v3020 | 566 | 8.0 |
| cuLaunchKernel | 276 | 3.9 |
| cudaMemsetAsync_v3020 | 306 | 3.4 |
| cudaLaunchKernelExC_v11060 | 414 | 3.4 |
| cudaEventQuery_v3020 | 1336 | 3.3 |
| cudaProfilerStop_v4000 | 0 | 2.9 |
| cudaStreamWaitEvent_v3020 | 1968 | 2.4 |

Mean train_step CPU range: 999 ms

## Host time inside NVTX ranges

| range | calls | CPU ms | median call ms | calls > 20 ms | CPU ms in those |
|---|---|---|---|---|---|
| fwd_bwd_mb0 | 1.0 | 922.1 | 918.04 | 1.0 | 922.1 |
| grad_clip | 1.0 | 178.1 | 193.09 | 1.0 | 178.1 |
| optimizer_step | 1.0 | 8.4 | 8.28 | 0.0 | 0.0 |
| mixer: NemotronV3Attention | 12.0 | 18.1 | 1.27 | 0.0 | 1.0 |
| fused_attention: FusedAttention | 12.0 | 4.5 | 0.29 | 0.0 | 0.0 |
| mixer: NemotronV3Mamba2Mixer | 46.0 | 148.1 | 3.43 | 0.0 | 0.0 |
| mixer: MoE | 46.0 | 548.0 | 6.06 | 3.7 | 289.8 |
| shared_experts: MLP | 46.0 | 24.0 | 0.52 | 0.0 | 0.0 |

## Ranges with calls > 20 ms (count per rank per step)

- `mixer: MoE`: 3.7
- `51: FSDPNemotronV3Block`: 2.0
- `FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM`: 1.0
- `model: FSDPNemotronV3Model`: 1.0
- `grad_clip`: 1.0
- `49: FSDPNemotronV3Block`: 0.7
- `experts: FSDPGroupedExpertsDeepEP`: 0.6
- `31: FSDPNemotronV3Block`: 0.4
- `29: FSDPNemotronV3Block`: 0.2
- `34: FSDPNemotronV3Block`: 0.1
- `27: FSDPNemotronV3Block`: 0.1
- `10: FSDPNemotronV3Block`: 0.1
