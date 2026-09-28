# CPU-side breakdown (8 ranks x 6 captured steps; values per rank per step)

## CUDA runtime API

| call | calls | ms |
|---|---|---|
| cudaStreamSynchronize_v3020 | 51 | 85.6 |
| cudaLaunchKernel_v7000 | 8230 | 67.8 |
| cuLaunchKernelEx | 2405 | 16.3 |
| cudaMemcpyAsync_v3020 | 803 | 11.7 |
| cudaLaunchKernelExC_v11060 | 828 | 5.9 |
| cudaProfilerStop_v4000 | 0 | 5.7 |
| cudaMemsetAsync_v3020 | 618 | 4.0 |
| cuLaunchKernel | 552 | 3.2 |
| cudaEventQuery_v3020 | 1152 | 3.0 |
| cudaStreamWaitEvent_v3020 | 2196 | 2.8 |
| cudaEventRecordWithFlags_v11010 | 1572 | 2.2 |
| cudaEventRecord_v3020 | 1420 | 1.8 |

Mean train_step CPU range: 884 ms

## Host time inside NVTX ranges

| range | calls | CPU ms | median call ms | calls > 20 ms | CPU ms in those |
|---|---|---|---|---|---|
| fwd_bwd_mb0 | 1.0 | 408.2 | 408.48 | 1.0 | 408.2 |
| fwd_bwd_mb1 | 1.0 | 429.7 | 429.24 | 1.0 | 429.7 |
| grad_clip | 1.0 | 34.9 | 35.23 | 1.0 | 34.7 |
| optimizer_step | 1.0 | 41.5 | 41.50 | 1.0 | 41.5 |
| mixer: NemotronV3Attention | 24.0 | 31.2 | 1.26 | 0.0 | 0.0 |
| fused_attention: FusedAttention | 24.0 | 6.6 | 0.27 | 0.0 | 0.0 |
| mixer: NemotronV3Mamba2Mixer | 92.0 | 245.2 | 2.90 | 0.0 | 0.0 |
| mixer: MoE | 92.0 | 256.2 | 2.56 | 0.0 | 0.0 |
| experts: GroupedExpertsDeepEP | 92.0 | 155.4 | 1.39 | 0.0 | 0.0 |
| shared_experts: MLP | 92.0 | 46.7 | 0.50 | 0.0 | 0.0 |

## Ranges with calls > 20 ms (count per rank per step)

- `FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM`: 2.0
- `model: FSDPNemotronV3Model`: 2.0
- `optimizer_step`: 1.0
- `grad_clip`: 1.0
