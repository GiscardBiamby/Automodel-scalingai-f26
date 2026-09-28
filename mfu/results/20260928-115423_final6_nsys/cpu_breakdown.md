# CPU-side breakdown (6 ranks x 3 captured steps; values per rank per step)

## CUDA runtime API

| call | calls | ms |
|---|---|---|
| cudaLaunchKernel_v7000 | 5123 | 530.4 |
| cudaStreamSynchronize_v3020 | 41 | 395.1 |
| cuLaunchKernelEx | 1728 | 146.9 |
| cuLaunchKernel | 345 | 93.2 |
| cudaMemsetAsync_v3020 | 398 | 59.4 |
| cudaMemcpyAsync_v3020 | 680 | 50.9 |
| cudaLaunchKernelExC_v11060 | 483 | 23.9 |
| cudaProfilerStop_v4000 | 0 | 11.4 |
| cudaEventQuery_v3020 | 1825 | 3.7 |
| cudaStreamWaitEvent_v3020 | 2060 | 2.5 |
| cudaEventRecordWithFlags_v11010 | 1568 | 2.1 |
| cudaEventRecord_v3020 | 1218 | 1.5 |

Mean train_step CPU range: 2033 ms

## Host time inside NVTX ranges

| range | calls | CPU ms | median call ms | calls > 20 ms | CPU ms in those |
|---|---|---|---|---|---|
| fwd_bwd_mb0 | 1.0 | 1842.8 | 1840.98 | 1.0 | 1842.8 |
| grad_clip | 1.0 | 349.9 | 350.27 | 1.0 | 349.9 |
| optimizer_step | 1.0 | 8.4 | 8.39 | 0.0 | 0.0 |
| mixer: NemotronV3Attention | 12.0 | 128.3 | 1.87 | 5.0 | 118.6 |
| fused_attention: FusedAttention | 12.0 | 65.5 | 0.34 | 0.0 | 0.0 |
| mixer: NemotronV3Mamba2Mixer | 23.0 | 70.8 | 3.01 | 0.0 | 0.0 |
| mixer: MoE | 23.0 | 390.7 | 17.26 | 0.0 | 0.0 |
| shared_experts: MLP | 69.0 | 192.3 | 0.68 | 0.0 | 0.0 |

## Ranges with calls > 20 ms (count per rank per step)

- `mixer: NemotronV3Attention`: 5.0
- `FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM`: 1.0
- `model: FSDPNemotronV3Model`: 1.0
- `37: FSDPCheckpointWrapper`: 1.0
- `36: FSDPCheckpointWrapper`: 1.0
- `35: FSDPCheckpointWrapper`: 1.0
- `34: FSDPCheckpointWrapper`: 1.0
- `33: FSDPNemotronV3Block`: 1.0
- `31: FSDPCheckpointWrapper`: 1.0
- `29: FSDPCheckpointWrapper`: 1.0
- `28: FSDPCheckpointWrapper`: 1.0
- `27: FSDPCheckpointWrapper`: 1.0
