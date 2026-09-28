# CPU-side breakdown (6 ranks x 3 captured steps; values per rank per step)

## CUDA runtime API

| call | calls | ms |
|---|---|---|
| cudaLaunchKernel_v7000 | 5561 | 535.0 |
| cudaStreamSynchronize_v3020 | 41 | 391.2 |
| cuLaunchKernelEx | 1752 | 254.2 |
| cudaMemsetAsync_v3020 | 422 | 98.5 |
| cudaMemcpyAsync_v3020 | 680 | 42.4 |
| cuLaunchKernel | 311 | 37.1 |
| cudaLaunchKernelExC_v11060 | 483 | 27.8 |
| cudaEventQuery_v3020 | 1840 | 3.7 |
| cudaProfilerStop_v4000 | 0 | 3.4 |
| cudaStreamWaitEvent_v3020 | 2060 | 2.5 |
| cudaEventRecordWithFlags_v11010 | 1568 | 2.4 |
| cudaStreamIsCapturing_v10000 | 3717 | 1.5 |

Mean train_step CPU range: 2012 ms

## Host time inside NVTX ranges

| range | calls | CPU ms | median call ms | calls > 20 ms | CPU ms in those |
|---|---|---|---|---|---|
| fwd_bwd_mb0 | 1.0 | 2049.2 | 2048.47 | 1.0 | 2049.2 |
| grad_clip | 1.0 | 341.9 | 341.78 | 1.0 | 341.9 |
| optimizer_step | 1.0 | 8.8 | 8.57 | 0.0 | 0.0 |
| mixer: NemotronV3Attention | 6.0 | 11.7 | 1.80 | 0.0 | 0.0 |
| fused_attention: FusedAttention | 18.0 | 25.6 | 0.52 | 0.0 | 0.0 |
| mixer: NemotronV3Mamba2Mixer | 23.0 | 71.9 | 3.06 | 0.0 | 0.0 |
| mixer: MoE | 23.0 | 456.0 | 20.11 | 13.2 | 272.5 |
| shared_experts: MLP | 69.0 | 116.7 | 0.78 | 0.0 | 0.0 |

## Ranges with calls > 20 ms (count per rank per step)

- `_checkpoint_wrapped_module: NemotronV3Block`: 20.5
- `mixer: MoE`: 13.2
- `10: FSDPCheckpointWrapper`: 2.0
- `13: FSDPCheckpointWrapper`: 2.0
- `15: FSDPCheckpointWrapper`: 2.0
- `17: FSDPCheckpointWrapper`: 2.0
- `20: FSDPCheckpointWrapper`: 2.0
- `24: FSDPCheckpointWrapper`: 2.0
- `27: FSDPCheckpointWrapper`: 2.0
- `29: FSDPCheckpointWrapper`: 2.0
- `31: FSDPCheckpointWrapper`: 2.0
- `34: FSDPCheckpointWrapper`: 2.0
