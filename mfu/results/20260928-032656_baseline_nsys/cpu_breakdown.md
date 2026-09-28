# CPU-side breakdown (8 ranks x 3 captured steps; values per rank per step)

## CUDA runtime API

| call | calls | ms |
|---|---|---|
| cudaLaunchKernel_v7000 | 14897 | 273.3 |
| cuLaunchKernelEx | 4799 | 153.1 |
| cudaLaunchKernelExC_v11060 | 1632 | 127.8 |
| cudaStreamSynchronize_v3020 | 27 | 72.3 |
| cudaGetDeviceProperties_v3020 | 14 | 48.0 |
| cudaMemcpyAsync_v3020 | 1507 | 24.8 |
| cuLaunchKernel | 1080 | 19.7 |
| cudaProfilerStop_v4000 | 0 | 15.5 |
| cudaEventQuery_v3020 | 2358 | 6.1 |
| cudaStreamWaitEvent_v3020 | 4376 | 5.5 |
| cudaEventRecordWithFlags_v11010 | 3136 | 4.7 |
| cudaMemsetAsync_v3020 | 540 | 4.3 |

Mean train_step CPU range: 6520 ms

## Host time inside NVTX ranges

| range | calls | CPU ms | median call ms | calls > 20 ms | CPU ms in those |
|---|---|---|---|---|---|
| fwd_bwd_mb0 | 1.0 | 1743.3 | 1511.52 | 1.0 | 1743.3 |
| fwd_bwd_mb1 | 1.0 | 1588.5 | 1560.32 | 1.0 | 1588.5 |
| fwd_bwd_mb2 | 1.0 | 1537.6 | 1533.13 | 1.0 | 1537.6 |
| fwd_bwd_mb3 | 1.0 | 1519.3 | 1517.61 | 1.0 | 1519.3 |
| grad_clip | 1.0 | 45.2 | 47.06 | 0.8 | 41.7 |
| optimizer_step | 1.0 | 41.6 | 41.60 | 1.0 | 41.6 |
| mixer: NemotronV3Attention | 48.0 | 3683.2 | 1.60 | 6.8 | 3625.3 |
| fused_attention: FusedAttention | 48.0 | 3599.7 | 0.28 | 6.8 | 3588.7 |
| mixer: NemotronV3Mamba2Mixer | 184.0 | 599.9 | 2.95 | 0.7 | 102.7 |
| mixer: MoE | 184.0 | 1482.4 | 2.56 | 5.8 | 971.9 |
| experts: GroupedExpertsDeepEP | 184.0 | 996.6 | 1.41 | 4.2 | 684.8 |
| shared_experts: MLP | 184.0 | 162.2 | 0.49 | 0.9 | 57.2 |

## Ranges with calls > 20 ms (count per rank per step)

- `mixer: NemotronV3Attention`: 6.8
- `attn_module: DotProductAttention`: 6.8
- `fused_attention: FusedAttention`: 6.8
- `mixer: MoE`: 5.8
- `experts: GroupedExpertsDeepEP`: 4.2
- `FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM`: 4.0
- `model: FSDPNemotronV3Model`: 4.0
- `5: FSDPNemotronV3Block`: 3.4
- `nvte_flash_attn_fwd`: 3.4
- `42: FSDPNemotronV3Block`: 3.4
- `nvte_flash_attn_bwd`: 3.4
- `6: FSDPNemotronV3Block`: 2.2
