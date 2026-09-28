# nsys breakdown (per step, 3 captured steps)

GPUs analysed: [0, 1, 2, 3, 4, 5]

## Time budget (mean over GPUs)

| bucket | ms | % of window |
|---|---|---|
| wall window | 2283.2 | 100.0% |
| GPU busy (any kernel) | 2225.6 |  97.5% |
| GPU idle (no kernel) | 57.5 |   2.5% |
| compute kernels (union) | 2006.9 |  87.9% |
| communication kernels (union) | 489.2 |  21.4% |
|   comm overlapped with compute | 270.4 |  11.8% |
|   comm exposed (not overlapped) | 218.7 |   9.6% |

## Summed kernel time by category (mean over GPUs; overlapping streams can sum >100%)

| category | ms | % of window |
|---|---|---|
| gemm | 1129.8 |  49.5% |
| mamba | 258.8 |  11.3% |
| comm:nccl_allgather | 252.6 |  11.1% |
| memcpy/cat/copy | 223.3 |   9.8% |
| comm:nccl_reducescatter | 139.5 |   6.1% |
| comm:deepep | 120.0 |   5.3% |
| elementwise/reduce | 115.4 |   5.1% |
| norm | 92.9 |   4.1% |
| moe:routing/permute | 74.4 |   3.3% |
| optimizer | 63.4 |   2.8% |
| other | 50.8 |   2.2% |
| attention | 11.6 |   0.5% |
| comm:nccl_allreduce | 4.3 |   0.2% |
| gemm:grouped | 0.5 |   0.0% |

## Top kernels on GPU 0

| kernel | category | ms | calls |
|---|---|---|---|
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 283.0 | 276 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_allgather | 259.6 | 609 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | gemm | 152.3 | 294 |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>` | comm:nccl_reducescatter | 143.9 | 234 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 142.0 | 138 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 133.9 | 138 |
| `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | gemm | 87.7 | 294 |
| `nvjet_tst_192x192_64x3_1x2_h_bz_coopB_NNN` | gemm | 84.6 | 192 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN` | gemm | 52.8 | 69 |
| `void deep_ep::intranode::dispatch<(int)2, (int)768, (int)8192>(int4 *, float *, float *, i` | comm:deepep | 50.8 | 207 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` | memcpy/cat/copy | 48.4 | 138 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` | gemm | 48.3 | 156 |
| `at::native::detail::split_with_sizes_copy_out_contiguous_no_cast_kernel(char **, char **, ` | memcpy/cat/copy | 47.7 | 603 |
| `_chunk_scan_fwd_kernel` | mamba | 39.8 | 138 |
| `void transformer_engine::normalization::rmsnorm_fwd_general_kernel<transformer_engine::nor` | norm | 37.8 | 297 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::FusedOptimize` | optimizer | 37.0 | 756 |
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NTT` | gemm | 37.0 | 69 |
| `void causal_conv1d_channellast_fwd_kernel<Causal_conv1d_channellast_fwd_kernel_traits<(int` | mamba | 36.7 | 207 |
| `_permute_kernel` | moe:routing/permute | 34.8 | 207 |
| `void deep_ep::intranode::combine<__nv_bfloat16, (int)2, (int)768, (int)4096>(T1 *, float *` | comm:deepep | 34.4 | 138 |
| `_chunk_state_fwd_kernel` | mamba | 33.3 | 207 |
| `_unpermute_kernel` | moe:routing/permute | 32.5 | 207 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` | elementwise/reduce | 32.0 | 555 |
| `nvjet_tst_256x128_64x4_2x1_v_bz_coopA_NTT` | gemm | 30.7 | 87 |
| `nvjet_tst_320x128_64x3_1x1_v_bz_coopB_NTT` | gemm | 27.0 | 69 |

## Per-GPU spread

| GPU | window ms | busy % | idle % | comm exposed % |
|---|---|---|---|---|
| 0 | 2283.3 |  97.4% |   2.6% |   9.6% |
| 1 | 2283.4 |  97.5% |   2.5% |   9.5% |
| 2 | 2283.3 |  97.6% |   2.4% |  10.1% |
| 3 | 2283.3 |  97.6% |   2.4% |   9.5% |
| 4 | 2282.3 |  97.5% |   2.5% |   9.1% |
| 5 | 2283.3 |  97.3% |   2.7% |   9.5% |

## NVTX ranges projected onto the GPU (nsys nvtx_gpu_proj_sum, all ranks)

```
Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,Proj Avg (ns),Proj Med (ns),Proj Min (ns),Proj Max (ns),Proj StdDev (ns),Total GPU Ops,Avg GPU Ops,Avg Range Lvl,Avg Num Child
:_checkpoint_wrapped_module: NemotronV3Block,PushPop,13585477653,8694745800,828,16407581.7,15164074.0,6339035,22191006,4139801.7,38502,46.5,5.0,2.0
:train_step_15,PushPop,13571427225,13574926353,6,2261904537.5,2262364196.5,2259400473,2262580285,1230324.4,19844,3307.3,0.0,8.0
:train_step_17,PushPop,13540582617,13543032169,6,2256763769.5,2256706765.5,2255145046,2258135835,1238092.1,19844,3307.3,0.0,8.0
:train_step_16,PushPop,13538469167,13541044051,6,2256411527.8,2256600950.0,2254004829,2258472684,2081813.8,19844,3307.3,0.0,8.0
:fwd_bwd_mb0,PushPop,10312735271,33169745619,18,572929737.3,572072726.0,570030040,579067182,2758036.1,45834,2546.3,1.0,11.0
:FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM,PushPop,10159725020,9997159638,18,564429167.8,564366793.5,562081859,566868401,1905595.5,44586,2477.0,2.0,8.0
:model: FSDPNemotronV3Model,PushPop,10126719285,9956220551,18,562595515.8,562561154.5,560350518,564949743,1830851.2,44154,2453.0,3.0,107.0
:mixer: MoE,PushPop,8359069552,7032738015,414,20190989.3,20279497.0,15845336,22015357,832667.4,26910,65.0,6.0,4.0
:mixer: NemotronV3Mamba2Mixer,PushPop,5071251401,1273926691,414,12249399.5,12726876.0,5866237,13832760,1488402.2,9936,24.0,6.0,3.0
NCCL:ncclAllGather,PushPop,4512624430,110195550,3636,1241095.8,758029.0,9824,5072751,1203471.9,3636,1.0,3.2,0.0
:experts: FSDPGroupedExpertsDeepEP,PushPop,4247424368,5821031413,414,10259479.1,10173047.0,9534975,11899408,433307.5,9936,24.0,7.0,1.0
:nvte_cublas_gemm_v2,PushPop,4097761564,1842721658,4626,885811.0,867629.0,83679,1927349,408591.5,8190,1.8,4.5,0.0
:shared_experts: MLP,PushPop,4045574536,3460750134,1242,3257306.4,2004350.0,1928780,6822744,1799812.4,9108,7.3,3.0,2.0
:in_proj: Linear,PushPop,3866952412,495768852,1242,3113488.3,2365430.5,2285955,5002248,1076878.9,3312,2.7,3.0,0.0
NCCL:ncclReduceScatter,PushPop,2398599368,54480539,1800,1332555.2,563868.0,7616,6261807,1639250.6,1800,1.0,0.2,0.0
:down_proj: Linear,PushPop,1701054101,641847861,1242,1369608.8,807707.5,747877,2941423,810773.8,2898,2.3,4.0,1.3
:up_proj: Linear,PushPop,1633455195,808108339,1242,1315181.3,867378.0,843877,3010318,667607.5,2898,2.3,4.0,1.3
:gate: Gate,PushPop,1247427569,3008167205,1242,1004370.0,1147674.0,393571,1934767,437774.6,20286,16.3,3.0,0.0
:17: FSDPCheckpointWrapper,PushPop,1230257035,1214202043,36,34173806.5,33911565.0,21174688,47521291,13044043.8,3438,95.5,2.0,5.5
:40: FSDPCheckpointWrapper,PushPop,1193970609,478663919,36,33165850.3,32609671.5,20287129,47011739,12833015.5,3438,95.5,2.0,5.5
:24: FSDPCheckpointWrapper,PushPop,1192895316,1201508788,36,33135981.0,32842466.5,19688139,46085834,12446485.3,3438,95.5,2.0,5.5
:10: FSDPCheckpointWrapper,PushPop,1181297371,1191843219,36,32813815.9,32443584.5,20241820,46333363,12461658.5,3438,95.5,2.0,5.5
:29: FSDPCheckpointWrapper,PushPop,1180847908,819771207,36,32801330.8,32570433.0,20437787,45677798,12197250.0,3438,95.5,2.0,5.5
:38: FSDPCheckpointWrapper,PushPop,1180222234,479216798,36,32783950.9,32651093.5,20351322,45555973,12193711.1,3438,95.5,2.0,5.5
:36: FSDPCheckpointWrapper,PushPop,1175808724,802858330,36,32661353.4,32356895.0,20527961,46076078,12135432.2,3438,95.5,2.0,5.5
:47: FSDPCheckpointWrapper,PushPop,1167923973,479369069,36,32442332.6,32015473.0,20300517,45616998,12113120.4,3438,95.5,2.0,5.5
:15: FSDPCheckpointWrapper,PushPop,1167249039,1190851985,36,32423584.4,32179719.0,20278271,45448063,12089654.5,3438,95.5,2.0,5.5
:13: FSDPCheckpointWrapper,PushPop,1166579349,812785418,36,32404981.9,32311914.5,19701132,45377870,12364220.2,3438,95.5,2.0,5.5
:20: FSDPCheckpointWrapper,PushPop,1166442100,824050233,36,32401169.4,31846774.5,19802204,45920514,12573036.3,3438,95.5,2.0,5.5
:8: FSDPCheckpointWrapper,PushPop,1165212333,1208256451,36,32367009.3,31962492.0,20089272,45578858,12144143.0,3438,95.5,2.0,5.5
:31: FSDPCheckpointWrapper,PushPop,1161115293,1202799826,36,32253202.6,32008241.0,20317641,44886435,11906073.7,3438,95.5,2.0,5.5
:45: FSDPCheckpointWrapper,PushPop,1159965112,474717594,36,32221253.1,32067774.5,19811433,44410952,11853648.4,3438,95.5,2.0,5.5
:34: FSDPCheckpointWrapper,PushPop,1158520891,811522427,36,32181135.9,31799030.0,19789620,45353864,12201086.9,3438,95.5,2.0,5.5
:22: FSDPCheckpointWrapper,PushPop,1157020291,1204068363,36,32139452.5,31745914.5,20178297,44814031,11908207.3,3438,95.5,2.0,5.5
:43: FSDPCheckpointWrapper,PushPop,1155969732,485707346,36,32110270.3,31841154.5,19700724,44950210,12275313.5,3438,95.5,2.0,5.5
:51: FSDPCheckpointWrapper,PushPop,1153461078,485799138,36,32040585.5,32044043.5,21811759,42299796,10189053.6,3438,95.5,2.0,5.5
:49: FSDPCheckpointWrapper,PushPop,1153057994,476798424,36,32029388.7,31944516.0,20348519,44991696,11658513.3,3438,95.5,2.0,5.5
:6: FSDPCheckpointWrapper,PushPop,1151309043,791418264,36,31980806.8,31672438.5,19611661,44922750,12327649.7,3438,95.5,2.0,5.5
:3: FSDPCheckpointWrapper,PushPop,1145887971,1180397915,36,31830221.4,31494528.5,19103376,44310078,12031339.0,3438,95.5,2.0,5.5
:27: FSDPCheckpointWrapper,PushPop,1142644749,825223785,36,31740131.9,31582050.5,19777812,44283521,11995949.3,3438,95.5,2.0,5.5
:mixer: NemotronV3Attention,PushPop,1127217594,2308670927,216,5218600.0,5131271.0,2903630,7819339,2146965.5,4644,21.5,3.0,5.0
:1: FSDPCheckpointWrapper,PushPop,1094581420,1104633417,36,30405039.4,30555435.5,16326779,43802530,12893131.5,3438,95.5,2.0,5.5
:norm: RMSNorm,PushPop,958805056,1469814396,2736,350440.4,305887.5,279649,695845,103020.3,3690,1.3,2.7,1.0
:nvte_rmsnorm_fwd,PushPop,680346989,33758688,1782,381788.4,309615.0,297983,695845,116006.1,1782,1.0,4.6,0.0
:optimizer_step,PushPop,677467789,151558586,18,37637099.4,37592312.5,37479314,37958878,147923.2,4644,258.0,1.0,0.0
:grad_clip,PushPop,642738590,6298730862,18,35707699.4,35614909.5,35360195,36293111,299923.9,8766,487.0,1.0,5.0
:39: FSDPCheckpointWrapper,PushPop,6309680
```
