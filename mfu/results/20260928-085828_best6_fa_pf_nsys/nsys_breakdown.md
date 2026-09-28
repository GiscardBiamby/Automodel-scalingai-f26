# nsys breakdown (per step, 4 captured steps)

GPUs analysed: [0, 1, 2, 3, 4, 5]

## Time budget (mean over GPUs)

| bucket | ms | % of window |
|---|---|---|
| wall window | 1172.0 | 100.0% |
| GPU busy (any kernel) | 1052.4 |  89.8% |
| GPU idle (no kernel) | 119.6 |  10.2% |
| compute kernels (union) | 712.0 |  60.8% |
| communication kernels (union) | 608.4 |  51.9% |
|   comm overlapped with compute | 267.9 |  22.9% |
|   comm exposed (not overlapped) | 340.4 |  29.0% |

## Summed kernel time by category (mean over GPUs; overlapping streams can sum >100%)

| category | ms | % of window |
|---|---|---|
| comm:nccl_allgather | 299.4 |  25.5% |
| comm:nccl_reducescatter | 274.9 |  23.5% |
| gemm | 271.4 |  23.2% |
| memcpy/cat/copy | 241.2 |  20.6% |
| comm:deepep | 87.4 |   7.5% |
| optimizer | 76.6 |   6.5% |
| elementwise/reduce | 56.7 |   4.8% |
| mamba | 51.8 |   4.4% |
| moe:routing/permute | 15.0 |   1.3% |
| norm | 13.4 |   1.1% |
| other | 12.3 |   1.0% |
| attention | 3.2 |   0.3% |
| comm:nccl_allreduce | 2.4 |   0.2% |
| gemm:grouped | 0.4 |   0.0% |

## Top kernels on GPU 0

| kernel | category | ms | calls |
|---|---|---|---|
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_allgather | 268.5 | 812 |
| `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_reducescatter | 254.4 | 400 |
| `at::native::detail::split_with_sizes_copy_out_contiguous_no_cast_kernel(char **, char **, ` | memcpy/cat/copy | 55.7 | 804 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 55.4 | 184 |
| `void at::native::detail::chunk_cat_cuda_kernel<float, c10::BFloat16>(T2 **, T1 *, long *, ` | memcpy/cat/copy | 52.8 | 308 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` | memcpy/cat/copy | 47.1 | 184 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 42.4 | 184 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 38.5 | 184 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMet` | optimizer | 38.2 | 2484 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::FusedOptimize` | optimizer | 36.9 | 1008 |
| `void deep_ep::intranode::notify_dispatch<(int)2>(const int *, int *, const int *, int *, i` | comm:deepep | 33.5 | 92 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | gemm | 27.4 | 300 |
| `void deep_ep::intranode::dispatch<(int)2, (int)768, (int)8192>(int4 *, float *, float *, i` | comm:deepep | 26.3 | 184 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` | memcpy/cat/copy | 24.3 | 184 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` | gemm | 24.1 | 256 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` | elementwise/reduce | 22.8 | 1236 |
| `void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueTy` | memcpy/cat/copy | 22.0 | 184 |
| `void deep_ep::intranode::combine<__nv_bfloat16, (int)2, (int)768, (int)4096>(T1 *, float *` | comm:deepep | 21.8 | 184 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cu` | memcpy/cat/copy | 16.0 | 1256 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN` | gemm | 13.4 | 92 |
| `void deep_ep::intranode::cached_notify_dispatch<(int)2>(const int *, int, void **, int **,` | comm:deepep | 10.5 | 92 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` | elementwise/reduce | 9.2 | 740 |
| `nvjet_tst_320x128_64x3_1x1_v_bz_coopB_NTT` | gemm | 8.8 | 92 |
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NTT` | gemm | 8.8 | 92 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | gemm | 8.7 | 116 |

## Per-GPU spread

| GPU | window ms | busy % | idle % | comm exposed % |
|---|---|---|---|---|
| 0 | 1171.7 |  88.4% |  11.6% |  27.8% |
| 1 | 1172.1 |  89.4% |  10.6% |  28.9% |
| 2 | 1172.1 |  89.2% |  10.8% |  28.6% |
| 3 | 1172.0 |  90.6% |   9.4% |  29.8% |
| 4 | 1172.0 |  90.4% |   9.6% |  29.4% |
| 5 | 1172.1 |  90.8% |   9.2% |  29.7% |

## NVTX ranges projected onto the GPU (nsys nvtx_gpu_proj_sum, all ranks)

```
Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,Proj Avg (ns),Proj Med (ns),Proj Min (ns),Proj Max (ns),Proj StdDev (ns),Total GPU Ops,Avg GPU Ops,Avg Range Lvl,Avg Num Child
:mixer: MoE,PushPop,17654982670,13150940001,1104,15991832.1,17781090.0,7651064,47683133,8208363.4,65928,59.7,3.0,4.0
:fwd_bwd_mb0,PushPop,8631028534,22130239680,24,359626188.9,359822970.0,356609873,362323931,2058657.9,65480,2728.3,1.0,11.0
:FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM,PushPop,8477970194,8411804515,24,353248758.1,353588064.0,349316314,356412291,2353788.3,63816,2659.0,2.0,8.0
:model: FSDPNemotronV3Model,PushPop,8396771016,8346966023,24,349865459.0,350689813.0,344956661,353915698,2403958.2,63240,2635.0,3.0,108.0
:experts: FSDPGroupedExpertsDeepEP,PushPop,7511551494,3881942264,1104,6803941.6,7148191.5,3223285,41167401,4213001.4,27888,25.3,4.0,0.5
:train_step_23,PushPop,7267012287,7269184708,6,1211168714.5,1211135522.5,1211071262,1211401436,120481.7,20936,3489.3,0.0,8.0
NCCL:ncclAllGather,PushPop,7141295298,144344754,4848,1473039.5,909914.5,7616,38377459,2339814.0,4848,1.0,3.0,0.0
:train_step_22,PushPop,7098276665,7100498118,6,1183046110.8,1183035458.0,1182811294,1183290395,187787.1,20936,3489.3,0.0,8.0
:train_step_21,PushPop,6788358668,6790516240,6,1131393111.3,1131330827.5,1131252810,1131799053,204990.6,20936,3489.3,0.0,8.0
:train_step_20,PushPop,6748021715,6752068439,6,1124670285.8,1124798188.5,1123683229,1125274235,538187.2,20936,3489.3,0.0,8.0
NCCL:ncclReduceScatter,PushPop,6512289207,73723298,2400,2713453.8,1236893.0,7520,27271971,3887712.5,2400,1.0,0.5,0.0
:mixer: NemotronV3Mamba2Mixer,PushPop,5297930918,3554273685,1104,4798850.5,4010242.0,1836814,31066454,5216086.9,47520,43.0,3.0,2.5
:51: FSDPNemotronV3Block,PushPop,1697356869,1741387863,48,35361601.4,32903092.5,26549376,46833589,8016240.0,3696,77.0,2.0,3.0
:nvte_cublas_gemm_v2,PushPop,1405595673,177936479,5064,277566.3,310591.0,21632,477758,119300.5,8424,1.7,5.0,0.0
:shared_experts: MLP,PushPop,1308847115,576664441,1104,1185549.9,988999.0,645348,2032067,458428.5,8832,8.0,4.0,2.0
:49: FSDPNemotronV3Block,PushPop,1125452958,744350593,48,23446936.6,19688010.0,12121270,36705646,6889478.5,3456,72.0,2.0,3.0
:29: FSDPNemotronV3Block,PushPop,997766721,1255992128,48,20786806.7,17589933.5,8174843,47915164,13111366.5,3456,72.0,2.0,3.0
:in_proj: Linear,PushPop,996538286,142900601,1104,902661.5,917246.0,584289,1840483,300629.8,3312,3.0,4.0,0.0
:optimizer_step,PushPop,900255190,200791262,24,37510632.9,37503156.0,37379829,37640414,71194.5,6192,258.0,1.0,0.0
:grad_clip,PushPop,848713996,4273398022,24,35363083.2,35329158.0,34984546,35965466,285299.0,11688,487.0,1.0,5.0
:31: FSDPNemotronV3Block,PushPop,837512379,1872595449,48,17448174.6,14718015.0,7996251,40503559,10642209.6,3456,72.0,2.0,3.0
:22: FSDPNemotronV3Block,PushPop,771637810,333546880,48,16075787.7,14351817.0,8041563,36408674,9238721.1,3456,72.0,2.0,3.0
:47: FSDPNemotronV3Block,PushPop,770360357,373758334,48,16049174.1,19278485.5,10449445,20367382,4186627.1,3456,72.0,2.0,3.0
:38: FSDPNemotronV3Block,PushPop,743913763,304745692,48,15498203.4,14153226.5,8132868,29350009,7640267.8,3456,72.0,2.0,3.0
:1: FSDPNemotronV3Block,PushPop,741072843,600958195,48,15439017.6,15915490.0,11618188,30492658,3370836.9,3168,66.0,2.0,2.5
:10: FSDPNemotronV3Block,PushPop,730372088,555277548,48,15216085.2,14090523.0,8056668,35810669,7917835.5,3456,72.0,2.0,3.0
:27: FSDPNemotronV3Block,PushPop,726638151,988223845,48,15138294.8,13870082.0,8049370,33501471,7287082.0,3216,67.0,2.0,3.0
:24: FSDPNemotronV3Block,PushPop,716623350,590118144,48,14929653.1,14229384.0,8069435,32153919,6938681.2,3456,72.0,2.0,3.0
:45: FSDPNemotronV3Block,PushPop,713507003,336738728,48,14864729.2,15376483.5,9057865,19956773,4872494.0,3456,72.0,2.0,3.0
:34: FSDPNemotronV3Block,PushPop,707294908,577679806,48,14735310.6,13334966.0,8071556,31040257,6900534.3,3216,67.0,2.0,3.0
:40: FSDPNemotronV3Block,PushPop,704967840,452815301,48,14686830.0,14561896.0,8155770,27168970,6061901.8,3456,72.0,2.0,3.0
:36: FSDPNemotronV3Block,PushPop,699068971,513333972,48,14563936.9,14209320.5,8194938,32354842,6397190.2,3456,72.0,2.0,3.0
:8: FSDPNemotronV3Block,PushPop,690092995,352897524,48,14376937.4,14242833.5,8054906,23224076,6158631.0,3456,72.0,2.0,3.0
:3: FSDPNemotronV3Block,PushPop,687980927,353982301,48,14332936.0,14487780.0,8032557,19706018,5195751.9,3456,72.0,2.0,3.0
:43: FSDPNemotronV3Block,PushPop,687540343,380179105,48,14323757.1,14712450.5,8249657,31848402,4752688.2,3216,67.0,2.0,3.0
:15: FSDPNemotronV3Block,PushPop,681779562,596864071,48,14203740.9,14099840.5,8107234,30278896,6139849.9,3456,72.0,2.0,3.0
:17: FSDPNemotronV3Block,PushPop,681188368,307340842,48,14191424.3,14257450.5,8048763,20421441,5841210.9,3456,72.0,2.0,3.0
:13: FSDPNemotronV3Block,PushPop,658904728,300278005,48,13727181.8,13481445.0,8115643,22438004,5449395.1,3216,67.0,2.0,3.0
:gate: Gate,PushPop,649917221,467817536,1104,588693.1,600255.0,395971,685569,56521.2,19872,18.0,4.0,0.0
:6: FSDPNemotronV3Block,PushPop,643876845,341080170,48,13414100.9,13816019.0,8096022,18046310,4488958.8,3216,67.0,2.0,3.0
:20: FSDPNemotronV3Block,PushPop,634000675,337768001,48,13208347.4,13343648.0,8075413,18264116,4864289.6,3216,67.0,2.0,3.0
:mixer: NemotronV3Attention,PushPop,594881083,434352790,288,2065559.3,2474224.0,712317,3618552,628964.3,5904,20.5,3.0,5.0
:up_proj: Linear,PushPop,575781865,197595926,1104,521541.5,408945.5,277890,886562,210367.3,2760,2.5,5.0,1.5
:down_proj: Linear,PushPop,509556911,210648834,1104,461555.2,384641.5,280898,812097,160167.6,2760,2.5,5.0,1.5
:norm: Float32RMSNorm,PushPop,469901690,653554835,2544,184709.8,198978.0,104192,511748,54370.6,8904,3.5,3.0,0.0
:25: FSDPNemotronV3Block,PushPop,460791034,178851586,48,9599813.2,3624551.0,1998829,31301621,10818311.2,2376,49.5,2.0,2.5
:30: FSDPNemotronV3Block,PushPop,407256612,186716056,48,8484512.8,4067900.0,1888845,27802299,8500836.2,2376,49.5,2.0,2.5
:26: FSDPNemotronV3Block,PushPop,354367328,143860416,48,7382652.7,2885
```
