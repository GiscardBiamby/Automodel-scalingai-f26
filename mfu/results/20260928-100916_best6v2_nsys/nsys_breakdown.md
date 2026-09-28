# nsys breakdown (per step, 3 captured steps)

GPUs analysed: [0, 1, 2, 3, 4, 5]

## Time budget (mean over GPUs)

| bucket | ms | % of window |
|---|---|---|
| wall window | 2481.6 | 100.0% |
| GPU busy (any kernel) | 2421.5 |  97.6% |
| GPU idle (no kernel) | 60.1 |   2.4% |
| compute kernels (union) | 2065.2 |  83.2% |
| communication kernels (union) | 644.2 |  26.0% |
|   comm overlapped with compute | 287.9 |  11.6% |
|   comm exposed (not overlapped) | 356.3 |  14.4% |

## Summed kernel time by category (mean over GPUs; overlapping streams can sum >100%)

| category | ms | % of window |
|---|---|---|
| gemm | 1137.1 |  45.8% |
| memcpy/cat/copy | 294.9 |  11.9% |
| comm:deepep | 257.9 |  10.4% |
| comm:nccl_allgather | 257.8 |  10.4% |
| mamba | 256.3 |  10.3% |
| comm:nccl_reducescatter | 144.8 |   5.8% |
| elementwise/reduce | 114.7 |   4.6% |
| moe:routing/permute | 74.9 |   3.0% |
| norm | 74.1 |   3.0% |
| optimizer | 62.2 |   2.5% |
| other | 50.3 |   2.0% |
| attention | 14.2 |   0.6% |
| comm:nccl_allreduce | 4.1 |   0.2% |
| gemm:grouped | 0.5 |   0.0% |

## Top kernels on GPU 0

| kernel | category | ms | calls |
|---|---|---|---|
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 284.1 | 276 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_allgather | 266.8 | 609 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | gemm | 158.3 | 312 |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>` | comm:nccl_reducescatter | 152.5 | 234 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 141.3 | 138 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 132.3 | 138 |
| `void deep_ep::intranode::dispatch<(int)2, (int)768, (int)8192>(int4 *, float *, float *, i` | comm:deepep | 131.0 | 207 |
| `nvjet_tst_192x192_64x4_1x2_h_bz_coopB_TNN` | gemm | 92.0 | 312 |
| `void deep_ep::intranode::combine<__nv_bfloat16, (int)2, (int)768, (int)4096>(T1 *, float *` | comm:deepep | 85.8 | 138 |
| `nvjet_tst_192x192_64x3_1x2_h_bz_coopB_NNN` | gemm | 84.1 | 192 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN` | gemm | 52.6 | 69 |
| `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigne` | memcpy/cat/copy | 47.7 | 138 |
| `at::native::detail::split_with_sizes_copy_out_contiguous_no_cast_kernel(char **, char **, ` | memcpy/cat/copy | 47.6 | 603 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` | memcpy/cat/copy | 47.5 | 1113 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` | gemm | 47.4 | 156 |
| `_chunk_scan_fwd_kernel` | mamba | 39.1 | 138 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::FusedOptimize` | optimizer | 36.9 | 756 |
| `void causal_conv1d_channellast_fwd_kernel<Causal_conv1d_channellast_fwd_kernel_traits<(int` | mamba | 36.2 | 207 |
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NTT` | gemm | 36.0 | 69 |
| `_permute_kernel` | moe:routing/permute | 35.4 | 207 |
| `void at::native::<unnamed>::vectorized_layer_norm_kernel<float, float, (bool)1>(int, T2, c` | norm | 34.6 | 315 |
| `_chunk_state_fwd_kernel` | mamba | 32.9 | 207 |
| `_unpermute_kernel` | moe:routing/permute | 32.5 | 207 |
| `void deep_ep::intranode::cached_notify_combine<(int)2>(void **, int *, int, int, int, int ` | comm:deepep | 32.1 | 138 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` | elementwise/reduce | 31.7 | 555 |

## Per-GPU spread

| GPU | window ms | busy % | idle % | comm exposed % |
|---|---|---|---|---|
| 0 | 2481.6 |  97.5% |   2.5% |  14.4% |
| 1 | 2481.8 |  97.6% |   2.4% |  14.3% |
| 2 | 2481.9 |  97.6% |   2.4% |  14.7% |
| 3 | 2481.9 |  97.6% |   2.4% |  14.4% |
| 4 | 2480.8 |  97.6% |   2.4% |  14.0% |
| 5 | 2481.9 |  97.5% |   2.5% |  14.4% |

## NVTX ranges projected onto the GPU (nsys nvtx_gpu_proj_sum, all ranks)

```
Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,Proj Avg (ns),Proj Med (ns),Proj Min (ns),Proj Max (ns),Proj StdDev (ns),Total GPU Ops,Avg GPU Ops,Avg Range Lvl,Avg Num Child
:_checkpoint_wrapped_module: NemotronV3Block,PushPop,16115417603,10267332702,936,17217326.5,15311248.5,4526355,26325050,6447624.7,42930,45.9,5.0,2.0
:train_step_15,PushPop,14778326825,14782426230,6,2463054470.8,2463653087.0,2460319109,2463917150,1386922.4,20798,3466.3,0.0,8.0
:train_step_16,PushPop,14716927485,14719615213,6,2452821247.5,2453132128.0,2451407714,2453690006,850898.7,20798,3466.3,0.0,8.0
:train_step_17,PushPop,14714569103,14717353412,6,2452428183.8,2452062479.0,2450839423,2454137839,1309768.3,20798,3466.3,0.0,8.0
:fwd_bwd_mb0,PushPop,11811033918,36885874726,18,656168551.0,655734703.0,651628799,663725324,3884683.9,48696,2705.3,1.0,11.0
:FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM,PushPop,11652100706,11437025245,18,647338928.1,647404233.0,643231317,651440280,2928541.3,47448,2636.0,2.0,8.0
:model: FSDPNemotronV3Model,PushPop,11617595721,11394572368,18,645421984.5,645544814.0,641433329,649368394,2823926.9,47016,2612.0,3.0,107.0
:mixer: MoE,PushPop,9630295759,8207509594,414,23261584.0,23134308.5,18488072,26148025,1045475.4,26910,65.0,6.0,4.0
:mixer: NemotronV3Mamba2Mixer,PushPop,5819563599,1293820501,414,14056916.9,14334046.5,5595623,18080416,1988622.1,9936,24.0,6.0,3.0
:experts: FSDPGroupedExpertsDeepEP,PushPop,5317088874,6988954524,414,12843209.8,12568745.0,11952217,15533614,780218.0,9936,24.0,7.0,1.0
NCCL:ncclAllGather,PushPop,4600765065,113230861,3636,1265336.9,692931.5,9792,6016877,1269504.5,3636,1.0,3.2,0.0
:nvte_cublas_gemm_v2,PushPop,4282493173,1482899310,5058,846677.2,865655.0,76416,1769646,409851.9,8838,1.7,4.1,0.0
:shared_experts: MLP,PushPop,4022264857,2101136836,1242,3238538.5,1981065.0,1925807,6449725,1793862.5,9108,7.3,3.0,2.0
:in_proj: Linear,PushPop,3839506154,1535667581,1242,3091389.8,2332629.0,2284581,4703629,1085940.3,3312,2.7,3.0,0.0
NCCL:ncclReduceScatter,PushPop,2492201496,56228971,1800,1384556.4,710559.0,7584,6114321,1631169.7,1800,1.0,0.2,0.0
:norm: Float32RMSNorm,PushPop,1980564714,1447748101,2844,696401.1,785148.0,392676,1228425,217602.2,10422,3.7,2.7,0.0
:down_proj: Linear,PushPop,1669515522,613078000,1242,1344215.4,791745.5,744294,2684949,794102.5,2898,2.3,4.0,1.3
:up_proj: Linear,PushPop,1643178526,1004753461,1242,1323010.1,865419.5,844962,2822001,677611.5,2898,2.3,4.0,1.3
:17: FSDPCheckpointWrapper,PushPop,1354567007,996714038,36,37626861.3,37420809.0,23907039,51533047,13724916.3,3564,99.0,2.0,5.5
:24: FSDPCheckpointWrapper,PushPop,1324423983,986271336,36,36789555.1,36561114.0,23398941,50648966,13284317.6,3564,99.0,2.0,5.5
:29: FSDPCheckpointWrapper,PushPop,1314658044,1179930728,36,36518279.0,36747046.5,23205212,49395121,12459243.6,3564,99.0,2.0,5.5
:38: FSDPCheckpointWrapper,PushPop,1312536573,961262966,36,36459349.3,37018569.0,23259330,50099906,12376273.7,3564,99.0,2.0,5.5
:10: FSDPCheckpointWrapper,PushPop,1304172185,974972263,36,36227005.1,35994105.5,22848916,49948085,12997621.7,3564,99.0,2.0,5.5
:20: FSDPCheckpointWrapper,PushPop,1304062652,1188312155,36,36223962.6,36052886.0,23074080,49503895,12477856.7,3564,99.0,2.0,5.5
:45: FSDPCheckpointWrapper,PushPop,1301944680,541548764,36,36165130.0,36741145.0,23225925,49123498,12112091.8,3564,99.0,2.0,5.5
:47: FSDPCheckpointWrapper,PushPop,1290988503,554841904,36,35860791.8,35592900.5,22966550,49940146,12701247.4,3564,99.0,2.0,5.5
:36: FSDPCheckpointWrapper,PushPop,1289720309,1181628966,36,35825564.1,35623958.5,23105239,49605779,12663354.2,3564,99.0,2.0,5.5
:8: FSDPCheckpointWrapper,PushPop,1288743970,945396389,36,35798443.6,35211074.5,22872768,49672104,12724501.7,3564,99.0,2.0,5.5
:13: FSDPCheckpointWrapper,PushPop,1287227363,1204318853,36,35756315.6,35550833.0,22964741,49098580,12673320.9,3564,99.0,2.0,5.5
:40: FSDPCheckpointWrapper,PushPop,1286062316,552810464,36,35723953.2,35471637.5,23093187,48897076,12610929.4,3564,99.0,2.0,5.5
:3: FSDPCheckpointWrapper,PushPop,1284996056,980248731,36,35694334.9,36160313.5,22648691,47803635,11830126.9,3564,99.0,2.0,5.5
:43: FSDPCheckpointWrapper,PushPop,1283955851,535434283,36,35665440.3,35387280.0,22702836,48893178,12672991.5,3564,99.0,2.0,5.5
:31: FSDPCheckpointWrapper,PushPop,1283071562,968539576,36,35640876.7,35521448.0,23070370,49064586,12603911.9,3564,99.0,2.0,5.5
:15: FSDPCheckpointWrapper,PushPop,1281833572,942747764,36,35606488.1,35475633.5,23011189,48365273,12529923.0,3564,99.0,2.0,5.5
:34: FSDPCheckpointWrapper,PushPop,1279902561,1171437107,36,35552848.9,35044426.5,22645363,49631817,12725098.9,3564,99.0,2.0,5.5
:51: FSDPCheckpointWrapper,PushPop,1279333438,554649626,36,35537039.9,35565936.5,24635555,46475739,10681360.6,3564,99.0,2.0,5.5
:49: FSDPCheckpointWrapper,PushPop,1276731643,542243125,36,35464767.9,35423454.0,23038101,48750069,12320152.6,3564,99.0,2.0,5.5
:6: FSDPCheckpointWrapper,PushPop,1274850398,1224987280,36,35412511.1,35188296.0,22719440,49311315,12611055.1,3564,99.0,2.0,5.5
:27: FSDPCheckpointWrapper,PushPop,1271459369,1193593026,36,35318315.8,35173556.0,22772307,48371257,12547193.6,3564,99.0,2.0,5.5
:22: FSDPCheckpointWrapper,PushPop,1268104947,956516401,36,35225137.4,34950048.0,22893456,47880477,12291384.3,3564,99.0,2.0,5.5
:1: FSDPCheckpointWrapper,PushPop,1222140417,787617210,36,33948344.9,35029723.0,19446091,47418722,12961996.8,3564,99.0,2.0,5.5
:gate: Gate,PushPop,1210915502,2446293857,1242,974972.2,1112050.0,394851,2025045,420773.5,20286,16.3,3.0,0.0
:46: FSDPCheckpointWrapper,PushPop,679163343,214623948,36,18865648.4,19950204.0,14384614,22254080,3320482.2,2214,61.5,2.0,3.5
:39: FSDPCheckpointWrapper,PushPop,678445722,214263630,36,18845714.5,19951286.0,14438450,22143264,3255034.7,2214,61.5,2.0,3.5
:30: FSDPCheckpointWrapper,PushPop,678406300,459986553,36,18844619.4,19760220.0,14448405,22368992,3330991.2,2214,61.5,2.0,3.5
:optimizer_step,PushPop,676211211,158108982,18,37567289.5,37567618.5,37363894,37836364,117770.8,4644,258.0,1.0,
```
