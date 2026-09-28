# nsys breakdown (per step, 6 captured steps)

GPUs analysed: [0, 1, 2, 3, 4, 5, 6, 7]

## Time budget (mean over GPUs)

| bucket | ms | % of window |
|---|---|---|
| wall window | 966.5 | 100.0% |
| GPU busy (any kernel) | 791.4 |  81.9% |
| GPU idle (no kernel) | 175.1 |  18.1% |
| compute kernels (union) | 519.3 |  53.7% |
| communication kernels (union) | 307.9 |  31.9% |
|   comm overlapped with compute | 35.8 |   3.7% |
|   comm exposed (not overlapped) | 272.1 |  28.2% |

## Summed kernel time by category (mean over GPUs; overlapping streams can sum >100%)

| category | ms | % of window |
|---|---|---|
| gemm | 245.6 |  25.4% |
| comm:deepep | 211.0 |  21.8% |
| comm:nccl_allgather | 81.3 |   8.4% |
| comm:nccl_reducescatter | 69.3 |   7.2% |
| optimizer | 60.9 |   6.3% |
| memcpy/cat/copy | 56.3 |   5.8% |
| elementwise/reduce | 55.4 |   5.7% |
| mamba | 51.7 |   5.4% |
| moe:routing/permute | 17.3 |   1.8% |
| norm | 12.9 |   1.3% |
| loss/softmax | 11.6 |   1.2% |
| other | 5.5 |   0.6% |
| attention | 3.4 |   0.4% |
| comm:nccl_allreduce | 3.3 |   0.3% |
| gemm:grouped | 0.8 |   0.1% |

## Top kernels on GPU 0

| kernel | category | ms | calls |
|---|---|---|---|
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_allgather | 87.5 | 942 |
| `void deep_ep::intranode::combine<__nv_bfloat16, (int)8, (int)768, (int)4096>(T1 *, float *` | comm:deepep | 71.9 | 552 |
| `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_reducescatter | 67.8 | 936 |
| `void deep_ep::intranode::dispatch<(int)8, (int)768, (int)8192>(int4 *, float *, float *, i` | comm:deepep | 64.5 | 552 |
| `void deep_ep::intranode::cached_notify_combine<(int)8>(void **, int *, int, int, int, int ` | comm:deepep | 48.9 | 552 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 37.6 | 552 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 35.3 | 552 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 34.3 | 552 |
| `void deep_ep::intranode::notify_dispatch<(int)8>(const int *, int *, const int *, int *, i` | comm:deepep | 30.3 | 276 |
| `nvjet_tst_128x256_64x4_1x2_h_bz_coopA_NNN` | gemm | 29.0 | 780 |
| `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` | gemm | 27.5 | 564 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN` | gemm | 20.5 | 288 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` | elementwise/reduce | 17.4 | 1854 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` | elementwise/reduce | 16.2 | 4074 |
| `void at::native::detail::chunk_cat_cuda_kernel<float, c10::BFloat16>(T2 **, T1 *, long *, ` | memcpy/cat/copy | 13.9 | 660 |
| `nvjet_tst_168x128_64x5_1x2_h_bz_TNN` | gemm | 11.8 | 624 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListSca` | optimizer | 11.8 | 1134 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` | memcpy/cat/copy | 10.0 | 2388 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | gemm | 9.5 | 348 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMet` | optimizer | 9.1 | 1134 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMet` | optimizer | 9.0 | 1134 |
| `nvjet_tst_256x128_64x4_2x1_v_bz_coopA_NTT` | gemm | 8.2 | 348 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cu` | memcpy/cat/copy | 7.7 | 3516 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListSca` | optimizer | 7.3 | 1134 |
| `_layer_norm_bwd_kernel` | norm | 7.2 | 276 |

## Per-GPU spread

| GPU | window ms | busy % | idle % | comm exposed % |
|---|---|---|---|---|
| 0 | 966.4 |  82.7% |  17.3% |  29.4% |
| 1 | 966.4 |  84.2% |  15.8% |  31.1% |
| 2 | 966.7 |  84.2% |  15.8% |  29.6% |
| 3 | 966.5 |  79.9% |  20.1% |  25.7% |
| 4 | 966.8 |  80.9% |  19.1% |  26.6% |
| 5 | 966.6 |  80.6% |  19.4% |  27.5% |
| 6 | 966.4 |  84.9% |  15.1% |  31.6% |
| 7 | 966.5 |  77.7% |  22.3% |  23.7% |

## NVTX ranges projected onto the GPU (nsys nvtx_gpu_proj_sum, all ranks)

```
Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,Proj Avg (ns),Proj Med (ns),Proj Min (ns),Proj Max (ns),Proj StdDev (ns),Total GPU Ops,Avg GPU Ops,Avg Range Lvl,Avg Num Child
:mixer: MoE,PushPop,22071978282,12296952816,4416,4998183.5,4892404.0,3715287,10232676,710230.3,229632,52.0,3.0,3.0
:FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM,PushPop,20833233842,20312069980,96,217012852.5,214906026.5,212515847,228295228,4018723.1,219072,2282.0,2.0,6.0
:model: FSDPNemotronV3Model,PushPop,20189871040,20107003375,96,210311156.7,208736196.5,205751105,221439712,3947971.4,217248,2263.0,3.0,107.0
:experts: GroupedExpertsDeepEP,PushPop,18029946087,7457073222,4416,4082868.2,4060099.5,2634166,9717862,756059.2,103776,23.5,4.0,0.5
:fwd_bwd_mb0,PushPop,10839548138,19594722789,48,225823919.5,224332161.5,219839658,235562218,5597933.8,112464,2343.0,1.0,8.0
:fwd_bwd_mb1,PushPop,10703608669,20627466975,48,222991847.3,221896157.5,220930557,228748312,2635555.7,112464,2343.0,1.0,8.0
:mixer: NemotronV3Mamba2Mixer,PushPop,9000580043,11770565934,4416,2038174.8,2170800.0,1567897,3749134,248729.1,181056,41.0,3.0,2.0
:train_step_27,PushPop,7763632096,7767334509,8,970454012.0,970668941.5,968057144,971398664,1032914.2,52064,6508.0,0.0,9.0
:train_step_23,PushPop,7674063996,7679699705,8,959257999.5,959110133.5,958299573,960518117,809220.3,52064,6508.0,0.0,9.0
:train_step_26,PushPop,7658061145,7661829072,8,957257643.1,957270379.5,956411369,958172825,803881.5,52064,6508.0,0.0,9.0
:train_step_25,PushPop,7636686881,7640362773,8,954585860.1,954592382.5,953763961,955212516,541091.2,52064,6508.0,0.0,9.0
:train_step_28,PushPop,7582940911,7586613708,8,947867613.9,947885882.0,947617459,948065988,146949.0,52064,6508.0,0.0,9.0
:train_step_24,PushPop,7571392864,7575042262,8,946424108.0,946459115.5,946161144,946685119,172838.0,52064,6508.0,0.0,9.0
NCCL:ncclAllGather,PushPop,3902789811,230749654,7536,517886.1,411102.0,8032,5145639,466500.0,7536,1.0,4.5,0.0
NCCL:ncclReduceScatter,PushPop,3325433352,213222038,7488,444101.7,331264.0,8352,4998413,637197.0,7488,1.0,0.0,0.0
:nvte_cublas_gemm_v2,PushPop,3045862495,630338417,20448,148956.5,114176.0,13088,3879862,409497.2,34176,1.7,4.9,0.0
:optimizer_step,PushPop,2921195520,1990012376,48,60858240.0,60836240.5,60691041,61081911,106982.8,64032,1334.0,1.0,0.0
:in_proj: Linear,PushPop,2068718757,549545863,4416,468459.9,474112.0,292096,639230,155492.9,13248,3.0,4.0,0.0
:shared_experts: MLP,PushPop,2009384181,2241423900,4416,455023.6,539793.0,258112,2590096,114957.2,35328,8.0,4.0,2.0
:mixer: NemotronV3Attention,PushPop,1411518076,1496610469,1152,1225276.1,1108764.0,577313,3396554,303056.6,23616,20.5,3.0,5.0
:grad_clip,PushPop,1299757517,1676500245,48,27078281.6,27077329.0,26933209,27204406,87145.9,22656,472.0,1.0,2.0
:gate: Gate,PushPop,1230658204,1761911284,4416,278681.7,232895.0,209215,1004353,68306.4,79488,18.0,4.0,0.0
:1: FSDPNemotronV3Block,PushPop,1219083453,940851796,192,6349393.0,5919929.0,4264272,10362212,1928874.4,10848,56.5,2.0,2.0
:51: FSDPNemotronV3Block,PushPop,1116827633,638255132,192,5816810.6,5842098.5,3997721,7748529,1101597.6,10848,56.5,2.0,2.0
:34: FSDPNemotronV3Block,PushPop,1066689869,611905783,192,5555676.4,5590435.0,4535433,6731241,436683.9,10848,56.5,2.0,2.0
:lm_head: FSDPLinear,PushPop,1023637514,82779170,192,5331445.4,5330138.5,3722950,7146892,1550755.8,576,3.0,1.5,1.5
:17: FSDPNemotronV3Block,PushPop,1020374716,580905050,192,5314451.6,5318156.5,4234863,6346638,512007.7,10848,56.5,2.0,2.0
:20: FSDPNemotronV3Block,PushPop,1011696517,627540738,192,5269252.7,5307203.5,4290607,6089050,398684.5,10848,56.5,2.0,2.0
:13: FSDPNemotronV3Block,PushPop,1006016367,601835840,192,5239668.6,5278612.5,4375476,5988067,383991.3,10848,56.5,2.0,2.0
:38: FSDPNemotronV3Block,PushPop,1000024300,623980110,192,5208459.9,5192361.5,4203765,6275839,440266.2,10848,56.5,2.0,2.0
:24: FSDPNemotronV3Block,PushPop,987662493,602304660,192,5144075.5,5145417.5,4391840,5775664,342625.1,10848,56.5,2.0,2.0
:40: FSDPNemotronV3Block,PushPop,987649529,582826393,192,5144008.0,5140630.5,4375663,6420415,424990.5,10848,56.5,2.0,2.0
:22: FSDPNemotronV3Block,PushPop,976851811,598294980,192,5087769.8,5083303.0,4149807,6714240,502918.0,10848,56.5,2.0,2.0
:15: FSDPNemotronV3Block,PushPop,973368931,594177605,192,5069629.8,5001443.5,4394927,6077225,379048.9,10848,56.5,2.0,2.0
:49: FSDPNemotronV3Block,PushPop,972957910,610456179,192,5067489.1,5073353.0,4133318,5721252,381315.2,10848,56.5,2.0,2.0
:29: FSDPNemotronV3Block,PushPop,971959422,581532451,192,5062288.7,5085430.0,4372244,5740156,328991.8,10848,56.5,2.0,2.0
:10: FSDPNemotronV3Block,PushPop,968960297,607750000,192,5046668.2,5030496.0,4204752,5824682,427648.3,10848,56.5,2.0,2.0
:45: FSDPNemotronV3Block,PushPop,967454133,605837495,192,5038823.6,4972099.0,4242000,5583945,333061.4,10848,56.5,2.0,2.0
:8: FSDPNemotronV3Block,PushPop,966694368,589130735,192,5034866.5,5036788.0,4428974,5648127,309499.3,10848,56.5,2.0,2.0
:27: FSDPNemotronV3Block,PushPop,966574218,596924527,192,5034240.7,5021971.0,4242736,5803163,361850.5,10848,56.5,2.0,2.0
:43: FSDPNemotronV3Block,PushPop,957370062,595808394,192,4986302.4,4977814.5,4161701,5768273,360455.4,10848,56.5,2.0,2.0
:47: FSDPNemotronV3Block,PushPop,943480324,578460616,192,4913960.0,4871977.0,4169941,5759103,368393.4,10848,56.5,2.0,2.0
:36: FSDPNemotronV3Block,PushPop,937972122,591771675,192,4885271.5,4853154.5,4180479,5555079,309879.9,10848,56.5,2.0,2.0
:31: FSDPNemotronV3Block,PushPop,926858155,603105721,192,4827386.2,4767278.0,4027904,5474877,343308.6,10848,56.5,2.0,2.0
:norm: Float32RMSNorm,PushPop,903106967,2528155102,10176,88748.7,101584.5,66560,738493,21263.2,35616,3.5,3.0,0.0
:6: FSDPNemotronV3Block,PushPop,902331955,601372403,192,4699645.6,4683307.0,4016529,5200190,287159.8,10848,56.5,2.0,2.0
:3: FSDPNemotronV3Block,PushPop,882926832,649812829,192,4598577.3,4565971.0,3948633,5496561,272584.0,10848,56.5,2.0,2.0
:up_proj: Linear,PushPop,758502511,785092527,4416,171762.3,180752.5,113505,230399,49056.2,11040,2.5
```
