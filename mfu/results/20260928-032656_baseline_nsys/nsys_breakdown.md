# nsys breakdown (per step, 3 captured steps)

GPUs analysed: [0, 1, 2, 3, 4, 5, 6, 7]

## Time budget (mean over GPUs)

| bucket | ms | % of window |
|---|---|---|
| wall window | 6545.8 | 100.0% |
| GPU busy (any kernel) | 2529.5 |  38.6% |
| GPU idle (no kernel) | 4016.3 |  61.4% |
| compute kernels (union) | 718.6 |  11.0% |
| communication kernels (union) | 1916.7 |  29.3% |
|   comm overlapped with compute | 105.8 |   1.6% |
|   comm exposed (not overlapped) | 1810.9 |  27.7% |

## Summed kernel time by category (mean over GPUs; overlapping streams can sum >100%)

| category | ms | % of window |
|---|---|---|
| comm:nccl_reducescatter | 787.1 |  12.0% |
| comm:nccl_allgather | 764.6 |  11.7% |
| comm:deepep | 624.2 |   9.5% |
| gemm | 340.5 |   5.2% |
| memcpy/cat/copy | 89.0 |   1.4% |
| elementwise/reduce | 88.7 |   1.4% |
| mamba | 74.8 |   1.1% |
| optimizer | 63.4 |   1.0% |
| moe:routing/permute | 22.6 |   0.3% |
| loss/softmax | 15.2 |   0.2% |
| norm | 15.1 |   0.2% |
| comm:nccl_allreduce | 8.1 |   0.1% |
| other | 7.3 |   0.1% |
| attention | 3.5 |   0.1% |
| gemm:grouped | 1.6 |   0.0% |

## Top kernels on GPU 0

| kernel | category | ms | calls |
|---|---|---|---|
| `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_reducescatter | 1208.0 | 936 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_allgather | 1091.8 | 939 |
| `void deep_ep::intranode::cached_notify_dispatch<(int)8>(const int *, int, void **, int **,` | comm:deepep | 218.7 | 276 |
| `void deep_ep::intranode::cached_notify_combine<(int)8>(void **, int *, int, int, int, int ` | comm:deepep | 142.1 | 552 |
| `void deep_ep::intranode::combine<__nv_bfloat16, (int)8, (int)768, (int)4096>(T1 *, float *` | comm:deepep | 118.6 | 552 |
| `void deep_ep::intranode::dispatch<(int)8, (int)768, (int)8192>(int4 *, float *, float *, i` | comm:deepep | 97.4 | 552 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 53.6 | 552 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 48.1 | 552 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 47.4 | 552 |
| `void deep_ep::intranode::notify_dispatch<(int)8>(const int *, int *, const int *, int *, i` | comm:deepep | 44.9 | 276 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` | elementwise/reduce | 34.6 | 5001 |
| `void at::native::detail::chunk_cat_cuda_kernel<float, c10::BFloat16>(T2 **, T1 *, long *, ` | memcpy/cat/copy | 27.8 | 660 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` | elementwise/reduce | 23.0 | 1479 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN` | gemm | 17.3 | 150 |
| `at::native::detail::split_with_sizes_copy_out_contiguous_no_cast_kernel(char **, char **, ` | memcpy/cat/copy | 13.4 | 936 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` | memcpy/cat/copy | 12.9 | 2388 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListSca` | optimizer | 11.8 | 567 |
| `nvjet_tst_256x128_64x4_2x1_v_bz_coopA_NTT` | gemm | 10.8 | 348 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cu` | memcpy/cat/copy | 9.9 | 3516 |
| `_chunk_scan_chunk_state_bwd_dx_kernel` | mamba | 9.3 | 276 |
| `_unpermute_kernel` | moe:routing/permute | 9.2 | 552 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMet` | optimizer | 9.1 | 567 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMet` | optimizer | 9.0 | 567 |
| `nvjet_tst_192x192_64x3_1x2_h_bz_coopB_NTN` | gemm | 9.0 | 138 |
| `_permute_kernel` | moe:routing/permute | 8.7 | 552 |

## Per-GPU spread

| GPU | window ms | busy % | idle % | comm exposed % |
|---|---|---|---|---|
| 0 | 6545.4 |  49.5% |  50.5% |  38.9% |
| 1 | 6545.8 |  31.8% |  68.2% |  21.1% |
| 2 | 6545.8 |  32.9% |  67.1% |  21.8% |
| 3 | 6545.9 |  36.5% |  63.5% |  25.4% |
| 4 | 6545.9 |  40.1% |  59.9% |  29.1% |
| 5 | 6545.9 |  33.8% |  66.2% |  22.3% |
| 6 | 6545.9 |  31.8% |  68.2% |  20.3% |
| 7 | 6545.8 |  52.9% |  47.1% |  42.5% |

## NVTX ranges projected onto the GPU (nsys nvtx_gpu_proj_sum, all ranks)

```
Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,Proj Avg (ns),Proj Med (ns),Proj Min (ns),Proj Max (ns),Proj StdDev (ns),Total GPU Ops,Avg GPU Ops,Avg Range Lvl,Avg Num Child
:mixer: NemotronV3Attention,PushPop,86493745929,88397528430,1152,75081376.7,1413368.5,290847,616191377,182642349.1,26058,22.6,3.0,5.0
:FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM,PushPop,74907064143,77097609383,96,780281918.2,701167097.5,686993673,1484986399,217841729.9,213026,2219.0,2.0,4.0
:model: FSDPNemotronV3Model,PushPop,74465424752,76882870225,96,775681507.8,697174509.0,683272795,1479091834,217620143.0,211874,2207.0,3.0,107.0
:train_step_10,PushPop,55470961743,55477234217,8,6933870217.9,6933975566.0,6932824765,6934286780,454315.6,85848,10731.0,0.0,11.0
:train_step_11,PushPop,51216878321,51220888725,8,6402109790.1,6402213420.5,6396979612,6406327756,2648322.2,85991,10748.9,0.0,11.0
:train_step_12,PushPop,49775549603,49779413050,8,6221943700.4,6222116963.0,6218078005,6225700558,2684950.3,85977,10747.1,0.0,11.0
:42: FSDPNemotronV3Block,PushPop,46824790470,48461886328,192,243879117.0,1921325.0,591107,616551601,281802457.4,5207,27.1,2.0,2.0
:attn_module: DotProductAttention,PushPop,38995211831,86828113059,1152,33850010.3,539615.5,97727,503691507,120448186.0,15552,13.5,4.0,2.0
:5: FSDPNemotronV3Block,PushPop,38932743321,39050678539,192,202774704.8,1320160.5,543679,509978845,234459828.1,5207,27.1,2.0,2.0
:fused_attention: FusedAttention,PushPop,38654594025,86392961462,1152,33554335.1,145152.0,43072,502062137,120183672.2,6336,5.5,5.0,1.5
:mixer: MoE,PushPop,24939655836,35577208323,4416,5647567.0,5158364.0,2950837,41000563,2472451.4,223606,50.6,3.0,3.0
:fwd_bwd_mb0,PushPop,23157144258,41840077872,24,964881010.8,704984781.0,698113405,1491064545,379255475.1,53544,2231.0,1.0,1.0
:experts: GroupedExpertsDeepEP,PushPop,21506885288,23919071607,4416,4870218.6,4522595.0,1958603,38066527,2551947.6,104880,23.8,4.0,0.5
NCCL:ncclReduceScatter,PushPop,18890205950,213419566,7488,2522730.5,641042.0,8256,616951451,26030952.7,7488,1.0,0.0,0.0
:fwd_bwd_mb1,PushPop,18419456904,38122993210,24,767477371.0,735261667.5,703112780,869764828,68037050.8,53623,2234.3,1.0,1.0
NCCL:ncclAllGather,PushPop,18349852929,230939545,7512,2442738.7,343887.0,7968,509995489,27129163.2,7512,1.0,4.5,0.0
:fwd_bwd_mb3,PushPop,16909648207,36463590679,24,704568675.3,703676949.5,694949137,715462816,7985947.1,53335,2222.3,1.0,1.0
:fwd_bwd_mb2,PushPop,16778275495,36902745927,24,699094812.3,701307442.5,689174974,706003057,6163051.9,53586,2232.8,1.0,1.0
:mixer: NemotronV3Mamba2Mixer,PushPop,15310465314,14398307282,4416,3467043.8,1851835.0,694817,389907620,22828208.5,171281,38.8,3.0,2.0
:48: FSDPNemotronV3Block,PushPop,3396789977,990963426,192,17691614.5,2006345.0,1273918,378898317,75346140.3,8311,43.3,2.0,2.0
:9: FSDPNemotronV3Block,PushPop,3093197627,1004737633,192,16110404.3,1951260.0,1284027,389936580,72860397.7,8311,43.3,2.0,2.0
:nvte_cublas_gemm_v2,PushPop,1981447371,2385137944,20448,96901.8,65568.0,7008,6598555,270966.5,24215,1.2,4.9,0.0
:51: FSDPNemotronV3Block,PushPop,1809998159,1144486224,192,9427073.7,7057838.0,3550246,41288245,6996942.2,10586,55.1,2.0,2.0
:1: FSDPNemotronV3Block,PushPop,1688068741,4114926260,192,8792024.7,9431790.0,3594978,13948667,2621880.6,10586,55.1,2.0,2.0
:shared_experts: MLP,PushPop,1497768137,3892116917,4416,339168.5,315551.5,120641,2341360,142323.6,28198,6.4,4.0,2.0
:optimizer_step,PushPop,1459393505,997524700,24,60808062.7,60792239.0,60661736,60969431,95732.9,32016,1334.0,1.0,0.0
:0: FSDPNemotronV3Block,PushPop,1339314290,1696084155,192,6975595.3,1699905.5,705345,136438995,25239506.9,8311,43.3,2.0,2.0
:in_proj: Linear,PushPop,1331556875,936633647,4416,301530.1,291362.0,121504,2586286,145005.2,11201,2.5,4.0,0.0
:49: FSDPNemotronV3Block,PushPop,1292339075,3234612031,192,6730932.7,4877248.0,3128673,12495398,3160285.9,10586,55.1,2.0,2.0
:40: FSDPNemotronV3Block,PushPop,1252095392,604497946,192,6521330.2,4711603.0,3430818,11925482,2519356.3,10586,55.1,2.0,2.0
:6: FSDPNemotronV3Block,PushPop,1128544405,8758007488,192,5877835.4,5578111.0,3344354,11420415,1196784.2,10586,55.1,2.0,2.0
:gate: Gate,PushPop,1096575716,6902687613,4416,248318.8,195296.0,131008,1026652,94740.8,79488,18.0,4.0,0.0
:3: FSDPNemotronV3Block,PushPop,1085494545,672031187,192,5653617.4,5424639.5,3346145,10909446,1435939.8,10586,55.1,2.0,2.0
:8: FSDPNemotronV3Block,PushPop,1085262943,661979630,192,5652411.2,5294739.5,3566882,11298712,1397170.1,10586,55.1,2.0,2.0
:38: FSDPNemotronV3Block,PushPop,1078659913,603986817,192,5618020.4,5062477.0,3455554,10949449,1525659.7,10586,55.1,2.0,2.0
:10: FSDPNemotronV3Block,PushPop,1078038013,3317060584,192,5614781.3,5663106.0,3148386,10448166,1533162.1,10586,55.1,2.0,2.0
:31: FSDPNemotronV3Block,PushPop,1044720609,577783544,192,5441253.2,4791026.0,3198402,10358342,1566853.8,10586,55.1,2.0,2.0
:17: FSDPNemotronV3Block,PushPop,1031301131,581193496,192,5371360.1,4844450.0,3228162,10227846,1458775.2,10586,55.1,2.0,2.0
:45: FSDPNemotronV3Block,PushPop,1031087122,587305611,192,5370245.4,4979417.5,3279714,11063377,1490117.0,10586,55.1,2.0,2.0
:20: FSDPNemotronV3Block,PushPop,1027665467,599853307,192,5352424.3,5197651.0,3412365,10904454,1333381.4,10586,55.1,2.0,2.0
:27: FSDPNemotronV3Block,PushPop,1023561273,624768438,192,5331048.3,5158114.0,3217346,10462022,1366329.8,10586,55.1,2.0,2.0
:13: FSDPNemotronV3Block,PushPop,1020849492,596972903,192,5316924.4,5065381.0,3380994,10693509,1356420.3,10586,55.1,2.0,2.0
:22: FSDPNemotronV3Block,PushPop,1012165126,6806920605,192,5271693.4,5116752.0,3281090,10446534,1341193.2,10586,55.1,2.0,2.0
:24: FSDPNemotronV3Block,PushPop,1005890680,594551359,192,5239014.0,5178015.5,3098082,9603686,1471554.8,10586,55.1,2.0,2.0
:36: FSDPNemotronV3Block,PushPop,993887829,579188712,192,5176499.1,4994125.0,3272450,10520699,1339337.0,10586,55.1,2.0,2.0
:34: FSDPNemotronV3Block,PushPop,990712913,612978567,192,5159963.1,5097749.0,3189250,9992248,1182157.5,10586,55.1,2.0,2.0
:47: FSDPNemotronV3Block,
```
