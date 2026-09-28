# nsys breakdown (per step, 3 captured steps)

GPUs analysed: [0, 1, 2, 3, 4, 5, 6, 7]

## Time budget (mean over GPUs)

| bucket | ms | % of window |
|---|---|---|
| wall window | 784.1 | 100.0% |
| GPU busy (any kernel) | 743.3 |  94.8% |
| GPU idle (no kernel) | 40.8 |   5.2% |
| compute kernels (union) | 559.9 |  71.4% |
| communication kernels (union) | 235.9 |  30.1% |
|   comm overlapped with compute | 52.4 |   6.7% |
|   comm exposed (not overlapped) | 183.4 |  23.4% |

## Summed kernel time by category (mean over GPUs; overlapping streams can sum >100%)

| category | ms | % of window |
|---|---|---|
| gemm | 298.2 |  38.0% |
| comm:deepep | 161.2 |  20.6% |
| mamba | 71.5 |   9.1% |
| comm:nccl_allgather | 57.0 |   7.3% |
| elementwise/reduce | 52.5 |   6.7% |
| comm:nccl_reducescatter | 46.3 |   5.9% |
| other | 31.8 |   4.1% |
| optimizer | 29.4 |   3.8% |
| norm | 25.0 |   3.2% |
| moe:routing/permute | 24.7 |   3.1% |
| memcpy/cat/copy | 23.9 |   3.0% |
| attention | 4.1 |   0.5% |
| comm:nccl_allreduce | 2.7 |   0.3% |
| gemm:grouped | 0.4 |   0.1% |

## Top kernels on GPU 0

| kernel | category | ms | calls |
|---|---|---|---|
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 61.9 | 138 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` | comm:nccl_allgather | 59.2 | 237 |
| `void deep_ep::intranode::cached_notify_combine<(int)8>(void **, int *, int, int, int, int ` | comm:deepep | 55.7 | 138 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 51.2 | 138 |
| `void cutlass::device_kernel<at::cuda::detail::enable_3x_kernel_for_sm9x<cutlass::gemm::ker` | gemm | 48.6 | 138 |
| `void deep_ep::intranode::dispatch<(int)8, (int)768, (int)8192>(int4 *, float *, float *, i` | comm:deepep | 48.2 | 138 |
| `void deep_ep::intranode::combine<__nv_bfloat16, (int)8, (int)768, (int)4096>(T1 *, float *` | comm:deepep | 48.2 | 138 |
| `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>` | comm:nccl_reducescatter | 38.5 | 165 |
| `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | gemm | 29.1 | 138 |
| `void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::FusedOptimize` | optimizer | 27.5 | 567 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN` | gemm | 20.2 | 69 |
| `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` | gemm | 20.0 | 69 |
| `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at:` | elementwise/reduce | 17.3 | 927 |
| `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BF` | elementwise/reduce | 11.8 | 555 |
| `_permute_kernel` | moe:routing/permute | 10.6 | 138 |
| `_unpermute_kernel` | moe:routing/permute | 10.6 | 138 |
| `_cce_backward_kernel` | other | 9.7 | 3 |
| `_layer_norm_bwd_kernel` | norm | 9.6 | 69 |
| `void causal_conv1d_channellast_fwd_kernel<Causal_conv1d_channellast_fwd_kernel_traits<(int` | mamba | 8.8 | 138 |
| `nvjet_tst_256x128_64x4_2x1_v_bz_coopA_NTT` | gemm | 8.7 | 69 |
| `void causal_conv1d_channellast_bwd_kernel<Causal_conv1d_channellast_bwd_kernel_traits<(int` | mamba | 8.4 | 69 |
| `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` | gemm | 8.0 | 69 |
| `_chunk_scan_chunk_state_bwd_dx_kernel` | mamba | 7.9 | 69 |
| `_chunk_state_fwd_kernel` | mamba | 7.4 | 138 |
| `void deep_ep::intranode::cached_notify_dispatch<(int)8>(const int *, int, void **, int **,` | comm:deepep | 6.5 | 69 |

## Per-GPU spread

| GPU | window ms | busy % | idle % | comm exposed % |
|---|---|---|---|---|
| 0 | 783.9 |  94.0% |   6.0% |  23.1% |
| 1 | 784.1 |  95.0% |   5.0% |  24.7% |
| 2 | 784.1 |  95.0% |   5.0% |  22.1% |
| 3 | 784.1 |  93.8% |   6.2% |  21.6% |
| 4 | 784.0 |  95.1% |   4.9% |  23.1% |
| 5 | 784.2 |  95.1% |   4.9% |  24.7% |
| 6 | 784.1 |  95.2% |   4.8% |  24.4% |
| 7 | 784.0 |  95.2% |   4.8% |  23.3% |

## NVTX ranges projected onto the GPU (nsys nvtx_gpu_proj_sum, all ranks)

```
Range,Style,Total Proj Time (ns),Total Range Time (ns),Range Instances,Proj Avg (ns),Proj Med (ns),Proj Min (ns),Proj Max (ns),Proj StdDev (ns),Total GPU Ops,Avg GPU Ops,Avg Range Lvl,Avg Num Child
:mixer: MoE,PushPop,10740510069,4553626350,1104,9728722.9,9626644.0,6629630,17957378,2050382.7,75624,68.5,3.0,3.0
:experts: GroupedExpertsDeepEP,PushPop,8896479830,2783511309,1104,8058405.6,7655927.5,5340900,15946508,1762691.8,25392,23.0,4.0,0.5
:fwd_bwd_mb0,PushPop,6441987843,14556391439,24,268416160.1,268463458.0,260463255,276370528,6315453.0,66240,2760.0,1.0,10.0
:FSDPNemotronHForCausalLM: FSDPNemotronHForCausalLM,PushPop,6330089834,6225263027,24,263753743.1,264089096.5,256434827,271128815,5928711.5,64608,2692.0,2.0,8.0
:train_step_20,PushPop,6290231188,6294692138,8,786278898.5,786288390.5,785992568,786543707,213300.0,27544,3443.0,0.0,8.0
:model: FSDPNemotronV3Model,PushPop,6282193948,6164925912,24,261758081.2,262234307.0,254172311,269085988,5780052.1,64032,2668.0,3.0,107.0
:train_step_22,PushPop,6242626032,6245456245,8,780328254.0,780380366.5,779747772,780865586,477061.0,27544,3443.0,0.0,8.0
:mixer: NemotronV3Mamba2Mixer,PushPop,6016570937,5185173153,1104,5449792.5,5587832.5,2578130,6842122,629144.0,45264,41.0,3.0,2.0
:train_step_21,PushPop,6012845525,6015563708,8,751605690.6,751706692.0,751318004,751782675,200252.5,27544,3443.0,0.0,8.0
:in_proj: Linear,PushPop,1467692625,134790482,1104,1329431.7,1324387.5,872415,2233688,426025.3,3312,3.0,4.0,0.0
NCCL:ncclAllGather,PushPop,1367169172,58889734,1896,721080.8,535409.0,9888,4577512,629431.7,1896,1.0,4.5,0.0
:shared_experts: MLP,PushPop,1137746518,964862073,1104,1030567.5,1305372.0,699232,1604267,302202.2,27600,25.0,4.0,2.0
NCCL:ncclReduceScatter,PushPop,1110401561,53417578,1872,593163.2,234143.5,8384,22248863,1826319.5,1872,1.0,0.0,0.0
:nvte_cublas_gemm_v2,PushPop,811605287,343944633,5064,160269.6,178945.0,19840,271807,62498.2,9528,1.9,5.0,0.0
:optimizer_step,PushPop,670973209,191114813,24,27957217.0,27936299.5,27745456,28153039,102479.4,4680,195.0,1.0,0.0
:grad_clip,PushPop,650945800,2803578640,24,27122741.7,27111162.5,27008237,27283545,71781.4,11328,472.0,1.0,2.0
:gate: Gate,PushPop,570945994,571884905,1104,517161.2,533711.5,476576,708765,25536.5,19872,18.0,4.0,0.0
:40: FSDPNemotronV3Block,PushPop,560702699,222564447,48,11681306.2,9843142.0,7951471,18151201,4145249.9,3408,71.0,2.0,2.0
:mixer: NemotronV3Attention,PushPop,550512174,1515323168,288,1911500.6,2242321.5,1058655,24358930,1774089.6,15120,52.5,3.0,5.0
:34: FSDPNemotronV3Block,PushPop,527745442,181056206,48,10994696.7,11246368.5,8506344,12713356,1668969.6,3408,71.0,2.0,2.0
:51: FSDPNemotronV3Block,PushPop,526345767,231644223,48,10965536.8,9264704.5,8081334,16795451,3012958.4,3408,71.0,2.0,2.0
:1: FSDPNemotronV3Block,PushPop,498670466,279721450,48,10388968.0,10172605.0,9635074,11536154,701619.9,3408,71.0,2.0,2.0
:24: FSDPNemotronV3Block,PushPop,495853929,236442282,48,10330290.2,10316023.5,8139377,12701579,2016230.6,3408,71.0,2.0,2.0
:17: FSDPNemotronV3Block,PushPop,495717097,208052537,48,10327439.5,10339947.5,8078827,12563916,2098085.6,3408,71.0,2.0,2.0
:13: FSDPNemotronV3Block,PushPop,491881799,194338305,48,10247537.5,10233374.5,8102054,12131545,1697715.1,3408,71.0,2.0,2.0
:20: FSDPNemotronV3Block,PushPop,491433225,211907527,48,10238192.2,10318996.0,7888088,12499148,1973537.4,3408,71.0,2.0,2.0
:38: FSDPNemotronV3Block,PushPop,478296165,208907915,48,9964503.4,9734067.5,7556662,12307373,1878555.3,3408,71.0,2.0,2.0
:22: FSDPNemotronV3Block,PushPop,476354333,238752999,48,9924048.6,9887738.5,7823695,12054967,1937751.5,3408,71.0,2.0,2.0
:47: FSDPNemotronV3Block,PushPop,469592937,215109436,48,9783186.2,9676614.5,7907590,11951090,1761582.4,3408,71.0,2.0,2.0
:29: FSDPNemotronV3Block,PushPop,469511658,209699843,48,9781492.9,9807691.0,7703640,12059819,1851947.7,3408,71.0,2.0,2.0
:15: FSDPNemotronV3Block,PushPop,469220981,233109243,48,9775437.1,9732608.0,7687309,11878297,1910938.8,3408,71.0,2.0,2.0
:43: FSDPNemotronV3Block,PushPop,466742612,195539913,48,9723804.4,9989450.0,7483393,11743334,1628705.9,3408,71.0,2.0,2.0
:27: FSDPNemotronV3Block,PushPop,464683874,187923364,48,9680914.0,9590247.5,7423394,12162635,1747949.1,3408,71.0,2.0,2.0
:6: FSDPNemotronV3Block,PushPop,461494432,188917079,48,9614467.3,9658876.5,7856428,11125255,1312699.4,3408,71.0,2.0,2.0
:49: FSDPNemotronV3Block,PushPop,459076760,227360508,48,9564099.2,9546078.5,7598542,11567931,1759819.8,3408,71.0,2.0,2.0
:8: FSDPNemotronV3Block,PushPop,457372869,216092017,48,9528601.4,9449446.0,7501560,11917512,1792480.8,3408,71.0,2.0,2.0
:10: FSDPNemotronV3Block,PushPop,453416332,218945787,48,9446173.6,9411136.5,7556764,11481180,1745288.4,3408,71.0,2.0,2.0
:down_proj: Linear,PushPop,446703608,390875171,1104,404622.8,489741.5,291583,525437,104361.4,12144,11.0,5.0,7.5
:45: FSDPNemotronV3Block,PushPop,445503331,217007153,48,9281319.4,9265942.0,7435201,11121218,1692805.8,3408,71.0,2.0,2.0
:31: FSDPNemotronV3Block,PushPop,442598502,214508396,48,9220802.1,9326012.5,7222754,11166962,1669700.4,3408,71.0,2.0,2.0
:up_proj: Linear,PushPop,440764607,400936615,1104,399243.3,485473.0,280800,534275,101638.0,12144,11.0,5.0,7.5
:36: FSDPNemotronV3Block,PushPop,429469067,229970485,48,8947272.2,8877573.0,7265520,10725945,1589621.8,3408,71.0,2.0,2.0
:3: FSDPNemotronV3Block,PushPop,420982648,179201520,48,8770471.8,9105130.5,6816924,10103326,1209424.4,3408,71.0,2.0,2.0
:norm: RMSNorm,PushPop,314123027,436177228,2544,123476.0,122879.5,114176,216670,12867.3,3816,1.5,3.0,1.0
:41: FSDPNemotronV3Block,PushPop,298133884,140848973,48,6211122.6,6142193.5,5204165,6879274,486625.8,2088,43.5,2.0,2.0
:35: FSDPNemotronV3Block,PushPop,287757097,143430295,48,5994939.5,5792696.0,5336902,6906538,363913.5,2088,43.5,2.0,2.0
:18: FSDPNemotronV3Block,PushPop,284144257,139557780,48,5919672.0,5804895.0,4840256,6696335,324913.8,2088,43.5,2.0,2.0
:25: FSDPNemotronV3Block,PushPop,283976281,631089502,48,5916172.5,5810061.5,5554813,6715068,262315.0,2088,43.5,2.0,2.0
:39: FSDPNemotronV3Block,Pu
```
