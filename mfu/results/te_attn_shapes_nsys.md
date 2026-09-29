# Nsight trace of the TE attention shape microbenchmark

Trace: `mfu/results/te_attn_shapes_nsys/profile.nsys-rep` (committed, 196 KB; 1 GPU, `--trace cuda,nvtx,cudnn`), produced by
`MFU_GPUS=7 mfu/docker.sh -- nsys profile -o mfu/runs/te_attn_shapes_nsys/profile --trace cuda,nvtx,cudnn python mfu/bench_te_attn_shapes.py`.
NVTX ranges `first_call_seq<S>` (first fwd+bwd at a new padded length) and `repeat_x5_seq<S>` (5 more calls, same length),
for S in {193, 257, 311, 384, 402, 449, 512}. Means per range (7 lengths):

| range | wall | GPU kernel time | CUDA API time | host time outside CUDA API |
|---|---|---|---|---|
| first call | 1,022.6 ms | 0.33 ms (34 kernels) | 10.7 ms | ~1,012 ms (99%) |
| 5 repeats | 7.9 ms (1.6 ms/call) | 1.66 ms (170 kernels) | 1.7 ms | ~6 ms |

The GPU work per call is identical (0.33 ms, 34 kernels); the first call spends ~1 s on the host outside any CUDA API
call while the GPU is idle, i.e. one-time host-side setup (cuDNN execution-graph construction for the new shape).
Largest CUDA API call inside a first-call range: `cudaGetDeviceProperties` 7 ms. CPU IP sampling and cuDNN API tracing
were not available inside the container, so the specific host functions are not recorded.
