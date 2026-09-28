# Where the FLOPs go: Nemotron-3-Nano-30B-A3B full SFT on SQuAD in NeMo Automodel

EE 290/194 Scalable AI, Assignment 1 Part B. Stack: NeMo Automodel (`main` @ `8f73178`, fork branch
`mfu-study`), FSDP2 + expert parallelism (EP=8), 1 node × 8× H100 80GB SXM (NVLink), host driver R550.
All numbers are medians over steady-state steps (steps 5-39 of 40) unless noted. Raw data: `mfu/results/`.

**Headline.** The shipped recipe reaches **2.2% useful MFU (1,127 tokens/s/GPU)** on SQuAD. Profiling showed
the GPUs idle 61% of each step; the root cause was not compute or bandwidth but *host-side stalls*: TE's
cuDNN fused attention rebuilt its execution graph for almost every micro-batch because pad-to-longest
batching produces a new sequence length each time (~1 s of CPU per micro-batch, measured). Removing that
(fixed shapes via THD packing), then the padding waste, the materialised 131k-vocab logits, and gradient
accumulation raised throughput **10.5× to 11,836 tokens/s/GPU (23.2% useful MFU)** with an unchanged loss
curve. The remaining gap to the upstream synthetic-data benchmark (33.6%) is dominated by DeepEP and
FSDP communication on the critical path and per-layer CPU launch overhead.

![useful MFU by configuration](results/figures/mfu_by_config.png)

---

## 1. Setup and exact reproduction

| | |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`: 52 layers = 23 Mamba2 + 23 MoE (128 routed experts, top-6, + 1 shared; ReLU² MLPs) + 6 GQA attention (32 q / 2 kv heads); hidden 2688; vocab 131,072; 31.6B total / 3.2B active matmul params per token |
| Data | `rajpurkar/squad` train, Automodel `make_squad_dataset` (chat template, loss on answer tokens only) |
| Recipe | `train_ft` full-parameter SFT; config `mfu/configs/nemotron_nano_v3_squad_baseline.yaml` = shipped `examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag.yaml` with the dataset swapped to SQuAD (all deviations listed in the file header; none touch the training math) |
| Parallelism | FSDP2 over 8 ranks, EP=8 (16 experts/GPU), no TP/PP/CP, no activation checkpointing |
| Batch | global 256 samples, local 8 per GPU → 4 micro-batches of grad accumulation; Adam, lr 1e-5 |
| Kernels (backend defaults) | TE linear + TE/cuDNN fused attention, `torch._grouped_mm` experts, DeepEP intranode dispatcher, mamba-ssm Triton SSD kernels |
| Software | torch 2.10.0+cu128, TE 2.19.0, cuDNN 9.10.2, mamba-ssm 2.3.0, causal-conv1d 1.6.0, DeepEP @ `10d4dd7` (image `mfu/docker/Dockerfile.cu128`) |

Why a custom image: every published `nemo-automodel` container (25.11 → 26.08) is CUDA 13 and needs driver ≥ 580;
this host runs R550 (CUDA 12.4) and CUDA-13 forward compatibility does not exist on R550 (verified). The image keeps
every `uv.lock` version but uses the cu128 build of the locked torch and compiles TE / mamba / DeepEP against it.
Porting issues found and fixed on the way (each documented in the Dockerfile): CUDA-13-only wheels in the export,
DeepEP built without NVSHMEM (single node needs only intranode kernels), a driver-580 NVML/libcuda pulled in by
`libnvidia-ml-dev` that shadowed the host driver, and **two cuDNN versions loaded in one process** (system 9.8 for TE,
wheel 9.10 for torch) which segfaulted in `fused_attn_bwd`. One framework bug was fixed: `fused_a2a._is_nvshmem_available`
treated "SM90 compiled" as "NVSHMEM available" and crashed NVSHMEM-less DeepEP builds.

Reproduce (from `assignments/a01/p2`, see `mfu/README.md`):

```bash
docker build -f mfu/docker/Dockerfile.cu128 -t automodel-mfu:cu128 .
mfu/docker.sh -- python mfu/check_env.py                              # kernel smoke test
mfu/run.sh --name baseline                                            # 40 steps -> mfu/runs/<ts>_baseline/
mfu/run.sh --name baseline_nsys --profile nsys --steps 14 --nsys-steps 10:13
mfu/docker.sh -- python mfu/analyze_nsys.py mfu/runs/<ts>_baseline_nsys   # GPU time budget
python3 mfu/analyze_cpu.py mfu/runs/<ts>_baseline_nsys                    # host-side breakdown
python3 mfu/summarize.py mfu/runs/*                                       # comparison table
```

**Metric definitions.** *Useful MFU* = real (non-padding) tokens/s/GPU × 19.36 GFLOP/token ÷ 989 TFLOP/s (H100 SXM
dense BF16). 19.36 GFLOP/token = 6 × 3.227B active matmul params + attention scores at SQuAD lengths; it matches
Automodel's `flops_utils.nemotronh_flops` within 0.1% (checked). The recipe's own MFU column counts padded positions
(and, for packed rows, attention across the whole 4096-token row), so it overstates useful work; both are reported.

## 2. Baseline and what "current OSS best" looks like

| run | step (s) | tokens/s/GPU | recipe MFU | useful MFU | pad efficiency | peak mem |
|---|---|---|---|---|---|---|
| **Baseline** (shipped recipe on SQuAD) | 5.865 | 1,127 | 3.40% | **2.21%** | 0.643 | 60.1 GiB |
| Upstream benchmark recipe `llm_benchmark/nemotron/nemotron_nano_v3_te_deepep.yaml` (same image) | 15.56 | 16,844 | 33.6% | 33.6% | 1.00 | 63.9 GiB |

The upstream benchmark is the best configuration NVIDIA ships for this model, but it is not a like-for-like
workload: synthetic 4096-token sequences (no padding, fixed shape), `fake_balanced_gate` (perfectly balanced
expert load), activation checkpointing, and 2.1M tokens/step with 16-way gradient accumulation (optimizer and
gradient sync amortised over 32× more tokens than a SQuAD step). It is an upper bound for this node, not a target
the real-data recipe can match exactly.

## 3. Profiling results

nsys 2024.6 capture of steps 10-12 on all 8 ranks (`--trace cuda,nvtx`, capture window via `cudaProfilerStart/Stop`),
with per-module NVTX (Automodel `autonvtx`) plus added step-phase ranges. Profiling overhead: 6.4 s vs 5.9 s
unprofiled per step. Traces: `mfu/runs/*_baseline_nsys/profile.nsys-rep`, `mfu/runs/*_pack4096_nsys/profile.nsys-rep`.

![GPU time budget](results/figures/gpu_time_budget.png)

**Baseline, per 6.5 s step (mean of 8 GPUs):**

| bucket | ms | % |
|---|---|---|
| GPU idle (no kernel on any stream) | 4,016 | **61.4%** |
| compute kernels (union) | 719 | 11.0% |
| communication kernels (union) | 1,917 | 29.3% |
| of which overlapped with compute | 106 | 1.6% |

Summed kernel time: NCCL reduce-scatter 787 ms, all-gather 765 ms (both `RING_LL`, i.e. small messages; the
reduce-scatter runs in fp32), DeepEP 624 ms, GEMM 341 ms, Mamba 75 ms, attention 3.5 ms. Only ~340 ms of a 6.5 s
step is matrix math.

![baseline timeline](results/figures/timeline_baseline.png)

The CPU side tells the story. CUDA API calls account for only ~0.8 s of host time per step, yet each micro-batch
takes **~1.5 s of host time to issue while the GPU needs ~0.7 s**, so the step is host-bound. Per-module NVTX shows
where: the 6 attention layers take **3.68 s of host time per rank per step**; their median call is 0.28 ms, but
every micro-batch contains two calls of ~470 ms and ~590 ms (the first attention layer in forward and the first in
backward), on every rank (`cpu_breakdown.md`). During those windows no kernels run anywhere (red blocks above);
ranks finish them at slightly different times, so the next FSDP all-gather / reduce-scatter absorbs the skew as
wait time, which is why NCCL kernel time looks large.

> **TODO (Nsight screenshots, required by the rubric):** open the two `.nsys-rep` files in the Nsight Systems GUI and
> capture (1) baseline, one `train_step_11` on rank 0 with the NVTX row expanded to `fused_attention: FusedAttention`
> (the ~0.5 s ranges) and the CUDA HW row empty underneath; (2) the packed run, one `train_step_25` showing dense
> GEMM rows and the DeepEP / NCCL kernels interleaved on the compute stream. The PNG timelines above are rendered
> from the same traces and can stand in until then.

## 4. Diagnosis: hypotheses and evidence

**H1 (primary). Dynamic shapes → cuDNN graph rebuilds → host stalls.** `default_collater` pads each micro-batch to
its longest sample, and SQuAD lengths vary (mean 206, p50 193, p90 294, p99 429, max 1037 tokens;
`results/squad_lengths.json`), so nearly every micro-batch has a new `[8, S]` shape. TE's cuDNN fused-attention
backend caches an execution graph per shape; a miss builds a new forward and backward graph.
*Evidence:* (a) the two slow calls per micro-batch sit exactly in the first attention layer of forward and backward;
(b) an isolated microbenchmark of the same layer configuration (`mfu/bench_te_attn_shapes.py`, 1 GPU) measures
**1,013-1,055 ms for the first fwd+bwd at a new length vs 1.23-1.34 ms for a repeat** (`results/te_attn_shapes.json`),
matching 470 + 590 ms in the trace; (c) fixing the shapes removes the stall (Section 6).
*Why it costs so much:* the stall is on the host, so the GPU queue drains; all ranks hit it together, so the whole
node idles.

**H2. Padding wastes a third of the compute.** With 8 samples padded to the longest, only 64.3% of positions are
real tokens (measured `num_tokens / num_input_positions`; 66.8% predicted from the length distribution). Packing
into 4096-token rows fills 97.2%.

**H3. Micro-batches too small for this model.** ~2.5k positions per GPU per micro-batch through 52 layers gives
tiny GEMMs, and the per-layer host cost (Python dispatch, many small Triton/TE launches, DeepEP's host-side
notify, FSDP hooks) is paid per micro-batch regardless of its size. Four micro-batches per step also mean four
FSDP all-gather/reduce-scatter rounds (`defer_fsdp_grad_sync: false`) in small `RING_LL` messages.

**H4. Materialised logits limit batch size.** The LM head produces `[tokens, 131,072]` logits upcast to fp32 for
the loss (4 GiB per 8k tokens); 2 packs per GPU OOMs at exactly that 4 GiB allocation.

## 5. Interventions (what changed and why it should help)

| id | change (config only unless noted) | mechanism |
|---|---|---|
| A | `collate_fn.pad_seq_len_divisible: 64` (`configs/nemotron_nano_v3_squad_pad64.yaml`) | ~10 distinct shapes instead of hundreds → cuDNN graphs built once, then cached (H1) |
| B | THD packing into 4096-token rows (`packed_sequence_size: 4096, packing_strategy: thd`, THD collater; 16 packs/step, 1/GPU) (`configs/nemotron_nano_v3_squad_pack4096.yaml`) | fixed shape (TE buckets ragged batch/token counts, so graphs stay cached) (H1); 97% fill (H2); 1.6× more tokens per micro-batch (H3) |
| C | B + `loss_fn: FusedLinearCrossEntropy`, `model.output_hidden_states: true` | never materialises the fp32 logits: less memory and fewer passes over a 131k-wide tensor (H4) |
| D | C + `local_batch_size: 2` (2 packs/GPU, no grad accumulation) | same tokens/step; half the FSDP rounds per token, 2× larger GEMMs, host cost amortised over 2× tokens (H3); enabled by C's memory saving |

## 6. Results and ablations

| config | step (s) | tokens/s/GPU | useful MFU | recipe MFU | pad eff. | peak mem | loss step 0 → 39 |
|---|---|---|---|---|---|---|---|
| Baseline | 5.865 | 1,127 | 2.21% | 3.40% | 0.643 | 60.1 GiB | 5.243 → 0.289 |
| A: pad to ×64 | 2.006 | 3,295 (2.9×) | 6.45% | 10.90% | 0.586 | 61.9 GiB | 5.244 → 0.290 |
| B: THD packing | 0.860 | 9,232 (8.2×) | 18.07% | 19.00% | 0.971 | 54.9 GiB | 5.109 → 0.256 |
| C: B + fused linear-CE | 0.826 | 9,598 (8.5×) | 18.79% | 19.77% | 0.971 | 49.0 GiB | 5.111 → 0.259 |
| **D: C + 2 packs/GPU** | **0.676** | **11,757 (10.4×)** | **23.02%** | 24.18% | 0.971 | 55.7 GiB | 5.110 → 0.259 |
| **E: D + `gc_every_steps: 50`** (`configs/nemotron_nano_v3_squad_best.yaml`) | **0.672** (mean 0.763) | **11,836 (10.5×)** | **23.17%** | 24.32% | 0.971 | 56.8 GiB | 5.110 → 0.259 |
| B + 2 packs/GPU without C | OOM (4 GiB logits allocation) | | | | | | |

* A alone confirms H1 in the full model: identical math (loss 5.244 → 0.290 vs 5.243 → 0.289), 2.9× faster, even
  though it adds padding (pad eff. 0.643 → 0.586). Steps still spike while new ×64 buckets are first seen, then settle
  at ~1.5 s.
* B's packed profile (967 ms/step): GPU busy 82% (was 39%), compute 54% (was 11%), attention host time 6.6 ms/step
  (was 3,590 ms). Loss differs slightly from the baseline because a packed step holds ~310 samples instead of 256;
  the curves track each other (5.11 → 0.26).
* C: −6 GiB peak memory, +4% throughput, loss identical to B within 0.02.
* D: +22% over C from removing gradient accumulation, with the same tokens per step.

![step times](results/figures/step_times.png)

**Where the best config still loses time** (packed profile, 967 ms/step): exposed communication 28% (DeepEP
dispatch/combine 211 ms/step on the compute stream; FSDP all-gather 81 + reduce-scatter 69 ms, 3.7% overlapped),
GPU idle 18% (host: Mamba mixers 245 ms and MoE 256 ms of CPU per step, i.e. ~2.7-2.9 ms of Python/launch work per
layer call), non-GEMM compute ~35% of compute (optimizer 61 ms, copies/concat 56 ms, elementwise 55 ms, Mamba 52 ms).

![packed timeline](results/figures/timeline_pack4096.png)

## 7. Recommendations: what to optimise next (ranked by evidence × expected gain)

1. **Make fixed shapes the default for SQuAD-style SFT.** Ship Nano-V3 SQuAD with THD packing (or at least
   `pad_seq_len_divisible`); the cuDNN re-plan cost (~1 s per new shape, per rank) turns any variable-length
   pad-to-longest recipe into a host-bound one. Upstream PR candidate: a `nemotron_nano_v3_squad.yaml` with packing,
   plus a warning when TE fused attention sees many distinct shapes.
2. **Take DeepEP off the critical path (≈22% of the packed step).** Try `backend.dispatcher_async_dispatch: true`
   (prepared but not measured: a groupmate was actively using GPU 0 and the run guard declined to start)
   and overlap the shared-expert MLP with dispatch/combine on a side stream (Automodel already has
   `shared_expert_overlap` for Kimi K3; Nemotron-V3 does not opt in). Expected: hide most of the 211 ms.
3. **Overlap and shrink FSDP traffic (≈16%, 3.7% overlapped).** Larger FSDP units/buckets (the `RING_LL` protocol
   shows messages are small), bf16 reduce-scatter instead of fp32 (halves bytes; check convergence), explicit
   forward/backward prefetch, and `defer_fsdp_grad_sync: true` whenever gradient accumulation is used.
4. **Cut per-layer host overhead (GPU idle 18%).** CUDA graphs for the Mamba mixer and MoE router/permute
   (Automodel's partial CUDA-graph manager), or `torch.compile` of the Mamba pre/post-processing; bigger micro-batches
   help for the same reason (D: +22%).
5. **Optimizer (6%).** Fused Adam (`fused=True` / TE FusedAdam) instead of the foreach path; at 65k tokens/step the
   step is short enough that a 61 ms optimizer matters. Larger global batches amortise it further (the upstream
   benchmark's 2.1M-token step is one reason it reaches 33.6%).
6. **Control Python GC** (`gc_every_steps`): removes the periodic ~1 s pauses (−9% mean step time on D, below).

**GC experiment (spike attribution).** On config D, the ~1 s spikes occur *between* training steps (every captured
`train_step` NVTX range is ~0.95 s, so the extra time is outside forward/backward/optimizer). SQuAD is held as Python
lists of token ids (tens of millions of int objects), so a generation-2 collection has to walk a very large heap.
With `step_scheduler.gc_every_steps: 50` (disables automatic GC; manual gen-1 collection every 50 steps, as in
torchtitan), 3 of the 6 spikes in steps 5-39 disappear (steps 18, 19, 25) and mean step time drops 0.836 → 0.763 s
(−9%; median unchanged at 0.67 s, useful MFU 23.2%). The other 3 spikes (steps 6, 8, 12) occur at the same steps in
both runs, i.e. they are data-dependent first-time builds for new shape buckets, and stop after step 12.
Recommendation: set `gc_every_steps` (or `gc.freeze()` after dataset construction) in the SFT recipes.

## 8. Reproducibility notes

* Seeds: `rng.seed 1111` (ranked), shuffled SQuAD; data order is deterministic across runs, which let the
  profiled window be chosen to exclude known spike steps.
* Every run directory records the exact command, resolved config, git commit + diff, image and `nvidia-smi`
  (`mfu/runs/<ts>_<name>/`); small artefacts are copied to `mfu/results/`.
* 40 steps per configuration; statistics exclude steps 0-4 (startup, first graph builds). Longer runs would lower
  the pad-to-×64 and packed means further as caches warm up; medians are reported for that reason.
* Shared machine: `run.sh` refuses to start if any GPU shows >4 GiB used or >10% utilisation. A groupmate's idle
  notebook held 0.6-1.8 GiB on GPU 0 (0% utilisation) during some runs.
* Caveats: custom CUDA 12.8 stack instead of NVIDIA's CUDA 13 image (absolute numbers may differ from the NGC
  container); DeepEP built intranode-only; nsys adds ~8% step time; the recipe's FLOPs formula is used unchanged for
  "recipe MFU".
