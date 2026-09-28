# MFU study: Nemotron-3-Nano-30B-A3B full SFT on SQuAD

Tooling for measuring throughput/MFU of Automodel's `train_ft` recipe on
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (hybrid Mamba2 / attention / MoE, ~31.6B total, ~3.5B active
params) and for profiling where time goes. Target: 1 node, 8x H100 80GB SXM.

## Layout

| path | what |
|---|---|
| `configs/nemotron_nano_v3_squad_baseline.yaml` | Baseline: shipped Nano-V3 full-SFT recipe with the dataset swapped to SQuAD (deviations listed in the file header) |
| `docker/Dockerfile.cu128` | Runtime image for this host (driver R550 / CUDA 12.4): locked deps + torch 2.10.0+cu128, TE/mamba-ssm/causal-conv1d/DeepEP built from source |
| `docker/requirements-cu128.txt` | `uv export` of uv.lock (extras cuda, moe, tracking) minus CUDA-13 wheels and source-built packages |
| `docker.sh` | Runs a command in `automodel-mfu:cu128` with this checkout mounted over `/opt/Automodel`, as your host user |
| `run.sh` | One run end-to-end: GPU-busy guard, run dir + provenance, torchrun (optionally under `nsys`), summary |
| `summarize.py` | Steady-state table from `training.jsonl`: step time, tokens/s/GPU, MFU, padding efficiency, memory |
| `analyze_nsys.py` | GPU time budget from an nsys capture: busy/idle, compute vs comm, exposed comm, per-category kernel time, top kernels, NVTX phases |
| `configs/nemotron_nano_v3_squad_{pad64,pack4096,best}.yaml` | Interventions A, B and the best 8-GPU config (round 2); see `REPORT.md` |
| `results/experiments_8gpu_round2.md` | 8-GPU round-2 ladder (6-GPU findings re-tested on 8 GPUs) |
| `configs/nemotron_nano_v3_squad_best_6gpu.yaml` | Best config on 6 GPUs (EP=2, 8 packs, full AC except attention, bf16 reduce, TE norm, 64 DeepEP SMs) |
| `bench_nccl.py` | NCCL all-gather / reduce-scatter bus bandwidth at FSDP message sizes |
| `results/experiments_6gpu.md` | Every 6-GPU experiment with its result and verdict |
| `notes/lecture_techniques.md` | Course-lecture techniques mapped to the measured profile |
| `analyze_cpu.py` | Host-side view of an nsys capture: CUDA API time, CPU time per NVTX range, outlier calls (e.g. cuDNN graph builds) |
| `bench_te_attn_shapes.py` | 1-GPU microbenchmark: TE fused attention first call at a new sequence length vs repeat |
| `squad_lengths.py` | SQuAD token-length distribution and the padding / packing efficiency it implies |
| `plots.py` | Report figures -> `results/figures/` (`uv run --no-project --with matplotlib python mfu/plots.py`) |
| `REPORT.md` | Findings, evidence, interventions, results, recommendations |
| `results/` | Small, committed artefacts per run (config, command, per-step metrics, nsys/CPU breakdowns) |
| `runs/` | Full run outputs incl. `.nsys-rep` traces (git-ignored) |

Recipe hooks added in `nemo_automodel/recipes/llm/train_ft.py` (all off by default):
* `profiling.nsys_start_step` / `profiling.nsys_end_step`: `cudaProfilerStart/Stop` around optimizer steps `[start, end)`.
* `nvtx: true` additionally emits `train_step_<n>`, `fwd_bwd_mb<i>`, `grad_clip`, `optimizer_step` ranges
  (on top of the existing per-module autonvtx ranges).
* Extra per-step metrics: `step_time`, `num_input_positions_per_step` (incl. padding).

## Why a custom image

All published `nvcr.io/nvidia/nemo-automodel` tags (25.11 to 26.08) are CUDA 13.x and require driver >= 580;
this host has driver 550.90.07 (CUDA 12.4) and CUDA-13 forward compatibility is not available on R550
(verified: `torch.cuda.is_available()` is False inside 26.06.00 even with its compat libs). main's
`uv.lock` also pins `torch==2.10.0+cu130`. `Dockerfile.cu128` keeps every locked version but swaps the torch
build to cu128 and compiles the native extensions against it. To regenerate the requirements file:
`uv export --frozen --no-hashes --no-emit-project --no-header --no-annotate --extra cuda --extra moe --extra tracking`
filtered as described in the Dockerfile header.

## One-time setup

```bash
# HF cache (the model is ~59 GB). Host-side download, reused by the container via a bind mount:
uvx --from huggingface_hub hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
uvx --from huggingface_hub hf download rajpurkar/squad --repo-type dataset
docker build -f mfu/docker/Dockerfile.cu128 -t automodel-mfu:cu128 .
# W&B: key in ~/.netrc (machine api.wandb.ai); run.sh forwards it as WANDB_API_KEY.
```

## Reproduce the baseline

```bash
cd assignments/a01/p2
mfu/run.sh --name baseline                      # 40 steps, metrics -> mfu/runs/<ts>_baseline/, W&B scalingai-f26/nemotron-nano-v3-mfu
mfu/run.sh --name baseline_nsys --profile nsys --steps 14 --nsys-steps 10:13
mfu/docker.sh -- python mfu/analyze_nsys.py mfu/runs/<ts>_baseline_nsys   # GPU-side breakdown (+ exports profile.sqlite)
python3 mfu/analyze_cpu.py mfu/runs/<ts>_baseline_nsys                   # host-side breakdown
mfu/run.sh --name best -c mfu/configs/nemotron_nano_v3_squad_best.yaml  # 8-GPU best (15.2k tok/s/GPU, 13.5x)
mfu/run.sh --name best6 -c mfu/configs/nemotron_nano_v3_squad_best_6gpu.yaml --devices 2,3,4,5,6,7  # 6-GPU best (13.9k tok/s/GPU)
MFU_GPUS=none mfu/docker.sh -- python -m pytest -q tests/unit_tests/moe   # CPU-only unit tests (no GPU touched)
python3 mfu/summarize.py mfu/runs/<ts>_baseline mfu/runs/<ts>_other   # compare runs
```

Any recipe key can be overridden after `--`, e.g.
`mfu/run.sh --name mbs4 -- --step_scheduler.local_batch_size=4`.

The machine is shared: `run.sh` refuses to start if any GPU has >4 GiB in use or >10% utilization (`--force` to override).

## Metric definitions

* **MFU (recipe)**: analytic model FLOPs (`flops_utils.nemotronh_flops`, 6·N·T style incl. Mamba scan,
  attention and routed-expert terms) of the padded `input_ids`, divided by `step_time × n_gpus × 989 TFLOP/s`.
* **pad efficiency**: non-padding tokens / all positions fed through the model.
* **MFU (real tokens)**: MFU (recipe) × pad efficiency: approximately the FLOPs that did useful work.
