#!/usr/bin/env python3
"""Break down where GPU time goes in an nsys capture of an MFU-study run.

Run inside the container (needs ``nsys`` for the sqlite export):
    mfu/docker.sh -- python mfu/analyze_nsys.py mfu/runs/<run_dir> [--steps N]

Produces ``<run_dir>/nsys_breakdown.md`` and ``nsys_breakdown.json`` with, per GPU and averaged:
  * wall window, busy time (union of all kernels), idle time (GPU does nothing),
  * compute vs communication time as interval unions, and *exposed* communication
    (comm time during which no compute kernel runs = comm that is not hidden by overlap),
  * summed kernel time per category (GEMM, grouped/MoE GEMM, attention, Mamba, NCCL, DeepEP, ...),
  * top kernels by total time,
  * GPU time projected onto the recipe's NVTX ranges (fwd/bwd per microbatch, grad clip, optimizer)
    via ``nsys stats -r nvtx_gpu_proj_sum``.
"""

import argparse
import json
import pathlib
import re
import sqlite3
import subprocess
from collections import defaultdict

# Ordered: first match wins. Patterns are matched against the demangled kernel name (lowercased).
CATEGORIES = [
    ("comm:deepep", r"deep_ep|deepep|intranode::|internode::|hybrid_ep"),
    ("comm:nccl_allgather", r"nccl.*allgather"),
    ("comm:nccl_reducescatter", r"nccl.*reducescatter"),
    ("comm:nccl_allreduce", r"nccl.*allreduce"),
    ("comm:nccl_alltoall_sendrecv", r"nccl.*(sendrecv|alltoall|send|recv)"),
    ("comm:nccl_other", r"nccl"),
    ("attention", r"flash|fmha|fused_attn|cudnn.*(attn|sdpa)|attention"),
    ("mamba", r"mamba|_chunk_scan|_chunk_state|_state_passing|_bmm_chunk|chunk_cumsum|causal_conv1d|selective_scan|ssd_"),
    ("gemm:grouped", r"grouped|group_gemm|groupgemm|_grouped_mm|ptr_array"),
    ("gemm", r"gemm|cutlass|xmma|cublas|nvjet|wgmma|matmul|_mm_"),
    ("norm", r"rmsnorm|rms_norm|layer_norm|layernorm|norm_fwd|norm_bwd"),
    ("loss/softmax", r"cross_entropy|log_softmax|softmax|nll_loss"),
    ("optimizer", r"adam|multi_tensor_apply|fused_adam"),
    ("moe:routing/permute", r"topk|sort|permute|unpermute|moe_|histogram|index_put|scatter_add"),
    ("memcpy/cat/copy", r"copy|cat_|catarray|memcpy|memset|fill"),
    ("elementwise/reduce", r"elementwise|vectorized|reduce_kernel|unrolled|pointwise|triton_"),
]
COMPILED = [(name, re.compile(pat)) for name, pat in CATEGORIES]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, rx in COMPILED:
        if rx.search(low):
            return cat
    return "other"


def union_length(intervals: list[tuple[int, int]]) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def intersect_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Length of the intersection of two *merged* interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def export_sqlite(rep: pathlib.Path) -> pathlib.Path:
    db = rep.with_suffix(".sqlite")
    if not db.exists():
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "--output", str(db), str(rep)],
            check=True,
        )
    return db


def load_kernels(db: pathlib.Path) -> dict[int, list[tuple[int, int, str]]]:
    con = sqlite3.connect(db)
    rows = con.execute(
        """SELECT k.deviceId, k.start, k.end, s.value
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id"""
    ).fetchall()
    per_dev: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for dev, s, e, name in rows:
        per_dev[dev].append((s, e, name))
    con.close()
    return per_dev


def analyze_device(kernels: list[tuple[int, int, str]]) -> dict:
    window = max(e for _, e, _ in kernels) - min(s for s, _, _ in kernels)
    by_cat: dict[str, int] = defaultdict(int)
    by_kernel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    comm_iv, comp_iv, all_iv = [], [], []
    for s, e, name in kernels:
        cat = categorize(name)
        by_cat[cat] += e - s
        k = by_kernel[name]
        k[0] += e - s
        k[1] += 1
        all_iv.append((s, e))
        (comm_iv if cat.startswith("comm:") else comp_iv).append((s, e))
    comm_m, comp_m = merge(comm_iv), merge(comp_iv)
    comm_union = sum(e - s for s, e in comm_m)
    comp_union = sum(e - s for s, e in comp_m)
    overlap = intersect_length(comm_m, comp_m)
    busy = union_length(all_iv)
    return {
        "window_ms": window / 1e6,
        "busy_ms": busy / 1e6,
        "idle_ms": (window - busy) / 1e6,
        "compute_union_ms": comp_union / 1e6,
        "comm_union_ms": comm_union / 1e6,
        "comm_overlapped_ms": overlap / 1e6,
        "comm_exposed_ms": (comm_union - overlap) / 1e6,
        "category_kernel_ms": {c: v / 1e6 for c, v in sorted(by_cat.items(), key=lambda x: -x[1])},
        "top_kernels": [
            {"name": n[:160], "category": categorize(n), "total_ms": v[0] / 1e6, "count": v[1]}
            for n, v in sorted(by_kernel.items(), key=lambda x: -x[1][0])[:25]
        ],
    }


def nvtx_projection(rep: pathlib.Path, out_dir: pathlib.Path) -> str:
    """Run nsys's NVTX->GPU projection report and return it as CSV text (empty on failure)."""
    prefix = out_dir / "nsys_stats"
    res = subprocess.run(
        ["nsys", "stats", "--force-export", "false", "--report", "nvtx_gpu_proj_sum", "--format", "csv",
         "--output", str(prefix), str(rep)],
        capture_output=True,
        text=True,
    )
    csvs = sorted(out_dir.glob("nsys_stats*nvtx_gpu_proj_sum*.csv"))
    return csvs[0].read_text() if res.returncode == 0 and csvs else ""


def pct(x: float, total: float) -> str:
    return f"{100.0 * x / total:5.1f}%" if total else "-"


def render(per_dev: dict[int, dict], steps: int | None, nvtx_csv: str) -> str:
    devs = sorted(per_dev)
    avg = lambda key: sum(per_dev[d][key] for d in devs) / len(devs)  # noqa: E731
    window = avg("window_ms")
    norm = f" (per step, {steps} captured steps)" if steps else ""
    div = steps or 1
    lines = [f"# nsys breakdown{norm}", "", f"GPUs analysed: {devs}", ""]
    lines += ["## Time budget (mean over GPUs)", "", "| bucket | ms | % of window |", "|---|---|---|"]
    for label, key in [
        ("wall window", "window_ms"),
        ("GPU busy (any kernel)", "busy_ms"),
        ("GPU idle (no kernel)", "idle_ms"),
        ("compute kernels (union)", "compute_union_ms"),
        ("communication kernels (union)", "comm_union_ms"),
        ("  comm overlapped with compute", "comm_overlapped_ms"),
        ("  comm exposed (not overlapped)", "comm_exposed_ms"),
    ]:
        lines.append(f"| {label} | {avg(key) / div:.1f} | {pct(avg(key), window)} |")

    cats: dict[str, float] = defaultdict(float)
    for d in devs:
        for c, v in per_dev[d]["category_kernel_ms"].items():
            cats[c] += v / len(devs)
    lines += ["", "## Summed kernel time by category (mean over GPUs; overlapping streams can sum >100%)", "",
              "| category | ms | % of window |", "|---|---|---|"]
    for c, v in sorted(cats.items(), key=lambda x: -x[1]):
        lines.append(f"| {c} | {v / div:.1f} | {pct(v, window)} |")

    lines += ["", f"## Top kernels on GPU {devs[0]}", "", "| kernel | category | ms | calls |", "|---|---|---|---|"]
    for k in per_dev[devs[0]]["top_kernels"]:
        lines.append(f"| `{k['name'][:90]}` | {k['category']} | {k['total_ms'] / div:.1f} | {k['count']} |")

    lines += ["", "## Per-GPU spread", "", "| GPU | window ms | busy % | idle % | comm exposed % |", "|---|---|---|---|---|"]
    for d in devs:
        r = per_dev[d]
        lines.append(
            f"| {d} | {r['window_ms'] / div:.1f} | {pct(r['busy_ms'], r['window_ms'])} | "
            f"{pct(r['idle_ms'], r['window_ms'])} | {pct(r['comm_exposed_ms'], r['window_ms'])} |"
        )
    if nvtx_csv:
        lines += ["", "## NVTX ranges projected onto the GPU (nsys nvtx_gpu_proj_sum, all ranks)", "", "```",
                  nvtx_csv.strip()[:6000], "```"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=None, help="captured optimizer steps, to report per-step numbers")
    args = ap.parse_args()

    reps = sorted(args.run_dir.glob("*.nsys-rep"))
    if not reps:
        raise SystemExit(f"no .nsys-rep in {args.run_dir}")
    rep = reps[0]
    steps = args.steps
    if steps is None:
        m = re.search(r"nsys_start_step=(\d+).*nsys_end_step=(\d+)", (args.run_dir / "command.txt").read_text())
        steps = int(m.group(2)) - int(m.group(1)) if m else None

    per_dev = {d: analyze_device(k) for d, k in load_kernels(export_sqlite(rep)).items() if k}
    nvtx_csv = nvtx_projection(rep, args.run_dir)
    md = render(per_dev, steps, nvtx_csv)
    (args.run_dir / "nsys_breakdown.md").write_text(md)
    (args.run_dir / "nsys_breakdown.json").write_text(json.dumps({"steps": steps, "per_gpu": per_dev}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
