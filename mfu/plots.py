"""Report figures for the Nemotron-Nano-V3 SQuAD MFU study.

Usage (host, no install):  uv run --no-project --with matplotlib python mfu/plots.py
Reads mfu/runs/* (training.jsonl, nsys_breakdown.json, profile.sqlite) and writes PNGs to mfu/results/figures/.
"""

import glob
import json
import pathlib
import sqlite3
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("mfu/results/figures")
# Reference categorical palette (light mode), fixed order; text uses ink tokens, never series colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FLOPS_PER_TOKEN, PEAK = 19.36e9, 989e12

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)

RUNS = [  # (label, run-dir glob suffix)
    ("Baseline\n(pad-to-longest)", "_baseline"),
    ("A: pad to\nmultiple of 64", "_pad64"),
    ("B: THD packing\n4096", "_pack4096"),
    ("B + fused\nlinear-CE", "_pack4096_flce"),
    ("B + fused CE\n+ 2 packs/GPU", "_pack4096_flce_lbs2"),
]


def run_dir(suffix: str) -> pathlib.Path | None:
    hits = sorted(p for p in glob.glob(f"mfu/runs/*{suffix}") if p.endswith(suffix))
    return pathlib.Path(hits[-1]) if hits else None


def steps(d: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in open(d / "training.jsonl")]


def fig_mfu() -> None:
    labels, useful, tps = [], [], []
    for label, suf in RUNS:
        d = run_dir(suf)
        if d is None:
            continue
        rows = steps(d)[5:]
        med = statistics.median(r["tps_per_gpu"] for r in rows)
        labels.append(label)
        tps.append(med)
        useful.append(100 * med * FLOPS_PER_TOKEN / PEAK)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(labels, useful, color=SERIES[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, u, t in zip(bars, useful, tps):
        ax.text(b.get_x() + b.get_width() / 2, u + 0.6, f"{u:.1f}%\n{t:,.0f} tok/s/GPU", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ref = run_dir("_oss_ref_benchmark")
    if ref is not None:
        r = statistics.median(x["tps_per_gpu"] for x in steps(ref)[5:])
        ref_mfu = 100 * r * 19.72e9 / PEAK
        ax.axhline(ref_mfu, color=INK2, linestyle="--", linewidth=1.2)
        ax.text(len(labels) - 0.5, ref_mfu + 0.6, f"upstream benchmark recipe (synthetic 4k seqs, balanced routing): {ref_mfu:.1f}%",
                ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_ylabel("useful MFU (%)")
    ax.set_ylim(0, max(useful + [35]) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Useful MFU by configuration (median of steps 5-39; real tokens x 19.36 GFLOP / 989 TFLOP/s)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "mfu_by_config.png", dpi=180)
    plt.close(fig)


def fig_budget() -> None:
    cols = [
        ("compute only", SERIES[0]),
        ("comm overlapped w/ compute", SERIES[2]),
        ("comm exposed", SERIES[1]),
        ("GPU idle", "#b9b8b2"),
    ]
    data = []
    for label, suf in [("Baseline", "_baseline_nsys"), ("THD packing 4096", "_pack4096_nsys")]:
        d = run_dir(suf)
        if d is None or not (d / "nsys_breakdown.json").exists():
            continue
        j = json.loads((d / "nsys_breakdown.json").read_text())
        g = j["per_gpu"].values()
        n, s = len(g), j["steps"]
        m = lambda k: sum(x[k] for x in g) / n / s  # noqa: E731
        comp_only = m("compute_union_ms") - m("comm_overlapped_ms")
        vals = [comp_only, m("comm_overlapped_ms"), m("comm_exposed_ms"), m("idle_ms")]
        data.append((label, [100 * v / m("window_ms") for v in vals], m("window_ms")))
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for i, (label, vals, win) in enumerate(data):
        left = 0.0
        for (name, color), v in zip(cols, vals):
            ax.barh(i, v, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2,
                    label=name if i == 0 else None)
            if v > 5:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
            left += v
    ax.set_yticks(range(len(data)), [f"{d[0]}\n({d[2]:,.0f} ms/step)" for d in data])
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("% of step wall time (mean over 8 GPUs, from nsys)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8.5)
    ax.set_title("Where each GPU's step time goes", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "gpu_time_budget.png", dpi=180)
    plt.close(fig)


def fig_step_times() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.0))
    for color, (label, suf) in zip(SERIES, RUNS):
        d = run_dir(suf)
        if d is None:
            continue
        rows = steps(d)[1:]
        x = [r["step"] for r in rows]
        y = [r["step_time"] / r["num_tokens_per_step"] * 1e6 for r in rows]  # us per real token (all GPUs)
        ax.plot(x, y, color=color, linewidth=2, label=label.replace("\n", " "))
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("step time per 1k real tokens (ms, log)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Step time per 1k real tokens; spikes are one-off host stalls", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "step_times.png", dpi=180)
    plt.close(fig)


LANES = [  # (lane label, predicate on category, color)
    ("NCCL (FSDP AG/RS)", lambda c: c.startswith("comm:nccl"), SERIES[1]),
    ("DeepEP dispatch/combine", lambda c: c == "comm:deepep", SERIES[3]),
    ("GEMM (dense + grouped)", lambda c: c.startswith("gemm"), SERIES[0]),
    ("Mamba / attention", lambda c: c in ("mamba", "attention"), SERIES[2]),
    ("other compute", lambda c: not (c.startswith("comm") or c.startswith("gemm") or c in ("mamba", "attention")),
     SERIES[6]),
]


def fig_timeline(suffix: str, title: str, fname: str) -> None:
    import sys

    sys.path.insert(0, "mfu")
    from analyze_nsys import categorize

    d = run_dir(suffix)
    if d is None or not (d / "profile.sqlite").exists():
        return
    con = sqlite3.connect(d / "profile.sqlite")
    dev = min(r[0] for r in con.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL"))
    # First complete train_step on the rank that owns this device (rank 0 <-> lowest pid).
    pid = con.execute("select min((globalTid>>24)&0xffffff) from NVTX_EVENTS").fetchone()[0]
    st = con.execute(
        """select e.start, e.end from NVTX_EVENTS e left join StringIds s on e.textId=s.id
           where coalesce(e.text,s.value) like 'train_step_%' and ((e.globalTid>>24)&0xffffff)=? and e.end is not null
           order by e.start limit 1 offset 1""", (pid,)).fetchone()
    t0, t1 = st
    ks = con.execute(
        """select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
           where k.deviceId=? and k.end>? and k.start<?""", (dev, t0, t1)).fetchall()
    cpu = con.execute(
        """select coalesce(e.text,s.value), e.start, e.end from NVTX_EVENTS e left join StringIds s on e.textId=s.id
           where ((e.globalTid>>24)&0xffffff)=? and e.start>=? and e.end<=? and e.end is not null
           and (coalesce(e.text,s.value) like 'fwd_bwd_mb%' or coalesce(e.text,s.value) in
                ('optimizer_step','grad_clip','fused_attention: FusedAttention'))""", (pid, t0, t1)).fetchall()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (lane, pred, color) in enumerate(LANES):
        segs = [((s - t0) / 1e6, (e - s) / 1e6) for s, e, n in ks if pred(categorize(n))]
        ax.broken_barh(segs, (i - 0.35, 0.7), facecolors=color, linewidth=0)
    for name, s, e in cpu:
        slow = name.startswith("fused_attention")
        if slow and (e - s) < 20e6:
            continue
        y = len(LANES) + (1 if slow else 0)
        color = SERIES[7] if slow else "#8f8e88"
        ax.broken_barh([((s - t0) / 1e6, (e - s) / 1e6)], (y - 0.35, 0.7), facecolors=color, edgecolor=SURFACE,
                       linewidth=1)
        if not slow and (e - s) / (t1 - t0) > 0.08:
            ax.text((s - t0) / 1e6 + (e - s) / 2e6, y, name.replace("fwd_bwd_mb", "fwd+bwd mb"), ha="center",
                    va="center", fontsize=7.5, color="white")
    ax.set_yticks(range(len(LANES) + 2),
                  [lane for lane, _, _ in LANES] + ["CPU: micro-batch", "CPU: attention call >20 ms"])
    ax.invert_yaxis()
    ax.set_xlim(0, (t1 - t0) / 1e6)
    ax.set_xlabel("ms since start of step (GPU 0 kernels; rank 0 host ranges)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=180)
    plt.close(fig)


RUNS_6GPU = [  # (label, run-dir suffix): each row adds one change to the previous one
    ("baseline\n(6 GPU)", "_baseline_6gpu"),
    ("packing +\nfused CE +\n2 packs + GC", "_best_6gpu"),
    ("+ fused\nAdam", "_best6_fusedadam"),
    ("+ 8 packs,\nfull AC", "_best6_lbs8_fullac"),
    ("+ bf16\ngrad RS", "_best6_lbs8_fullac_bf16rs"),
    ("+ TE\nRMSNorm", "_best6v2_rmsnorm_te"),
    ("+ DeepEP\n64 SMs", "_best6v3_deepep_sms64"),
    ("+ no AC on\nattention", "_best6v6_noac_attn"),
]


def fig_6gpu() -> None:
    labels, useful, tps = [], [], []
    for label, suf in RUNS_6GPU:
        d = run_dir(suf)
        if d is None:
            continue
        med = statistics.median(r["tps_per_gpu"] for r in steps(d)[5:])
        labels.append(label)
        tps.append(med)
        useful.append(100 * med * FLOPS_PER_TOKEN / PEAK)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(labels, useful, color=SERIES[0], width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, u, t in zip(bars, useful, tps):
        ax.text(b.get_x() + b.get_width() / 2, u + 0.5, f"{u:.1f}%\n{t / 1000:.1f}k", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_ylabel("useful MFU (%)")
    ax.set_ylim(0, max(useful) * 1.25)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("6 GPUs (EP=2): useful MFU as changes accumulate (labels: MFU, tokens/s/GPU)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "mfu_6gpu_progression.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_mfu()
    fig_budget()
    fig_step_times()
    fig_6gpu()
    fig_timeline("_baseline_nsys", "Baseline: one step on GPU 0 (4 micro-batches); red = TE/cuDNN attention graph builds",
                 "timeline_baseline.png")
    fig_timeline("_pack4096_nsys", "THD packing: one training step on GPU 0 (2 micro-batches)", "timeline_pack4096.png")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
