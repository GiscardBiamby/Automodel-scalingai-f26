"""End-to-end evidence for full-epoch runs: loss curves, progress over wall time, and a summary table.

Usage (host): uv run --no-project --with matplotlib python mfu/epoch_evidence.py RUN_DIR [RUN_DIR ...]
Reads <run>/training.jsonl and <run>/validation.jsonl; writes mfu/results/figures/epoch_loss.png,
mfu/results/figures/epoch_progress.png and mfu/results/epoch_summary.md.
"""

import datetime as dt
import json
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plots import GRID, INK, INK2, OUT, SERIES, SURFACE  # noqa: E402,F401  (shared style + rcParams)

FLOPS_PER_TOKEN, PEAK = 19.36e9, 989e12


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()] if path.exists() else []


def ts(row: dict) -> float:
    return dt.datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp()


def label_for(run: pathlib.Path) -> str:
    return "final 8-GPU config" if "final" in run.name else ("baseline recipe" if "baseline" in run.name else run.name)


def main(runs: list[pathlib.Path]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_l, ax_l = plt.subplots(figsize=(8, 3.6))
    fig_p, ax_p = plt.subplots(figsize=(8, 3.4))
    rows_md = []
    for color, run in zip(SERIES, runs):
        tr, va = load(run / "training.jsonl"), load(run / "validation.jsonl")
        if not tr:
            continue
        label = label_for(run)
        cum, seen = [], 0
        for r in tr:
            seen += r["num_tokens_per_step"]
            cum.append(seen)
        by_step = dict(zip((r["step"] for r in tr), cum))
        ax_l.plot([c / 1e6 for c in cum], [r["loss"] for r in tr], color=color, linewidth=1.5, label=f"{label}: train")
        if va:
            ax_l.plot(
                [by_step.get(v["step"], 0) / 1e6 for v in va],
                [v["val_loss"] for v in va],
                color=color,
                linewidth=0,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=f"{label}: validation",
            )
        t0 = ts(tr[0]) - tr[0]["step_time"]
        ax_p.plot(
            [0] + [(ts(r) - t0) / 60 for r in tr], [0] + [c / 1e6 for c in cum], color=color, linewidth=2, label=label
        )
        wall = ts(tr[-1]) - t0
        steady = tr[5:] or tr
        tps = statistics.median(r["tps_per_gpu"] for r in steady)
        tps_mean = cum[-1] / wall / 8  # whole epoch incl. warm-up and validation passes
        va_str = f"{va[0]['val_loss']:.3f} → {va[-1]['val_loss']:.3f}" if va else "-"
        rows_md.append(
            f"| {label} (`{run.name}`) | {len(tr)} | {cum[-1] / 1e6:.2f} M | {wall / 60:.1f} min | {tps:,.0f} | {tps_mean:,.0f} | "
            f"{100 * tps * FLOPS_PER_TOKEN / PEAK:.1f}% | {max(r['mem'] for r in tr):.1f} GiB | "
            f"{tr[0]['loss']:.3f} → {tr[-1]['loss']:.3f} | "
            f"{va_str} |"
        )
    ax_l.set_yscale("log")
    ax_l.set_xlabel("real (non-padding) training tokens seen (millions)")
    ax_l.set_ylabel("loss (log scale)")
    ax_l.legend(fontsize=8, ncol=2)
    ax_l.set_title("One SQuAD epoch: training and validation loss vs tokens seen", loc="left", fontsize=10)
    fig_l.tight_layout()
    fig_l.savefig(OUT / "epoch_loss.png", dpi=180)
    ax_p.set_xlabel("wall-clock time since first step (minutes)")
    ax_p.set_ylabel("real tokens processed (millions)")
    ax_p.legend(fontsize=8)
    ax_p.set_title("One SQuAD epoch on 8x H100: progress over time", loc="left", fontsize=10)
    fig_p.tight_layout()
    fig_p.savefig(OUT / "epoch_progress.png", dpi=180)
    md = [
        "# Full-epoch end-to-end runs (8x H100, SQuAD train, validation every 25 steps on 1,000 examples)",
        "",
        "| run | steps | real tokens | wall time (first to last step) | median tok/s/GPU | epoch-average tok/s/GPU | useful MFU (median) | peak mem |"
        " train loss first → last | val loss first → last |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *rows_md,
        "",
        "Figures: `results/figures/epoch_loss.png`, `results/figures/epoch_progress.png`.",
    ]
    pathlib.Path("mfu/results/epoch_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main([pathlib.Path(p) for p in sys.argv[1:]])
