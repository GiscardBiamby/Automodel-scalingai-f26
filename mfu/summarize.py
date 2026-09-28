#!/usr/bin/env python3
"""Summarize steady-state efficiency metrics of one or more MFU-study runs.

Reads ``<run_dir>/training.jsonl`` (written by the train_ft recipe), drops warmup steps and reports
step time, throughput, MFU and padding efficiency. Stdlib only, so it runs on the host.

Usage:
    python3 mfu/summarize.py RUN_DIR [RUN_DIR ...] [--warmup N] [--json]

Metric definitions:
    step_time_s        wall time between consecutive optimizer steps (includes data loading, logging)
    tokens/s/GPU       non-tail-padding tokens per second per GPU (recipe's ``tps_per_gpu``)
    MFU (recipe)       recipe-reported MFU in %: analytic model FLOPs of the *padded* input_ids / peak
    pad efficiency     non-padding tokens / all positions fed through the model
    useful MFU         real (non-padding) tokens/s/GPU * --flops-per-token / --peak-tflops. Default 19.36 GFLOP/token
                       = Nemotron-Nano-V3 fwd+bwd at SQuAD lengths (6 x 3.227B active matmul params + attention
                       scores at ~200 tokens; matches flops_utils.nemotronh_flops within 0.1%). Unlike the recipe's
                       MFU it excludes padding and the attention term of packed rows treated as one sequence.
"""

import argparse
import json
import pathlib
import statistics


def load_steps(run_dir: pathlib.Path) -> list[dict]:
    rows = []
    with open(run_dir / "training.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(run_dir: pathlib.Path, warmup: int, flops_per_token: float, peak_tflops: float) -> dict:
    rows = load_steps(run_dir)
    steady = rows[warmup:] if len(rows) > warmup else rows

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in steady if r.get(key) is not None]

    def stats(key: str) -> dict | None:
        v = col(key)
        if not v:
            return None
        return {"mean": statistics.fmean(v), "median": statistics.median(v), "min": min(v), "max": max(v)}

    out = {
        "run": run_dir.name,
        "steps_total": len(rows),
        "steps_used": len(steady),
        "warmup_dropped": len(rows) - len(steady),
        "step_time_s": stats("step_time"),
        "tps_per_gpu": stats("tps_per_gpu"),
        "tps": stats("tps"),
        "mfu_pct": stats("mfu"),
        "mem_gib": stats("mem"),
        "loss_first": rows[0].get("loss") if rows else None,
        "loss_last": rows[-1].get("loss") if rows else None,
    }
    real = col("num_tokens_per_step")
    positions = col("num_input_positions_per_step")
    if real and positions and len(real) == len(positions):
        out["pad_efficiency"] = sum(real) / sum(positions)
        out["tokens_per_step"] = statistics.fmean(real)
        out["positions_per_step"] = statistics.fmean(positions)
    tps = stats("tps_per_gpu")
    if tps:
        out["useful_mfu_pct"] = {k: 100 * v * flops_per_token / (peak_tflops * 1e12) for k, v in tps.items()}
    return out


def fmt(x, spec: str = ".2f") -> str:
    if x is None:
        return "-"
    if isinstance(x, dict):
        return f"{x['mean']:{spec}} (med {x['median']:{spec}})"
    return f"{x:{spec}}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--warmup", type=int, default=5, help="leading steps to drop (default 5)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a markdown table")
    ap.add_argument("--flops-per-token", type=float, default=19.36e9, help="model fwd+bwd FLOPs per real token")
    ap.add_argument("--peak-tflops", type=float, default=989.0, help="per-GPU dense BF16 peak (H100 SXM)")
    args = ap.parse_args()

    results = [summarize(d, args.warmup, args.flops_per_token, args.peak_tflops) for d in args.run_dirs]
    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("| run | steps | step time (s) | tokens/s/GPU | MFU % (recipe) | pad eff. | useful MFU % | peak mem (GiB) | loss first→last |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['run']} | {r['steps_used']}/{r['steps_total']} | {fmt(r['step_time_s'], '.3f')} "
            f"| {fmt(r['tps_per_gpu'], '.0f')} | {fmt(r['mfu_pct'])} | {fmt(r.get('pad_efficiency'), '.3f')} "
            f"| {fmt(r.get('useful_mfu_pct'))} | {fmt(r['mem_gib']['max'] if r['mem_gib'] else None, '.1f')} "
            f"| {fmt(r['loss_first'], '.3f')}→{fmt(r['loss_last'], '.3f')} |"
        )


if __name__ == "__main__":
    main()
