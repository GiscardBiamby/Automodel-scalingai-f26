#!/usr/bin/env python3
"""CPU-side view of an nsys capture: where host time goes, and which NVTX ranges have outlier calls.

Complements analyze_nsys.py (GPU kernels). Needs ``<run_dir>/profile.sqlite`` (written by analyze_nsys.py).
Stdlib only, runs on the host:
    python3 mfu/analyze_cpu.py mfu/runs/<run_dir> [--steps N]

Reports, per rank per captured step:
  * CUDA runtime API time by call (launches vs. synchronizing calls),
  * host (CPU) time inside the recipe's step-phase and per-module NVTX ranges,
  * outlier calls: NVTX ranges whose single-call CPU time is > --outlier-ms (e.g. cuDNN graph builds).
Writes ``<run_dir>/cpu_breakdown.md``.
"""

import argparse
import collections
import pathlib
import re
import sqlite3

RANGES = [
    "fwd_bwd_mb0",
    "fwd_bwd_mb1",
    "fwd_bwd_mb2",
    "fwd_bwd_mb3",
    "grad_clip",
    "optimizer_step",
    "mixer: NemotronV3Attention",
    "fused_attention: FusedAttention",
    "mixer: NemotronV3Mamba2Mixer",
    "mixer: MoE",
    "experts: GroupedExpertsDeepEP",
    "shared_experts: MLP",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--outlier-ms", type=float, default=20.0)
    args = ap.parse_args()

    steps = args.steps
    if steps is None:
        m = re.search(r"nsys_start_step=(\d+).*nsys_end_step=(\d+)", (args.run_dir / "command.txt").read_text())
        steps = int(m.group(2)) - int(m.group(1))
    con = sqlite3.connect(args.run_dir / "profile.sqlite")
    ranks = [r[0] for r in con.execute("select distinct (globalTid>>24)&0xffffff from CUPTI_ACTIVITY_KIND_RUNTIME")]
    per = len(ranks) * steps
    out = [f"# CPU-side breakdown ({len(ranks)} ranks x {steps} captured steps; values per rank per step)", ""]

    out += ["## CUDA runtime API", "", "| call | calls | ms |", "|---|---|---|"]
    q = """select s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r
           join StringIds s on r.nameId = s.id group by s.value order by 3 desc limit 12"""
    for name, n, t in con.execute(q):
        out.append(f"| {name} | {n / per:.0f} | {t / per / 1e6:.1f} |")

    def ranges(name: str) -> list[tuple[int, int]]:
        return con.execute(
            """select e.start, e.end from NVTX_EVENTS e left join StringIds s on e.textId = s.id
               where coalesce(e.text, s.value) = ? and e.end is not null""",
            (name,),
        ).fetchall()

    step_rows = con.execute(
        """select e.start, e.end from NVTX_EVENTS e left join StringIds s on e.textId = s.id
           where coalesce(e.text, s.value) like 'train_step_%' and e.end is not null"""
    ).fetchall()
    step_ms = sum(e - s for s, e in step_rows) / max(len(step_rows), 1) / 1e6
    out += ["", f"Mean train_step CPU range: {step_ms:.0f} ms", ""]
    out += [
        "## Host time inside NVTX ranges",
        "",
        f"| range | calls | CPU ms | median call ms | calls > {args.outlier_ms:.0f} ms | CPU ms in those |",
        "|---|---|---|---|---|---|",
    ]
    for name in RANGES:
        d = sorted((e - s) / 1e6 for s, e in ranges(name))
        if not d:
            continue
        slow = [x for x in d if x > args.outlier_ms]
        out.append(
            f"| {name} | {len(d) / per:.1f} | {sum(d) / per:.1f} | {d[len(d) // 2]:.2f} | "
            f"{len(slow) / per:.1f} | {sum(slow) / per:.1f} |"
        )

    # Largest single calls across all ranges (identifies one-off stalls such as graph compilation).
    rows = con.execute(
        """select coalesce(e.text, s.value), (e.end - e.start) / 1e6 from NVTX_EVENTS e
           left join StringIds s on e.textId = s.id where e.end is not null
           and coalesce(e.text, s.value) not like 'train_step_%' and coalesce(e.text, s.value) not like 'fwd_bwd_mb%'
           and coalesce(e.text, s.value) not like 'NCCL%'"""
    ).fetchall()
    leaf = collections.Counter()
    for name, ms in rows:
        if ms > args.outlier_ms:
            leaf[name] += 1
    out += ["", f"## Ranges with calls > {args.outlier_ms:.0f} ms (count per rank per step)", ""]
    for name, n in leaf.most_common(12):
        out.append(f"- `{name}`: {n / per:.1f}")
    md = "\n".join(out) + "\n"
    (args.run_dir / "cpu_breakdown.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
