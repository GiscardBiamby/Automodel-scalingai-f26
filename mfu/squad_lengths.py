"""SQuAD sequence-length statistics for the Nemotron-Nano-V3 recipe, and the padding they imply.

Builds the training split exactly as the recipe does (``make_squad_dataset`` with the model's tokenizer and
chat template), then reports the token-length distribution and the expected fraction of useful tokens when
batching ``local_batch_size`` random samples padded to the longest (the baseline's ``default_collater``),
versus packing to a fixed length.

Usage: mfu/docker.sh -- python mfu/squad_lengths.py [--local-batch-size 8] [--pack 4096] [--save-lengths FILE]
"""

import argparse
import json
import random
import statistics

from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.squad import make_squad_dataset

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-batch-size", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--pack", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--save-lengths", default=None, help="write every example's token length (JSON list) here")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ds = make_squad_dataset(tok, dataset_name="rajpurkar/squad", split="train")
    lengths = [len(x["input_ids"]) for x in ds]
    labels = [sum(1 for t in x["labels"] if t != -100) for x in ds]
    lengths_sorted = sorted(lengths)
    if args.save_lengths:
        with open(args.save_lengths, "w") as f:
            json.dump(lengths, f, separators=(",", ":"))

    def q(p: float) -> int:
        return lengths_sorted[min(len(lengths_sorted) - 1, int(p * len(lengths_sorted)))]

    out = {
        "num_samples": len(lengths),
        "len_mean": statistics.fmean(lengths),
        "len_p50": q(0.5),
        "len_p90": q(0.9),
        "len_p99": q(0.99),
        "len_max": lengths_sorted[-1],
        "label_tokens_mean": statistics.fmean(labels),
        "label_fraction": sum(labels) / sum(lengths),
        "pad_to_longest": {},
        "packing_fill": {},
    }
    rng = random.Random(0)
    for b in args.local_batch_size:
        eff = []
        for _ in range(args.trials):
            batch = rng.sample(lengths, b)
            eff.append(sum(batch) / (b * max(batch)))
        out["pad_to_longest"][b] = statistics.fmean(eff)
    # Greedy sequential packing (what a packed dataloader approximately achieves) -> fill ratio.
    for size in args.pack:
        shuffled = lengths[:]
        rng.shuffle(shuffled)
        packs, cur = [], 0
        for n in shuffled:
            n = min(n, size)
            if cur + n > size:
                packs.append(cur)
                cur = 0
            cur += n
        packs.append(cur)
        out["packing_fill"][size] = sum(packs) / (len(packs) * size)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
