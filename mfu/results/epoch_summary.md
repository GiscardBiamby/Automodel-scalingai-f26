# Full-epoch end-to-end runs (8x H100, SQuAD train, validation every 25 steps on 1,000 examples)

| run | steps | real tokens | wall time (first to last step) | median tok/s/GPU | epoch-average tok/s/GPU | useful MFU (median) | peak mem | train loss first → last | val loss first → last |
|---|---|---|---|---|---|---|---|---|---|
| final 8-GPU config (`20260929-052305_epoch_final8`) | 189 | 18.02 M | 3.1 min | 15,350 | 12,085 | 30.0% | 70.6 GiB | 5.151 → 0.053 | 0.753 → 0.089 |
| baseline recipe (`20260929-053034_epoch_baseline`) | 343 | 18.03 M | 26.7 min | 1,411 | 1,405 | 2.8% | 60.0 GiB | 5.245 → 0.040 | 1.929 → 0.089 |

Figures: `results/figures/epoch_loss.png`, `results/figures/epoch_progress.png`.
