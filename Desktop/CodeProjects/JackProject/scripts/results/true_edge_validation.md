# True edge validation — elastic net vs L2

_Generated 2026-08-17T11:22:07 by `scripts/validate_true_edge.py` (report-only)._

## Method

The local DB's 2026 games stop at 2026-07-07 while every odds-carrying row runs 2026-07-16 onward, so features for priced games could not be rebuilt. Instead the raw features are **recovered by inverting `x_scaled_features`** from `predictions_log.json` (`raw = scaled * scaler.scale_ + scaler.mean_`), using each row's own model_version scaler — the historical one recovered from git (`{'b9133b95d2ec': '83fda83'}`). No DB needed.

Reconstruction gate: replaying the shipped model through this path reproduces the logged probability to **max abs error 0.00049** on 49 current-version rows (the residual is the log's `round(prob, 3)`). The script aborts rather than report if this fails.

Both candidate LRs are trained on **leak-fixed** data; GB and the 50-model bootstrap XGB are held fixed from the shipped pkl, isolating the penalty change. Edges use the de-vigged stored prices and `_rate_edge`'s live thresholds (good 0.05-0.12, extreme >0.12).

n = **362** odds rows (313 under b9133b95d2ec, 49 under 2c50e24e590d).

## Results

| metric | L2 (C=0.5) | elastic net | change |
|---|---|---|---|
| value bets (good) | 120.0000 | 117.0000 | -3.0000 |
| extreme (>0.12) | 44.0000 | 43.0000 | -1.0000 |
| bad (<-0.05) | 47.0000 | 45.0000 | -2.0000 |
| unsure | 151.0000 | 157.0000 | 6.0000 |
| mean edge (all) | 0.0420 | 0.0404 | -0.0016 |
| mean edge (value) | 0.0805 | 0.0796 | -0.0008 |
| resolved value bets | 115.0000 | 113.0000 | -2.0000 |
| value-bet win% | 0.5652 | 0.5752 | 0.0100 |
| flat-bet ROI | 0.0919 | 0.1093 | 0.0174 |

- games entering the value-bet band under elastic net: **4**
- games leaving it: **7**

## Reading this honestly

The resolved value-bet counts here are small (n=115 vs 113), far below the ~400 bets/bucket CLAUDE.md notes are needed to resolve a 10pp win-rate gap. **Win% and ROI differences at this n are not evidence** — they are reported because they are the quantities that matter operationally, not because they discriminate between the two models.

The load-bearing numbers are the **counts**: whether the change moves games across the 0.05 threshold in bulk. That is a mechanical property of the probability shift and is measurable at this n.
