# Updated model test statistics — integrated code at merge (2026-07-08)

2025 holdout (train 2021–2024 n=9,718, test n=2,430), LR+GB ensemble, weekly-retrain
hyperparams (300/3/0.05). xgboost unavailable locally, so the 50-model bootstrap leg is
not included — production adds it on Render.

| Config | AUC | LogLoss | Brier | Acc |
|---|---|---|---|---|
| A) leak-fixed, unweighted (Phase 4b reference) | 0.5762 | 0.6813 | 0.2442 | 0.5663 |
| B) leak-fixed + YEAR_WEIGHTS incl. **2021→0.3** (production retrain config) | 0.5752 | 0.6816 | 0.2443 | 0.5535 |
| Δ (B − A) | −0.0010 | +0.0002 | +0.0001 | −0.0128 |

Historical reference points (same holdout):
| Old leaked model (same-season SP) | 0.6083 | 0.6719 | 0.2396 | — |
| No SP/bullpen at all | 0.5569 | 0.6852 | 0.2461 | — |

## Reading
- **Config A reproduces Phase 4b exactly (0.5762)** — the integrated branch's pipeline is
  consistent; no regression from the merge, k_per_pa fix (live-baselines only), or other changes.
- **The 2021 down-weight is neutral on holdout discrimination** (ΔAUC −0.001 ≈ noise;
  the accuracy dip at the 0.5 threshold is within sampling error for n=2,430, SE ≈ 1%).
  This is expected: with the leak fixed, 2021's noisy league-average SP rows barely move a
  2025 evaluation. The weight primarily protects the FINAL production model (trained
  through 2026 with 2025/26 at 1.8), whose true test is live 2026 performance after the
  Render retrain — not this holdout.
- Honest expectation for the deployed retrain: ~0.575 AUC / ~0.68 LogLoss on holdout-like
  data, better live calibration than the old model's ECE 0.0786; anything near the old
  0.61 would indicate the leak fix didn't take (see RETRAIN_CHECKLIST §4).
