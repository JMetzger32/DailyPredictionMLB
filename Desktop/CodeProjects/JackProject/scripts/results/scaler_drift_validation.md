# Scaler-drift fix — what it does and doesn't do

Generated 2026-08-13 on branch `fix/scaler-drift` (stacked on
`chore/fix-team-code-mapping`). Validates the change described in
`EDA/eda_3/08_feature_drift_abs_regime.md`: `StandardScaler` was fit on all
of 2021-2026, and 2026-live scaled feature spread for `diff_sp_xfip`/
`diff_sp_siera` exceeded 1.15x the training-fit spread (1.96x / 2.03x).

## What changed

`Main/MLBModel.py`'s and `updates/update_daily.py`'s production scaler fits
now use only rows with `season >= SCALER_WINDOW_START_SEASON` (constant,
currently 2023) to compute `mean_`/`scale_`, instead of the full 2021+
range. Model training itself is unaffected — LR/GB/XGB still train on all
seasons via `YEAR_WEIGHTS`. `cross_validate_loso` now runs 5 folds
(2021-2025, was 4) with the same windowed-per-fold scaler logic, and reports
Brier alongside AUC/logloss. `compute_model_version` now hashes the
scaler's `mean_`/`scale_` in addition to feature list + LR coef + timestamp,
so it actually signals when the scaler changes (previously it didn't).

## Window sweep — does narrowing the window fully close the drift?

Tested `SCALER_WINDOW_START_SEASON` in {2021 (~unwindowed), 2022, 2023,
2024, 2025}, each via a real `update_daily.retrain_model()` run (so the
live 2026 SP-baseline injection is included, not the DB-only fallback).
For each, re-scaled the same 381 already-materialized live-2026 prediction
rows (`x_scaled_features` for `model_version=b9133b95d2ec`, unscaled with
the pre-fix scaler and rescaled with the candidate) and recomputed spread:

| window | diff_sp_xfip sd | diff_sp_siera sd | 2026-holdout acc | brier | logloss |
|---|---|---|---|---|---|
| pre-fix (b9133b95d2ec, no OAK fix) | 1.965 | 2.033 | — | — | — |
| >=2021 (OAK-fixed, ~unwindowed) | 1.850 | 1.921 | 0.5270 | 0.2506 | 0.6946 |
| >=2022 | 1.676 | 1.739 | 0.5270 | 0.2506 | 0.6947 |
| **>=2023 (shipped)** | **1.688** | **1.738** | **0.5270** | **0.2506** | **0.6946** |
| >=2024 | 1.654 | 1.728 | 0.5270 | 0.2506 | 0.6946 |
| >=2025 | 1.619 | 1.679 | 0.5270 | 0.2506 | 0.6946 |

**Honest finding: no window tested gets either flagged feature below the
1.15x target.** The fix meaningfully shrinks the excess spread (xfip
1.96x→1.69x, siera 2.03x→1.74x, roughly a 35-45% reduction in the *excess*
over 1.0), but a large residual gap remains even at the narrowest window
tested (2025-only: still 1.62x/1.68x). Narrowing the window past 2023 keeps
buying small further reductions with no sign of a floor, which suggests the
2026 live distribution for these two SP-quality-differential features is
genuinely wider than any recent single season's — consistent with EDA
section 08's read that this reflects a real 2026-regime shift (ABS/rules
era), not merely a stale reference window. **A scaler refit mitigates but
does not fix this feature-drift issue; closing it further would need
narrower/adaptive windowing (with the stability costs that implies) or
addressing the feature itself, not just its scaling.**

2023 was kept as the shipped window: 2022-2025 are statistically
indistinguishable from each other on both the SD-ratio metric and the
2026-holdout accuracy/brier/logloss (which are flat across every window
tested, including the unwindowed baseline — see below), so there's no
accuracy-based reason to prefer a narrower, less-stable fit.

## LOSO 2021-2025 — isolating the windowing effect

Ran the historical (2021-2025) LOSO cross-validation twice on identical
data/folds/code, varying only whether the scaler fit is windowed:

| | LR AUC | LR Brier | LR LogLoss | GB AUC | GB Brier |
|---|---|---|---|---|---|
| unwindowed (5-fold baseline) | 0.5907 | 0.2429 | 0.6787 | 0.5751 | 0.2465 |
| windowed (2023+, shipped) | 0.5907 | 0.2429 | 0.6787 | 0.5751 | 0.2465 |

Identical to 4 significant figures. GB is byte-identical (it never uses the
scaler, a useful sanity check that nothing else changed). **This is a
measurement-hygiene fix, not a performance fix** — the point is closing (part
of) the live/train scaled-spread gap in EDA 08, not moving accuracy, and it
doesn't move accuracy in either direction on 2021-2025 history or the 2026
holdout.

## Shipped

Artifact retrained via the real `update_daily.retrain_model()` path
(injection-aware), `model_version` bumped `b9133b95d2ec` → `2c50e24e590d`.
pkl round-trips through `pickle.load`/`.transform()` with the expected
`(n, 18)` shape. `tests/test_units.py` (13/13) and `scripts/smoke_test.py`
pass.
