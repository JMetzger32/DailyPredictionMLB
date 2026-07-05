# PR Summary — fix/eval-and-leak-phased

Phased fixes to the MLB prediction app: persistence, evaluation hygiene, training-path
reconciliation, and removal of look-ahead leakage on the model's dominant feature. Every
number below was measured in this branch; evidence files are committed under
`scripts/results/`, `scripts/PHASE4_RESULTS.md`, and the diagnostic scripts that produced them.

## Measured impact by phase

### Phase 1 — stop the bleeding
| Fix | Measured impact |
|---|---|
| 1a betting_log persistence decoupled from odds window (+ past-date odds backfill, mismatch logging) | Table was **0 rows** despite 1,222 logged predictions (0 with odds, 0 with bet_rating — the upsert gated on bet_rating). New gate verified on a DB copy: **0 → 1,211 rows** from the existing log. |
| 1b `prediction_timestamp` (write-once) + `post_game_created` flag | New entries are auditable; post-game backfills permanently excludable from evaluation. Not backfilled historically (by design). |
| 1c calibration bucketing unified | Old 6-band `_calibration_bucket` replaced by the canonical 10-even-bin scheme; asserted identical to `/api/calibration` binning. |
| 1d `model_version` content hash (12-char sha256 of features+LR coefs+timestamp) in both save paths + on every prediction | Deterministic & change-sensitive (verified); live pkl stamps on next retrain. |

### Phase 2 — baseline diagnostics
- **Live 2026 calibration (OLD model, 1,212 clean picks):** overall accuracy **53.5%**, badly
  over-confident — 60-70% bin won 53.8%, 70-80% bin won 58.1%.
- **Leak test (2025 holdout, train 2021–2024, LR+GB ensemble):** FULL features AUC **0.6083**
  vs SP/bullpen-dropped **0.5569** → drop **0.0514** > 0.01 gate → leak fix warranted.

### Phase 3 — reconcile training paths
| Fix | Measured impact |
|---|---|
| 3a removed post-hoc 1.4× SP-ERA coefficient boost from MLBModel.py | The boost existed only offline; the weekly retrain silently dropped it — paths now produce identical (unboosted) models. |
| 3b ensemble per-model loop | Previously one failing bootstrap model silently dropped **all 50** (bare `except: pass`). Now per-model catch + index logging + <40/50 warning. Verified partial (2/3), all-fail (fallback to LR/GB), and healthy (45/45, no warning) paths. **No real failure rate observed** — local xgboost is stubbed; live rate observable via new logs after deploy. |
| 3c `INSERT ... ON CONFLICT DO UPDATE` upsert | `created_at` set once & preserved; NULL writes no longer clobber resolved columns (verified: odds-less re-upsert kept `actual_winner/correct/clv/bet_rating`, advanced `updated_at`). |

### Phase 4 — leak fix (prior-season SP/bullpen features)
2025 holdout, LR+GB ensemble (train 2021–2024):
| Config | AUC | LogLoss | Brier |
|---|---|---|---|
| Leaked (same-season SP/bullpen) | 0.6083 | 0.6719 | 0.2396 |
| **Leak-fixed (prior-season)** | **0.5762** | **0.6813** | **0.2442** |
| No SP/bullpen at all | 0.5569 | 0.6852 | 0.2461 |

- **Leak size: 0.0321 AUC** (≈ 2/3 of the apparent SP signal was look-ahead).
- **Genuine pre-game SP signal retained: +0.0193 AUC** over a no-SP model — worth keeping.

### Follow-up verification (Steps 1–3 evidence)
- **Calibration before/after (ECE, 10-bin):**
  - 2025 holdout: leaked **0.0313** → leak-fixed **0.0266** (improved, −0.0047, same rows).
  - Live 2026 OLD model: **0.0795** — ~2.5–3× worse than any holdout; the leaked holdout
    flattered the model because its test rows carried the same leak.
  - Leak-fixed 2026 approximation: **0.0400** (caveat: pipeline prior-season SP inputs, not the
    live snapshot). → `scripts/results/calibration_comparison.md`
- **Prior-season substitution integrity:** **46.3%** of 2021–2025 rows fall back to
  league-average (2021 = 100%, no 2020 data; 2022–24 ≈ 35%; 2025 = 24%). `sp_era` std
  1.130 → 0.890 (−21%) — honest signal dilution from removing the leak. Future retrain should
  consider down-weighting 2021. → `scripts/results/prior_season_substitution_check.md`
- **Security scan:** pasted Odds API key found in **0 of 122 commits** (pickaxe + regex + tree
  grep), not in tree/index/plan file; nothing to purge. `.gitignore` secret patterns added
  (no hook, per decision). Recommend rotating the key (it exists in the chat transcript).
  → `scripts/results/security_scan.md`

## Not done here (deliberately)
- **The live `updates/mlb_model_artifacts.pkl` was NOT regenerated** — local env lacks working
  xgboost (no libomp), and retraining without it would silently drop the 50-model bootstrap
  ensemble. The deployed artifact still has leaked features + the baked-in 1.4× boost +
  `model_version=None` until retrained per `scripts/RETRAIN_CHECKLIST.md` (env prep, validation
  gates, old-vs-new slate diff, rollback).
- Nothing merged to `main`; nothing pushed; no deploys.

## Branch contents
18 commits: 4 Phase-1 fixes, 2 Phase-2 diagnostics, 3 Phase-3 fixes, 2 Phase-4 (fix + results),
1 smoke test, 5 follow-up verification/reporting commits, plus this summary.
Key evidence files: `scripts/results/calibration_comparison.md`,
`scripts/results/prior_season_substitution_check.md`, `scripts/results/security_scan.md`,
`scripts/PHASE4_RESULTS.md`, `scripts/RETRAIN_CHECKLIST.md`.
