# Merge Readiness — integrate/full-fix → main

(New file, created Phase F1 2026-07-07 — no prior MERGE_READINESS.md existed.)
Branch: `integrate/full-fix`, **22 commits ahead of origin/main, nothing pushed.**

## Go / No-Go checklist

| Gate | Status | Evidence |
|---|---|---|
| Branch reconciliation (A1) | ✅ GO | Zero conflicts (ancestry-proven fast-forward, merge `55710ce`); both branches' logic verified intact |
| Immutability path audit (A2) | ✅ GO | All 3 creation paths correctly flagged/unflagged; 0 pick-rewrite sites — `scripts/results/immutability_path_audit.md` |
| Backup/restore clobber safety (B1) | ✅ GO | `_should_restore` hardened (pkl never overwritten while present; date-aware JSON guard); 7/7 dry-run matrix — `scripts/results/backup_restore_safety_check.md` |
| Calibration contamination (C1) | ✅ RESOLVED | **0/1,212 contaminated**; over-confidence finding (ECE 0.0786) holds on clean data; earlier "inconsistent entries" claim withdrawn — `scripts/results/calibration_contamination_check.md` |
| Quick fixes (D1–D3) | ✅ GO | `/api/refresh` TRIGGER_SECRET-gated + button removed; past-date cards served from log (`from_log`); pandas==3.0.0 + xgboost==3.3.0 pinned |
| IMPROVEMENTS sweep (G1–G8) | ✅ GO | Fallback features, rest_days alignment, predict.py path, favicon+og, model_version+RETRO badge, 5-test unit suite; calibration chart pre-existed; statuses in IMPROVEMENTS.md (deferred: base template, hamburger, market blend, recalibration, career-SP, bullpen-rolling, handedness; rejected: betting-page unrated line) |
| 2021 down-weight (E1) | ✅ GO | YEAR_WEIGHTS 2021: 1.0→0.3, verified identical in both training paths |
| Smoke test | ✅ PASS | `[smoke] OK` (LR+GB path; xgb stubbed locally) |
| Unit tests | ✅ PASS | 5/5 (`tests/test_units.py`) |
| All prod modules compile | ✅ PASS | app, MLBModel, predict, update_daily, schedule_fetcher, fetch_2026_games, init_betting_log |
| **Retrain (E2/E3)** | 🕓 **PENDING — post-merge** | Local impossible (no libomp/brew/docker; 2 attempts logged). Runs on Render after deploy: `/api/retrain-model?key=…` per `scripts/RETRAIN_CHECKLIST.md`. Until then the live pkl keeps the old (leaked, boosted, version-less) model — **unchanged from today**, so merging does not degrade anything. |

## Overall: **GO** — one pending item (retrain) is deliberately post-merge and cannot be done pre-merge.

## Deploy-day watchlist (first 24–48 h after merge)
1. Render logs at boot: `[github] ... backed up to GitHub` lines appear (first working
   backups ever); repo receives `Auto-backup predictions log YYYY-MM-DD` commits.
2. `[startup] betting_log upserted from ~1222 log + N odds entries`; `/api/debug/odds`
   → `betting_log_entries` ≥ 1341.
3. `[github] Skipping restore of ... — <reason>` lines look sane (no unexpected restores).
4. Site check: favicon in tabs; navbars consistent; no "Refresh Baselines" button;
   past-date cards show logged picks; RETRO badge only on post-hoc entries.
5. Trigger the retrain (`/api/retrain-model?key=…`), then `/api/model/info` shows a
   12-char `model_version` and new predictions carry it; holdout metrics ≈ 0.576 AUC
   (per checklist §4), NOT ~0.61.
6. Rotate `ODDS_API_KEY` + `TRIGGER_SECRET` when convenient (both appeared in chat).
