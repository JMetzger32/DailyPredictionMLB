# Code Review — Findings & Improvement Ideas (STATUS UPDATED)

Original review on `chore/review-cleanup`; statuses updated on `integrate/full-fix` after
the A–G verification/fix phases. ✅ = fixed & committed, 🕓 = deferred (reason given),
❌ = original claim found incorrect.

## Fixed in the original review (chore/review-cleanup)
| Fix | Impact |
|---|---|
| ✅ `import requests` missing in app.py | **Critical** — GitHub backup/restore never worked (0 auto-backup commits in history). Now functional. |
| ✅ `k_per_pa` absent from live team baselines | Trained feature was always 0 in production; now flows from the DB pipeline. |
| ✅ Live-resolve rewrote stored entries | Stored picks now scored against their own logged pick/probability. |
| ✅ Past-date backfills unflagged | Now carry `post_game_created: true`. |
| ✅ `/api/model/info` stale claims | 1.4× boost text removed; real hyperparams; dynamic size; `model_version` exposed. |
| ✅ Inconsistent navbars | All six pages link to all sections. |
| ✅ Minor hygiene | duplicate startup print, `utcnow()`, double `_pl_for_bet`. |

## Deferred risks from the review — status after Phases A–G
1. ✅ **`POST /api/refresh` unauthenticated** → gated behind TRIGGER_SECRET (`?key=` or JSON
   body); public "Refresh Baselines" button + handler removed (D1).
2. ✅ **MLB-API team fallback missing model features** → now supplies `k_per_pa` (batter
   K/AB), `opp_whip` ((BB+H)/IP), `opp_hr_per9` (HR/IP×9), `roll7_bullpen_fatigue`
   (prior-carried); obsolete `roll7_bullpen_ip` key removed (G1).
3. ✅ **Past-date cards recomputed live** → prediction fields now overlaid from the stored
   log entry (`from_log: true`); display extras stay fresh (D2).
4. ✅ **rest_days off-by-one train/live** → live now `clip(days_since_last_game, 1, 7)`,
   identical to training (G2).
5. ❌ **WITHDRAWN — original claim was wrong.** "Entries with predicted_winner inconsistent
   with home_win_prob" do not exist: 0/1,212 (see
   `scripts/results/calibration_contamination_check.md`). The sub-50% bins in the earlier
   live table were a binning-methodology artifact (raw home-prob bins), not corruption.
6. ✅ `Main/predict.py` cwd-relative pkl → `__file__`-relative (G3).
7. 🕓 `%-I` strftime portability — Windows-only concern; Render/macOS unaffected.
8. 🕓 Shared Jinja base template — prevents future navbar drift, but a 6-template refactor
   carries visual-regression risk without browser testing; navbars are currently unified.

## Accuracy improvements — status
1. 🕓→**READY** Regenerate the model artifact — all prerequisites now on `integrate/full-fix`
   (leak fix + k_per_pa + 2021 down-weight + no 1.4×). Needs an xgboost-capable env; see
   `scripts/RETRAIN_CHECKLIST.md` / `scripts/RETRAIN_EXECUTION_LOG.md`. Deploy-then-
   `/api/retrain-model` is the practical path.
2. 🕓 Live recalibration layer (live ECE ≈ 0.079, 60-70% picks win ~54%) — needs eval design;
   revisit after the leak-fixed retrain has some live history.
3. 🕓 Market-probability blend + CLV as primary KPI — needs eval; changes bet_rating semantics.
4. ✅ Down-weight 2021 (YEAR_WEIGHTS 1.0 → 0.3 in both training paths) — its SP features are
   100% league-average post-leak-fix (E1).
5. 🕓 Career-to-date SP prior (recovers signal for ~35% fallback rows) — retrain-cycle change.
6. 🕓 Rolling bullpen ERA (last 30 days) — retrain-cycle change.
7. 🕓 Pitcher handedness feature — requires historical handedness backfill first.

## Design / UX — status
1. 🕓 Shared Jinja base template (see #8 above).
2. 🕓 Mobile hamburger nav — needs browser testing; navbar wraps acceptably today.
3. ✅ Favicon (inline-SVG ⚾) on all six pages + description/og meta on home (G4).
4. ✅ Card transparency — `model_version` in /api/predictions; RETRO badge (with tooltip) on
   post-game-created cards (G6).
5. ✅ Calibration chart — **already existed** on the accuracy page consuming /api/calibration
   (verified G5; no change needed — original review missed it).
6. 🚫 Betting-page unrated-history line — **rejected by user**: overall history is the Season
   Record page's job; betting page stays odds-rated-only.
7. ✅ og: meta tags on home (G4). Hero-image preload not done (marginal).
8. ✅ "Refresh Baselines" button removed (D1).

## Ops — status
1. ✅ Backups now flow (requests import) + restore hardened against stale-backup clobber
   (`_should_restore`: pkl never overwritten while present; date-aware JSON guard — B1,
   `scripts/results/backup_restore_safety_check.md`). Watch Render logs for
   `[github] ... backed up`.
2. 🕓 Rotate secrets (`ODDS_API_KEY`, `TRIGGER_SECRET` were pasted in chat) — **user action**.
3. ✅ Pinned `pandas==3.0.0`, `xgboost==3.3.0` (D3; sklearn was already pinned).
4. ✅ Test suite started: `tests/test_units.py` (5 tests: calibration bucket parity, odds
   fields, pitcher matching, restore decisions, real upsert SQL COALESCE semantics) +
   `scripts/smoke_test.py`. Run: `.venv/bin/python tests/test_units.py`.
