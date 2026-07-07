# Code Review — Findings & Improvement Ideas

Full-codebase review (branch `chore/review-cleanup`). Everything **fixed** is committed on
this branch; everything below is **written down but deliberately not changed**, either
because it alters behavior in a way you should sign off on, or it's a future project.

## Fixed on this branch (summary)
| Fix | Impact |
|---|---|
| `import requests` missing in app.py | **Critical** — GitHub backup/restore of every log + artifacts silently never worked (0 auto-backup commits in repo history). Now functional. |
| `k_per_pa` absent from live team baselines | A trained model feature (`diff_roll30_k_per_pa`) was **always 0 in production**. Now flows from the DB pipeline (takes effect after next daily update). |
| Live-resolve rewrote stored entries with view-time predictions | Stored picks are now scored against their own logged pick/probability (prediction immutability). |
| Past-date backfills unflagged | `_log_predictions_for_date` backfills now carry `post_game_created: true` like the mid-request path. |
| `/api/model/info` stale claims | No longer advertises the removed 1.4× boost; real retrain hyperparams; dynamic training size; exposes `model_version`. |
| Inconsistent navbars | All six pages now link to all sections (index had no Betting/Picks links at all). |
| Minor: duplicate startup print, deprecated `utcnow()`, double `_pl_for_bet` call | Hygiene. |

---

## Bugs / risks found but NOT changed (need your sign-off)

1. **`POST /api/refresh` is unauthenticated** ([Main/app.py](Main/app.py) `refresh()`), and the
   public "Refresh Baselines" button on the predictions page calls it. Any visitor can trigger a
   multi-minute `update_daily.main()` run (CPU + third-party API usage). Recommend: require
   `TRIGGER_SECRET` and remove the button from the public navbar (or hide behind a key prompt).
2. **MLB-API team fallback is missing model features** (`fetch_team_baseline_from_mlb_api`,
   [updates/update_daily.py:354-378](updates/update_daily.py#L354)): no `k_per_pa`, `opp_whip`,
   `opp_hr_per9`, `roll7_bullpen_fatigue` (writes the obsolete `roll7_bullpen_ip` key instead).
   Those features silently default when this fallback path is used. Low priority (DB rolling is
   the primary source), but worth aligning.
3. **Past-date prediction cards are recomputed live, not read from the log**
   (`/api/predictions?date=<past>` re-runs `predict_game` with *current* baselines). The card can
   disagree with what was actually predicted (and with the accuracy page). Consider serving
   stored log entries for past dates.
4. **rest_days off-by-one, train vs live**: training clips rest_days to ≥1 (played yesterday = 1);
   live `get_team_rest_days` returns 0 for played-yesterday. Diffs mostly cancel, but boundary
   cases differ.
5. **Old log entries exist with `predicted_winner` inconsistent with `home_win_prob`** (e.g. bins
   0–10% containing wins in the calibration view) — artifacts of pre-fix backfills. The
   immutability fix stops new ones; historical ones remain and slightly distort calibration plots.
6. `Main/predict.py` (CLI helper) loads the pkl by relative path — only works if run from the
   right directory. Dev tool only.
7. `%-I` in `strftime` (schedule_fetcher) is not Windows-portable — fine on Render/macOS.
8. Six templates duplicate the same `<head>`/navbar block by hand — that's how the navs drifted
   apart. Moving to a Jinja base template (`{% extends "base.html" %}`) prevents recurrence.

## Accuracy improvements (ranked by expected value)

1. **Regenerate the model artifact** (`scripts/RETRAIN_CHECKLIST.md`). The deployed pkl still has
   leaked same-season SP features + the baked-in 1.4× boost. Everything measured says the honest
   model is ~0.576 AUC but *better calibrated live* — and after the k_per_pa fix a retrain also
   lets the model actually use that feature.
2. **Live recalibration layer**: the deployed model's live ECE is 0.0795 (badly over-confident:
   60-70% picks win ~54%). Fit a simple Platt/logistic recalibration on resolved 2026 predictions
   monthly, or raise the current fixed 4% home-prior blend. Cheap, directly improves the
   probabilities users see.
3. **Use the market**: you now store odds durably. Blending model probability with market-implied
   probability (even 50/50) almost always improves log-loss/Brier vs a solo model; also lets you
   report CLV as the primary betting KPI on the betting page.
4. **Down-weight or drop 2021** in training — its SP features are 100% league-average after the
   leak fix (no 2020 data), so it contributes noise (see
   `scripts/results/prior_season_substitution_check.md`).
5. **Career-to-date SP prior** instead of pure prior-season: recovers signal for the ~35% of
   rows whose pitcher lacks an S-1 row, still leak-free. Or compute in-season rolling SP stats
   from the game logs themselves.
6. **Rolling bullpen ERA** (last 30 days from game logs) instead of a season-long/prior-season
   constant per team.
7. **Pitcher handedness as a real feature** — requires backfilling historical handedness
   (`build_handedness_lookup()` exists but was never called); currently handedness is hardcoded 0
   in training so the model can't learn platoon effects.

## Design / UX improvements

1. **Shared Jinja base template** — one navbar/head to rule all pages (prevents drift; smaller
   templates).
2. **Mobile nav**: with 5 links + date controls + refresh, the index navbar wraps to 2–3 rows on
   phones. Collapse page links into a Bootstrap hamburger at `sm` breakpoint.
3. **Favicon**: none is served — add a ⚾ favicon (browser tabs currently show a blank page icon).
4. **Card transparency**: show `model_version` and "prediction made at {prediction_timestamp}" in
   the card footer or details modal; badge post-game backfills ("retro pick") so users can tell.
5. **Calibration chart**: `/api/calibration` already returns bin data — add a small
   reliability chart to the accuracy or explain page (Chart.js is already loaded).
6. **Betting page**: add an "unrated picks" line (count + accuracy for odds-less history) so the
   page doesn't look empty pre-odds; add a CLV-over-time chart once closing odds accumulate.
7. **Home page**: add `og:` meta tags (link previews), `loading="eager"` + compressed hero image
   (baseballBackground.webp also sits unused at the project root — only `static/` copy is used).
8. **Remove the public "Refresh Baselines" button** (pairs with gating `/api/refresh`).

## Ops improvements

1. **Verify backups now flow**: after deploy, Render logs should show
   `[github] ... backed up to GitHub` lines; the repo should start receiving
   `Auto-backup predictions log YYYY-MM-DD` commits. If not, check `GITHUB_TOKEN` on Render.
2. **Rotate secrets**: `ODDS_API_KEY` and `TRIGGER_SECRET` were pasted into chat transcripts.
3. **Pin dependencies** in `Main/requirements.txt` (only scikit-learn is pinned; a surprise
   major-version bump of pandas/xgboost on redeploy could break unpickling or the pipeline).
4. **Grow the test suite**: `scripts/smoke_test.py` exists; add pytest cases for
   `_compute_odds_fields`, `_calibration_bucket`, the betting upsert (COALESCE semantics), and
   `find_pitcher_by_name`.
