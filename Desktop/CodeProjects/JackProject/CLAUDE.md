# DailyPredictionMLB — repo guide

Flask app that predicts MLB games and tracks betting performance.
Live at dailypredictionmlb.onrender.com (Render **free tier** — see Deploy notes).

## Map

- `app.py` (root) — thin wrapper importing `Main/app.py` for gunicorn.
- `Main/app.py` — THE app (~2500 lines): all routes, APScheduler jobs, GitHub
  backup/restore, caches. `Main/MLBModel.py` — training + `predict_games_batch`
  (batched ensemble: LR + GB + XGB + 50 bootstrap XGBs; never predict per-game
  in a loop, it's 15x slower).
- `updates/` — data pipeline: `schedule_fetcher.py` (MLB StatsAPI + odds),
  `update_daily.py`, `init_betting_log.py`, `mlb_model_artifacts.pkl` (committed
  model artifact; `model_version` key identifies it).
- `Databases_and_logs/` — `mlb_allseasons.db` (SQLite: games, pitcher_stats,
  betting_log), `predictions_log.json` (append-only prediction history),
  `job_status.json` (runtime, gitignored).
- `templates/` + `static/` — Bootstrap UI. `scripts/` — analysis/backtests;
  evidence for past work in `scripts/results/`. `archive/` — retired files, gitignored.
- `OPERATIONS.md` — runbook: symptom → what to check. Read it before debugging prod.

## Commands

- Tests: `.venv/bin/python tests/test_units.py` (plain asserts, no pytest needed).
  App functions are regex-extracted from source (`_extract`) to avoid the
  network-heavy import of `Main/app.py`.
- Run locally: `PORT=5099 .venv/bin/python Main/app.py` — startup takes 1–3 min
  (SP refresh + backfill); poll `/health` until ready. Uses local xgboost (needs
  `brew install libomp`).
- Smoke test: `.venv/bin/python scripts/smoke_test.py` (pkl load + one prediction).

## Rules (learned the hard way)

1. **Running the app dirties tracked data files** (DB, predictions_log, even the
   pkl via baseline refresh). Before ANY commit: `git status`, then
   `git checkout --` the data files unless the data change is the point.
   KILL the actual python process after boot tests (killing the shell wrapper
   leaves an orphan whose 8 AM scheduler keeps writing files).
2. **Git repo root is $HOME**, not the project dir — always run git from the
   project dir with cwd-relative paths (`git log -- scripts/` works;
   `git log -- Desktop/...` silently returns nothing).
3. Never push/merge/deploy without the user's explicit go-ahead.
4. Commit messages via `git commit -F <file>` (heredocs break in this shell).
5. `*.db` is gitignored — the committed DB required `git add -f`.
6. **Local clone can silently drift far behind origin** — production auto-commits
   `predictions_log.json`/`betting_log.json`/`closing_odds_log.json` straight to git
   every day, so a local checkout goes stale fast if you don't pull. `git fetch origin`
   before trusting local data for analysis. `scripts/download_logs.py` (GitHub
   Contents API) 404s on `predictions_log.json` once it exceeds the API's ~1MB read
   limit (it's ~1.3MB as of 2026-07) — use `git pull` instead, which has no such limit.

## Domain facts

- Honest model accuracy is ~54% (home-team baseline 52%); the old "59% holdout"
  was leak-inflated (prior-season SP stats leak, fixed 2026-07). Betting uses
  quarter-Kelly (changes to the 0.05 edge threshold require backtest evidence:
  `scripts/backtest_threshold.py`, ~200+ resolved bets).
- **Calibration (ECE)**: was ~0.08 under the pre-leak-fix model. The current
  `model_version` (`b9133b95d2ec`, live since 2026-07-16) measured **0.024 live at
  n=283 (2026-08-06)** — better than the ~0.05 seen at n=91, and close to the ~0.02 of
  a clean offline holdout, so the train/serve-skew gap has largely closed. A fitted
  Platt calibrator gets ECE to 0.0097 out-of-fold (slope **0.772 < 1 — the model IS
  overconfident**), but **it was tested and deliberately NOT shipped** (2026-08-06,
  `scripts/results/calibrated_edge_comparison.md`): better calibration did not produce
  better betting — Brier got slightly worse (0.2426→0.2452), value-bet win% fell
  54.5%→53.1%, and flat-bet ROI fell +10.3%→+0.8%. Recalibration fixes the confidence
  *scale* but adds no *discrimination*. Don't re-propose it without new evidence.
- **Edge does NOT reliably predict winning — and the reasons are subtler than they look.**
  Current numbers (n=222 rated bets, 2026-07-16→08-06, `scripts/segment_value_bets.py`):
  edge 0.05–0.08 wins 63.9%/+20.2% ROI, 0.08–0.12 wins 51.6%/−1.1%, 0.12+ wins
  51.9%/**+5.8%**. Edge is non-monotonic (it stops discriminating above ~8pp), but the
  older CLAUDE.md claim that >0.12 loses ~40% is **stale — that was n=38**. Three
  statistical caveats that should stop over-reading any of this:
  1. `logit(win) ~ edge` gives **p=0.42** — edge has no *demonstrable* relationship with
     winning at this n. Neither the bullish nor bearish reading is supported.
  2. The famous Value-Bet-vs-Toss-Up "inversion" (54.5% vs 59.4%) is **p=0.51 — not
     significant.** Do not treat it as an established model property.
  3. Model AUC 0.598 vs market AUC 0.594 — **the model does not demonstrably beat the
     market.** That (discrimination/features), not calibration, is the real open problem.
- **FIXED 2026-08-07** (`fix/edge-calibration`): `bet_rating` is still written once at
  odds-attach time and frozen (values `good` edge 0.05–0.12, `extreme` >0.12 — excluded
  from value-bet lists and Kelly sizing, `bad` < −0.05, `unsure`; `_kelly_stake` still
  caps the sizing probability at `market_prob ± 0.10`), but the stored column is **no
  longer trusted for display bucketing**. `/api/betting` and `/api/betting/weekly` now
  call `_rate_edge(model_edge)` (`Main/app.py`, single source of truth also used inside
  `_compute_odds_fields` when first persisting the column) so every row is classified
  under TODAY's thresholds regardless of when odds attached. This fixed the vintage-mix
  bug where 10 pre-07-23 rows with edge ≥0.12 stayed misfiled as `good` (before the
  `extreme` tier existed) — that mixing was where most of the apparent Value-Bet/Toss-Up
  inversion came from. Betting page now reads Value Bets **58.2%** (vs Toss-Ups 59.4%, a
  1.2pp gap, not the old 54.5%/4.9pp) and Quarter-Kelly net **$14.04** (not $0.04). No
  historical `bet_rating` values were rewritten — this is read-time-only, matching the
  precedent at `_calibration_bucket`'s docstring (old rows intentionally not migrated).
- **De-vig is already correct — don't re-investigate it.** `away_implied`/`home_implied`
  are normalized by their sum in `updates/schedule_fetcher.py` (verified: all stored pairs
  sum to exactly 1.0). Edge also reconstructs exactly from `away_ml`/`home_ml` (222/222
  rows, max diff 0.0), so offline analysis isn't limited to rows storing implied probs.
- **FIXED 2026-08-07** (`fix/edge-calibration`): the `clv` field used to be mislabeled —
  `model_prob − closing_implied` (the model's own edge re-measured against a later line,
  never a comparison of two prices; stored "CLV" read +4.67% and looked like the model
  crushes the close, while true CLV was −0.52%, indistinguishable from zero). `clv` now
  means real closing line value (`closing_implied − bet_implied` for the picked side,
  computed in `_store_closing_odds`, `bet_implied` from `away_implied`/`home_implied`
  when persisted else recomputed from `away_ml`/`home_ml` via the new `_implied_probs`
  helper). The old quantity is preserved under its honest name, `edge_vs_close` — still
  useful as an overconfidence diagnostic, but **never cite it as evidence the model beats
  the market.** 158 already-resolved rows in both JSON logs were backfilled in place
  (`scripts/migrate_clv_field.py`, pure recomputation from already-stored fields, no
  refetching) so `clv_stats`/`avg_clv` on the betting page are correct immediately, not
  just for new entries. See `scripts/results/clv_and_home_skew.md`.
- **Value bets skew home (64 vs 30) and the cause is NOT the `_HOME_PRIOR` blend** — an
  earlier version of this file blamed the blend; that was wrong. The blend pulls toward
  0.53, *below* the model's own 0.5432 mean, so it slightly REDUCES home lean. The real
  mechanism: the model runs **+1.37pp more home-leaning than the market** (model 0.5483,
  market 0.5346, actual 0.5315 — the market is closer to truth), paired t-test
  **p=0.027**, one of the few significant findings. That excess lean inflates home-side
  edge on nearly every game and roughly doubles home value-bet flagging. Overconfidence
  itself is symmetric (+1.6pp on both sides), so this is a home-feature problem, not a
  calibration one.
- **Don't re-bin edge hunting for a profitable band — the sample can't support it.**
  Bands of n≈20-30 produce a sawtooth (66%/50%/73%/58%/42%/55%) with every 95% CI
  overlapping every other. Detecting a 10pp win-rate gap needs ~400 bets *per bucket*;
  5pp needs ~1,600. No threshold beats betting every positive-edge game. **Prefer true
  CLV as the tuning target**: sd 0.12 vs 0.50 for a win/loss, so one CLV reading is worth
  ~17 win/loss readings, and it's known the same evening instead of waiting on outcomes.
- **FanGraphs scraping is currently dead.** `pybaseball.pitching_stats()` (used by
  `update_daily.py`'s `fetch_sp_baselines`) 403s unconditionally (confirmed 2026-07,
  reproduces off-Render too — not an IP block specific to Render). Production always
  falls through to `fetch_sp_baselines_from_mlb_api`, which already blends
  current-season SP stats toward the prior season by games-started
  (`alpha = min(gs/10, 1.0)`). Don't assume `fetch_sp_baselines`'s FanGraphs path or
  its `min_gs` threshold logic is actually running live — check for the 403 first.
- **SP baselines are keyed inconsistently, so merges MUST dedup by name** (fixed
  2026-07-28, commit `4a45193`). The committed pkl keys pitchers by Retrosheet ID
  (`kikuy001`) while `fetch_sp_baselines_from_mlb_api` keys by name-slug
  (`yusei_kikuchi`), so a plain `{**old, **fresh}` merge leaves TWO entries per
  pitcher. `find_pitcher_by_name` returns the first exact-name match in dict order
  (old keys first), so the stale prior-season entry shadowed the fresh one and the
  card showed last year's ERA/WHIP/FIP as if live. Both merge points now call
  `update_daily.merge_sp_baselines_dedup(old, fresh)` — the daily job AND the app
  startup refresh (which previously did no dedup at all) — deduping by the SAME
  `_normalize_name` the lookup uses. It also flags prior-only survivors
  `is_prior_year=True` (the pkl ships all 369 entries as `is_prior_year=False`, so
  they'd otherwise render unlabeled). `find_pitcher_by_name` also now picks the
  freshest among multiple exact matches (higher `gs`, not prior-year) as a backstop.
  Note the raw display fields (`era_raw`/`whip_raw`/`fip_raw`) and the `is_*` flags
  are display-only — the model reads `era`/`whip`/`xfip`/… so this is a UI-correctness
  fix, not a model change. Team stats never had this bug (fixed Retrosheet team-code
  keys, plain `.get()`, no fuzzy name matching).
- betting_log rows need BOTH `bet_rating` (odds at prediction time) AND `correct`
  (resolution) to count. Odds are unrecoverable after game time; results are
  always re-fetchable. See scripts/results/phase1_root_causes.md.

## Deploy notes (Render free tier)

- **Disk resets to the deploy snapshot on every restart/spin-down** (~15 min idle).
  The durable persistence is the GitHub contents-API restore-on-boot
  (`GITHUB_TOKEN` env var; that API only reads files < 1 MB, and
  `predictions_log.json`/`betting_log.json` now exceed that) — but the daily
  `Auto-backup` job also commits these files straight to git regardless of size,
  so a fresh deploy still gets current data via the git snapshot itself. If
  `Auto-backup` commits stop landing in the repo, runtime state silently dies.
- 0.1 vCPU: inference is ~30x slower than local; anything per-request must be
  batched/cached. `/api/status` shows job health, token presence, model version.
- **The Odds API budget is 500 credits/month and it HAS run out before.** 2026-07-29→
  07-31 has zero rated bets (~41 games): predictions ran and games resolved, but odds
  never attached, and the blackout ended exactly on 08-01 when the month rolled over.
  Because spin-downs are frequent, every cold-boot page view used to re-fetch odds
  already in the log — the leak that drained it. Fixed 2026-08-06 (`bef2049`): a gate in
  `/api/predictions` reuses odds already on log entries, and `get_last_odds_quota()`
  surfaces `odds_quota` in `/api/status` + `/api/debug/odds` (a 401 with remaining 0
  means exhausted). **If odds stop attaching, check `odds_quota` FIRST** — an empty odds
  map otherwise looks identical to "no games today". Gaps like this drop games from
  betting analyses entirely, so check date continuity before trusting an n.
- Env vars: `ODDS_API_KEY`, `GITHUB_TOKEN`, `TRIGGER_SECRET` (gates /api/retrain-model etc.).
