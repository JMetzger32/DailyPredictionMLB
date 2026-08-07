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
  `model_version` (`b9133b95d2ec`, live since 2026-07-16) runs ~0.05 live
  (n=91 as of 07-22) — better, but still worse than the ~0.02 ECE of a clean
  architecture-matched offline holdout, so part of the gap is live-serving-specific
  (train/serve skew), not inherent to the model. Run
  `scripts/calibration_live_check.py` to check current live ECE against the
  currently-deployed `model_version` and see whether a fitted Platt/isotonic
  calibrator now beats the flat 4%-blend via cross-validation — it only recommends
  shipping one once it actually does (at n<100 it typically doesn't; don't force it).
- **Edge is NOT monotonic with bet quality right now.** Live data (n=38 value bets,
  2026-07-16 to 07-22) showed edge 0.05–0.08 winning 64.3% / +15.0% ROI, but edge
  >0.12 winning only ~30% / -40% ROI — a bigger claimed edge is currently a symptom
  of model overconfidence (the model disagrees hardest with the market exactly where
  it's least reliable), not a stronger signal. `bet_rating` has 4 values: `good`
  (edge 0.05–0.12), `extreme` (edge >0.12 — excluded from value-bet lists and Kelly
  sizing), `bad` (edge < -0.05), `unsure`. `_kelly_stake` also caps the probability
  used for sizing at `market_prob + 0.10` regardless of rating. Note `bet_rating` is
  computed once when odds attach to a game and persisted — the `good`/`extreme` split
  only applies going forward, not retroactively to already-tagged rows.
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
- Env vars: `ODDS_API_KEY`, `GITHUB_TOKEN`, `TRIGGER_SECRET` (gates /api/retrain-model etc.).
