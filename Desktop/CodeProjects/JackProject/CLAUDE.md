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

## Domain facts

- Honest model accuracy is ~54% (home-team baseline 52%); the old "59% holdout"
  was leak-inflated (prior-season SP stats leak, fixed 2026-07). ECE ≈ 0.08 →
  betting uses quarter-Kelly, edge threshold 0.05 (changes require backtest
  evidence: `scripts/backtest_threshold.py`, ~200+ resolved bets).
- betting_log rows need BOTH `bet_rating` (odds at prediction time) AND `correct`
  (resolution) to count. Odds are unrecoverable after game time; results are
  always re-fetchable. See scripts/results/phase1_root_causes.md.

## Deploy notes (Render free tier)

- **Disk resets to the deploy snapshot on every restart/spin-down** (~15 min idle).
  The ONLY durable persistence is the GitHub contents-API backup
  (`GITHUB_TOKEN` env var; files < 1 MB restore on boot). If backups aren't
  landing (`Auto-backup` commits in the repo), all runtime state silently dies.
- 0.1 vCPU: inference is ~30x slower than local; anything per-request must be
  batched/cached. `/api/status` shows job health, token presence, model version.
- Env vars: `ODDS_API_KEY`, `GITHUB_TOKEN`, `TRIGGER_SECRET` (gates /api/retrain-model etc.).
