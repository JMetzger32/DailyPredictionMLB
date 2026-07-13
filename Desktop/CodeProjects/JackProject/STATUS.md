# STATUS — as of 2026-07-13 (branch fix/betting-accuracy-recovery, pre-deploy)

If something is broken, start with **OPERATIONS.md**, not this file.

## Root causes found and fixed this cycle

1. **Betting page empty for months** — betting rows need BOTH odds (captured
   pre-game) and a result; Render's free-tier disk wipes on every restart
   (~15 min idle), the GitHub-backup safety net has NEVER landed a commit, and
   its failure printed only to unread logs. Every wipe left results-without-odds.
   Fixed in code: loud `[github] PUSH OK/FAIL` logs, `/api/status` shows the last
   backup error + token presence, betting page explains its own emptiness with row
   counts. **Still open on ops side**: GITHUB_TOKEN is set on Render and unexpired
   (per user, 2026-07-13) yet no backup commits exist — first post-deploy check is
   `/api/status → jobs.github_backup.last_error` for the real reason.
2. **Accuracy page looked frozen** — 60s server cache + background heal thread +
   zero Cache-Control headers. Fixed: `no-store` on all /api/*, freshness line +
   heal-in-progress flag in UI.
3. **Accuracy "drop" after retrain** — not a bug: the old 59% holdout was
   leak-inflated (prior-season SP stats); honest is ~54% vs 52% home baseline.
   Live model artifact: model_version f35f679086c3 (val acc 0.544, Brier 0.2468).

## Features added

- `/api/betting/weekly` (ISO-week aggregates, `?week=` detail) + weekly UI table
- Quarter-Kelly sizing: `?bankroll/kelly_fraction/max_stake_pct` params, per-bet
  stake/P&L columns, summary tile; default 1/4 Kelly because live ECE ≈ 0.08
- `/api/status` job-health endpoint + job-status registry (`[job:*] OK/FAIL` logs)
- Past-date predictions: inference skipped when stored predictions exist;
  fully-resolved dates response-cached (2ms vs ~1s locally, much bigger win on Render)
- Backtests implemented (report-only): `scripts/backtest_threshold.py`,
  `scripts/backtest_kelly.py` — **HOLD: no threshold change until ~200 resolved
  value bets exist** (currently 0; data starts accumulating once backups land)
- `CLAUDE.md` (repo guide), `OPERATIONS.md` (runbook), evidence per phase in
  `scripts/results/phase*.md`; stale process docs archived (user removed archive/)

## Verification

11/11 unit tests · smoke OK · 9/9 e2e pipeline stages (incl. the exact
restart/NULL-re-upsert case that caused the outage) · all 9 endpoints 200 —
see scripts/results/phase9_verification.md.

## Unresolved / waiting

- **Why GitHub backups fail on Render** — needs the deployed branch's /api/status
  or a `[github]` line from Render logs (token itself checks out).
- Betting/Kelly/weekly pages stay empty until odds+result pairs accumulate
  post-deploy (~15 games/day; ~200 needed for threshold work).
- origin/main is 4 commits ahead (README/CNAME only) — merge at deploy time.
