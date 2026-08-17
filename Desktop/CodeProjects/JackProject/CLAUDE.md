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
- **FIXED 2026-08-17** (`fix/sp-lookahead-leak`): **2026 SP rows had a look-ahead leak.**
  `merge_sp_stats` uses each pitcher's prior-season (S−1) stats on purpose
  (`Main/MLBModel.py:495-502`), but `update_daily.py::retrain_model` overwrote every 2026
  row with a *current* season-to-date snapshot from the pkl. Verified before the fix:
  within-pitcher sd of `sp_era`/`sp_xfip`/`sp_siera` was **0.000000 across 217/217**
  pitchers with ≥3 starts — an April game trained on August stats, at the highest sample
  weight (1.8). Root cause was a key mismatch, not the snapshot itself: 2026 `games` rows
  store pitcher **names**, never retro IDs (0 of 2706), so the `(pid, season−1)` lookup
  missed and fell back to league average. Fix: `resolve_missing_pitcher_ids` (called at the
  top of `merge_sp_stats`, so both training paths get it from one place) maps names →
  retro IDs. Two data-quality traps make a naive match fail — `pitcher_stats.player_name`
  carries Baseball-Reference handedness markers (`MacKenzie Gore*`, `*`=LHP, `#`=switch)
  **and** mojibake (`Cristopher SÃ¡nchez*`, UTF-8 stored as Latin-1, so accent-folding
  alone won't fix it). Repairing both lifts resolution 53.8% → **70.7% of 2026 rows**,
  inside the normal 63-76% range every other season gets. Provenance check: 100% of
  resolved rows now carry completed-2025 stats (was 0%). **Residual: ~290 rows (2026
  debuts / 2025 absences) still use the snapshot and keep look-ahead** — deliberate, the
  alternative is league average. Note **within-pitcher sd cannot validate this fix** —
  prior-season stats are constant within a season by design; the test is *provenance*
  (which season the constant comes from). See `scripts/results/sp_leak_verification.md`.
- **Leak fix did NOT uniformly shrink SP weights** (`scripts/results/elasticnet_penalty_validation.md`).
  The natural hypothesis — leak inflated SP importance, so fixing it shrinks those
  coefficients — is **wrong as stated**. It redistributed weight *within* the SP cluster:
  `diff_sp_xfip` −0.1003 → **−0.1537** (grew), `diff_sp_siera` −0.0660 → **+0.0034**
  (collapsed to zero), `diff_sp_k_bb` −0.0390 → −0.0118. VIF barely moved (xfip 11.55 →
  11.51, siera 13.04 → 13.08). Consistent with eda_4's finding that xfip/siera are ~95%
  redundant and the split between them is unstable — the leak was determining *which* of
  the pair won, not how much the pair mattered in total.
- **Moneyline LR penalty is elastic net, not L2** (`LR_PENALTY_KWARGS` in
  `Main/MLBModel.py`, mirrored in `update_daily.retrain_model`): `l1_ratio=0.3, C=0.01,
  solver='saga', max_iter=5000`. Chosen under a pre-registered rule (log loss
  neutral-or-better in ≥4/5 LOSO folds AND ≥1 coefficient driven to exactly zero),
  re-run on leak-fixed data. Zeroes 6 of 18: `diff_roll30_obp`,
  `diff_roll30_runs_allowed`, `diff_roll10_win_pct`, `diff_roll7_bullpen_fatigue`,
  `diff_sp_ip_gs`, `diff_sp_k_bb` — matching eda_4's predicted elimination order. SP
  signal is retained via `diff_sp_xfip` (largest SP coefficient) and `diff_sp_era`.
  `cross_validate_loso`: AUC 0.59039 → 0.59186, log loss 0.67885 → 0.67848, 4/5 folds
  neutral-or-better. **These deltas are inside noise — the value is simplification, not
  accuracy.** Served-probability impact is small (mean |Δ| 0.0023; predicted winner flips
  on 19/1353 games) because LR is only 1/3 of the ensemble and is then blended 4% toward
  `_HOME_PRIOR`. Run-line (`home_covers`) models deliberately still use L2.
- **The committed DB is NOT a fresher copy — it's byte-identical to the local working
  file** (both 14434304 bytes, 2026 games stopping at **20260707**, verified 2026-08-17).
  `Databases_and_logs/mlb_allseasons.db` was last committed at `a1383d8` and production
  never re-commits it (only the JSON logs are auto-backed-up). So `git checkout` of the DB
  buys nothing, and **any analysis needing 2026 game rows after 2026-07-07 has no local
  source** — the odds-carrying prediction rows all start 2026-07-16, so DB features and
  stored prices have *zero date overlap*. Refresh the JSON logs instead:
  `git checkout origin/main -- Databases_and_logs/predictions_log.json ...` (targeted, no
  merge commit) — that alone took odds rows 313 → 362.
- **To analyse games the DB doesn't have, invert `x_scaled_features`.** Every
  `predictions_log.json` entry stores the exact 18-float scaled vector used at predict
  time, so `raw = scaled * scaler.scale_ + scaler.mean_` recovers the raw features with no
  DB at all. Rows carry the `model_version` they were scored under, and older scalers are
  recoverable from git because the pkl is committed (`b9133b95d2ec` lives at commit
  `83fda83`) — so *all* odds rows are usable, not just the current version's. Verified
  exact: replaying the shipped ensemble through this path reproduces the logged
  probability to max abs error **0.00049** (just the log's `round(prob, 3)`). Always gate
  on that round-trip before trusting the reconstruction — `scripts/validate_true_edge.py`
  aborts if it fails.
- **Elastic net does NOT collapse the betting system** (`scripts/results/true_edge_validation.md`,
  n=362 odds rows). Value bets **120 → 117**, extreme 44 → 43, mean edge 0.0420 → 0.0404;
  4 games enter the value band and 7 leave. That was the real risk of a 50x regularization
  increase and it did not materialise, because LR is only 1/3 of the ensemble. Win% (56.5%
  → 57.5%) and flat-bet ROI (+9.2% → +10.9%) also improved, but at n≈115 resolved bets
  that is **not evidence** — CLAUDE.md's own power note says ~400/bucket is needed. Cite
  the counts, not the ROI.
- **Model version is now `24c1e93ac246`** (retrained + deployed 2026-08-17). Version
  history: `b9133b95d2ec` (live 2026-07-16) → `2c50e24e590d` (`fix/scaler-drift`,
  `e2d8572`, 2026-08-13: scaler fit on a recent-seasons window instead of all of 2021+)
  → `24c1e93ac246` (`fix/sp-lookahead-leak`: 2026 SP leak fixed **and** elastic-net
  penalty, so LR, GB and the 50 bootstrap XGBs were all rebuilt on leak-free features).
  Retrain holdout improved on all three: accuracy 0.5270 → **0.5418**, Brier 0.25063 →
  **0.24862**, log loss 0.69465 → **0.69042** — and that understates it, because the old
  model's 2026 validation was itself scored on leaked features. The flip side: the two
  aren't strictly comparable, since the validation data changed too.
  **Everything below this line about calibration, edge segmentation and home-lean was
  measured against `b9133b95d2ec` or `2c50e24e590d`.** Retraining moved served
  probabilities by mean 0.0150 (max 0.059, 20/362 winner flips) and churned ~26% of the
  value-bet slate, so treat those older numbers as historical context, not current fact,
  until re-measured. `scripts/compare_retrained_artifact.py` regenerates this comparison
  for any future retrain.
- **Calibration (ECE)**: was ~0.08 under the pre-leak-fix model. Model version
  `b9133b95d2ec` (live 2026-07-16→2026-08-13) measured **0.024 live at
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
- **Home-lean root cause (2026-08-07, `scripts/results/home_lean_feature_analysis.md`):
  feature-weighting, not a stale home-field prior.** LR intercept (52.98%) matches the
  actual 2026 home rate (53.15%) almost exactly — ruled out. The real mechanism: the
  model's probability moves with `diff_sp_xfip`/`diff_sp_siera` (SP defense-independent
  quality) and `diff_roll10_win_pct`/`diff_roll10_runs_scored` (10-game recent form) **~2-4x
  more than the market's does** (e.g. siera: model corr 0.62 vs market 0.15) — the
  classic signature of overfitting to features an efficient market discounts more
  heavily. `diff_sp_xfip`/`diff_sp_siera` are 98.8% correlated (near-duplicate signal)
  but the LR already weights `xfip` far more than `siera`, so that redundancy is a
  simplification opportunity, **not** the cause. This is a general
  overfit-to-noisy-features issue that happens to net home-leaning on the current
  sample because home teams currently look marginally better on exactly those
  over-weighted features — **not** a home/away-specific bug (no home indicator exists
  anywhere in `FEATURE_COLS`; every feature is a symmetric `home_X − away_X` diff).
  Deliberately NOT fixed by patching model weights ad hoc — needs a proper retrain +
  validation cycle (new CV holdout, leak checks vs the current live artifact —
  `2c50e24e590d` as of 2026-08-16, see the model-version note above) before touching
  `Main/MLBModel.py`. If retraining, start by shrinking/reconsidering `diff_roll10_*`
  and re-checking how much weight `diff_sp_xfip`/`diff_sp_siera` should carry. Note
  this analysis's own correlations were measured against `b9133b95d2ec` and are also
  unverified against `2c50e24e590d`.
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
