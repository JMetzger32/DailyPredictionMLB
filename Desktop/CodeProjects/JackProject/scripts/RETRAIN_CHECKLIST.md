# Artifact regeneration checklist — mlb_model_artifacts.pkl

**⚠ UPDATED (Phase E2): the retrain MUST run on `integrate/full-fix` (or main after it
merges) — NOT the old `fix/eval-and-leak-phased` branch alone.** The integrated branch
combines everything the new artifact needs:
- prior-season (S-1) SP/bullpen features — the leak fix
- `k_per_pa` flowing into live team baselines (feature was always 0 in production)
- **2021 down-weighted** (YEAR_WEIGHTS 1.0 → 0.3) in both training paths
- no 1.4× post-hoc boost; `model_version` + `saved_at` stamped at save

**Why:** the deployed pkl (13 MB, `updates/mlb_model_artifacts.pkl`) still contains the OLD
model: leaked same-season SP/bullpen features, the (now-removed) 1.4× SP-ERA coefficient boost
baked into `lr_model.coef_`, and `model_version=None`. All fixes are in code; the artifact
catches up only when retrained in an xgboost-capable environment. **Do not retrain on a machine
without working xgboost** — the bootstrap ensemble would be silently dropped from the artifact.

**Practical path (recommended):** merge → Render deploys (Render HAS xgboost) → hit
`GET /api/retrain-model?key=<TRIGGER_SECRET>` (weekly-retrain path, ~5-10 min) or run
`python Main/MLBModel.py` in a Render shell for the full offline rebuild incl. baselines.
Execution attempts are logged in `scripts/RETRAIN_EXECUTION_LOG.md`.

## 0. Environment prerequisites
- [ ] Python env with `Main/requirements.txt` installed — notably `xgboost`,
      `scikit-learn==1.9.0` (the current pkl was written by 1.9.0; keep the version pinned to
      avoid `InconsistentVersionWarning`/unpickle risk), `pandas`, `numpy`.
- [ ] xgboost actually loads: `python -c "import xgboost"` — on macOS this needs OpenMP
      (`brew install libomp`); on Render/Linux it works out of the box.
- [ ] DB present at `Databases_and_logs/mlb_allseasons.db` with 2021–2026 games,
      `pitcher_stats` (2021–2025), `team_bullpen_stats` (2021–2025).
- [ ] On branch `integrate/full-fix` (or after merge, `main` with these commits).

## 1. Back up the current artifact (rollback point)
```bash
cp updates/mlb_model_artifacts.pkl updates/mlb_model_artifacts.pkl.bak
```
- [ ] Backup exists and is ~13 MB.

## 2. Snapshot OLD predictions for the side-by-side (BEFORE retraining)
Run the smoke/predict path against the current pkl and save a fixed slate:
- [ ] Pick a fixed slate: today's schedule or a fixed list of ~15 team/SP pairs from
      `team_baselines`/`sp_baselines`.
- [ ] For each pair, record `home_win_prob` from `predict_game(...)` with the OLD pkl
      (e.g. adapt `scripts/smoke_test.py` to loop and dump JSON to
      `scripts/results/old_model_slate.json`).

## 3. Retrain
Two equivalent paths — pick one:
- **Full offline rebuild (preferred for this regeneration):**
  ```bash
  python Main/MLBModel.py
  ```
  Rebuilds features (now prior-season SP/bullpen), trains LR/GB/XGB + 50-model bootstrap,
  rebuilds team/SP baselines, writes the pkl with `model_version` stamped. No 1.4× boost.
- **Weekly-retrain path (what prod cron uses):** call `update_daily.retrain_model()`
  (via `GET /api/retrain-model?secret=...` on the server, or a Python one-liner). Updates
  models in the existing pkl; keeps current baselines.

## 4. Validate the new artifact
- [ ] `python scripts/smoke_test.py` → `[smoke] OK`, `xgb_ensemble=on` (NOT "stubbed"),
      and a non-None `model_version` printed.
- [ ] Inspect keys: `xgb_bootstrap_models` has 50 entries; `model_version` is a 12-char hash;
      `saved_at` present.
- [ ] Confirm no boost: max |lr coef| for `diff_sp_era` should be its fitted value (the old
      artifact had it multiplied 1.4×; the new one must be the plain fit).
- [ ] 2025-holdout sanity: expect ensemble AUC ≈ 0.576 / LogLoss ≈ 0.681 / Brier ≈ 0.244
      (leak-fixed baselines from `scripts/PHASE4_RESULTS.md`; the 2021 down-weight and
      k_per_pa fix may shift these slightly — a point or two either way is fine); the old
      leaked ~0.608 is NOT expected — matching it would suggest the leak fix didn't take.

## 5. Side-by-side OLD vs NEW prediction comparison
- [ ] Score the same fixed slate from step 2 with the NEW pkl → `new_model_slate.json`.
- [ ] Diff per game: `Δprob = new - old`. Flag:
      - any **sign flip** (predicted winner changes) — expected for some games, but review
        each one; a flip rate > ~30% of the slate warrants investigation.
      - any |Δprob| > 0.15 — investigate the inputs (SP baseline changes are the usual cause).
- [ ] Write the table to `scripts/results/old_vs_new_slate.md`.

## 6. Deploy / rollback
- [ ] Deploy = commit the regenerated pkl (it lives in-repo) or let the Render 8 AM job's
      `_push_file_to_github(ARTIFACTS_PATH, ...)` back it up after the on-server retrain.
- [ ] Watch the next 2–3 days of live predictions: `model_version` now appears on each new
      predictions_log entry, so old-vs-new is cleanly separable in the log.
- [ ] **Rollback:** restore the backup and reload:
      ```bash
      cp updates/mlb_model_artifacts.pkl.bak updates/mlb_model_artifacts.pkl
      # then restart the app or hit POST /api/refresh to reload artifacts
      ```
      Rollback triggers: smoke test fails, bootstrap ensemble missing, holdout metrics
      wildly off the §4 expectations, or live accuracy collapses vs the old model's 53.5%.

## Known caveats going in
- 2021 training rows have 100% league-average SP features (no 2020 data) and ~35% of
  2022–2024 rows fall back too (see `scripts/results/prior_season_substitution_check.md`);
  consider down-weighting 2021 in a future iteration.
- Honest expectations: the leak-fixed model measures ~0.576 AUC on the 2025 holdout — lower
  than the leaked 0.608, but real. Live calibration should improve (ECE 0.0266 vs the old
  model's live 0.0795; see `scripts/results/calibration_comparison.md`).
