# Pitching IP measurement fix — validation

Generated 2026-08-13 on branch `fix/pitching-ip-measurement` (stacked on the
other three Phase 1 batch-1 branches). Implements the corrected-IP formula
from `EDA/eda_3/01_pitching_ip_audit.md` (mean|error|=0.368 IP vs. the
even-split, corr(error, win)=-0.553) in production, with a 3-tier fallback.

## What changed

`Main/MLBModel.py::build_team_game_log()` gains an optional
`boxscore_ip_lookup` param and a new `_resolve_total_ip()` helper that
computes per-side pitching IP in priority order: (1) real per-side IP
summed from `game_pitcher_lines` (`feat/boxscore-ingestion`, once games are
backfilled there), (2) the corrected opponent's-outs formula where
`length_outs` is populated (2020-2025), (3) the even-split `length_outs/6`
default — unchanged behavior, and today the only option for 2026, since
`length_outs` is 100% NULL there.

`load_boxscore_ip_lookup(db_path, df)` (new) builds the `game_id -> {home_ip,
away_ip}` dict from `game_pitcher_lines`, returning `{}` (safe no-op) if the
table is empty or missing. Wired into all three call sites that build a
`tgl` from `games`: `Main/MLBModel.py`'s `__main__`, and — a scope
correction from the plan — **two** call sites in `updates/update_daily.py`,
not one: `retrain_model()` (training) and `compute_rolling_baselines_from_db()`.

**The second one matters**: `compute_rolling_baselines_from_db()` is the
*live* daily-job team-baseline computation (`update_daily.py::main()`,
the 8 AM ET job), not a training-only path. Its docstring says it exists to
guarantee "no feature mismatch" with training, and its output directly
includes `roll30_opp_whip` and `roll7_bullpen_fatigue` — both live
`FEATURE_COLS` model inputs — computed from the same buggy `total_ip`. This
means the even-split bug was **not** confined to historical backtesting as
originally scoped; it was also silently affecting every live prediction's
bullpen/opponent-WHIP features for 2026 games (100% even-split, since
`length_outs` is NULL all season) until this fix. `compute_team_baseline`/
`fetch_team_baseline_from_mlb_api` (the plan's original claim) are a
*separate*, genuinely-unaffected fallback path used only when DB rolling
data is unavailable for a team — that part of the original scoping was
correct, just incomplete.

## LOSO 2021-2025 — even-split vs. corrected formula (tier 2)

`boxscore_ip_lookup` is empty against the real DB (no boxscores backfilled
yet — see `feat/boxscore-ingestion`), so this isolates tier 2 (corrected
formula, 2020-2025) vs. tier 3 (even split) — 2026 is identical in both runs
either way, since tier 2 can't compute there.

| | LR AUC | LR Brier | LR LogLoss | GB AUC | GB Brier |
|---|---|---|---|---|---|
| even-split (baseline) | 0.5907 | 0.2429 | 0.6787 | 0.5751 | 0.2465 |
| corrected formula | 0.5904 | 0.2429 | 0.6789 | 0.5752 | 0.2464 |

Flat, no consistent direction (LR marginally down, GB marginally up, both
within noise) — matches EDA section 01's own read: this is a correctness
fix, not a discrimination improvement, and shouldn't be expected to move
accuracy.

## Home-lean shift check (does the fix explain any of the +1.37pp home lean?)

Reproduced section 01's exact check on the production code path:

| feature | even-split mean(diff) | corrected mean(diff) | sign flip? |
|---|---|---|---|
| diff_roll30_opp_whip | +0.0004 | +0.0009 | no |
| diff_roll7_bullpen_fatigue | -0.0159 | **+0.2294** | **yes** |

Matches the EDA numbers exactly. **This fix does not shrink the home/away
gap — `diff_roll7_bullpen_fatigue` grows ~14x larger and flips from a small
home disadvantage to a home advantage.** Confirms section 01's conclusion:
the IP-measurement bug is not a contributing cause of the model's home lean
(`home_lean_feature_analysis.md`'s feature-weighting explanation stands).
**Ship this as a measurement-correctness fix. Do not describe it as
reducing home bias** — the evidence points the other way.

## Tier-1 fallback path (game_pitcher_lines-sourced)

No real boxscore data exists in the DB yet to exercise this against (branch
3's full backfill is a deferred follow-up), so validated with a synthetic
fixture: 3 known game_ids given fabricated `{home_ip, away_ip}` values
distinct from what tier 2 would produce. Confirmed the fixture's values
were used verbatim for those games, and games absent from the fixture fell
through to tier 2 unaffected — the 3-tier priority order works correctly
end-to-end.

## Not shipped as a retrain

Unlike `fix/scaler-drift`, this branch does not retrain/commit a new pkl —
it's a training-pipeline code fix, same treatment as
`chore/fix-team-code-mapping`: takes effect on the next scheduled or manual
retrain, not immediately. `model_version` is unchanged (`2c50e24e590d`).

`tests/test_units.py` (13/13) and `scripts/smoke_test.py` pass.
