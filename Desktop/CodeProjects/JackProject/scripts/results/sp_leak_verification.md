# SP look-ahead leak verification

_Generated 2026-08-17T09:39:44 by `scripts/verify_sp_leak.py` (report-only)._

## What is being tested

`merge_sp_stats` uses each pitcher's **prior-season (S-1)** stats for training rows — an explicit leak fix (`Main/MLBModel.py:495-502`) — because same-season aggregates fold in games played *after* the row's own game date. `update_daily.py::retrain_model` (lines 1006-1050) then overwrites the 2026 rows with a **current season-to-date snapshot** from the pkl.

If that snapshot is constant per pitcher across game dates, every 2026 training row carries future information relative to its own game. Measured below as the within-pitcher stddev across each pitcher's own starts (pitchers with >= 3 starts).

## Injection coverage

- live SP baselines injected into **290/2706** 2026 team-game rows
- `sp_baselines` entries in pkl: 369

## 2026 — within-pitcher spread (217 pitchers, 2457 rows)

| stat | mean within-pitcher sd | max | pitchers with sd == 0 | % constant |
|---|---|---|---|---|
| `sp_era` | 0.000000 | 0.000000 | 217/217 | 100.0% |
| `sp_xfip` | 0.000000 | 0.000000 | 217/217 | 100.0% |
| `sp_siera` | 0.000000 | 0.000000 | 217/217 | 100.0% |

## 2025 control — same metric, no injection (276 pitchers)

2025 takes the normal prior-season path, so its spread shows what this metric looks like when there is no injection. Note a *legitimately* constant value is expected here too — prior-season stats are by construction fixed for the whole season — so 2025 is a reference for the metric's behaviour, not a clean contrast.

| stat | mean within-pitcher sd | pitchers with sd == 0 | % constant |
|---|---|---|---|
| `sp_era` | 0.000000 | 276/276 | 100.0% |
| `sp_xfip` | 0.000000 | 276/276 | 100.0% |
| `sp_siera` | 0.000000 | 276/276 | 100.0% |

## Why within-pitcher stddev CANNOT validate the fix

Constancy is the right signal for *detecting* the original leak, but it cannot confirm the repair. Prior-season stats are constant within a season **by design**, so the stddev is zero whether the constant came from completed-2025 (correct, knowable pre-game) or from 2026-to-date (leaked). The distinguishing question is **provenance**, not variance — which is what the next section tests.

## Provenance — the decisive test

For each 2026 row, does `sp_era` equal that pitcher's **completed 2025** ERA (correct: fully knowable before any 2026 game) or their in-progress 2026 ERA (leaked)?

| pipeline | path | rows | % matching completed-2025 ERA |
|---|---|---|---|
| pre-fix | `injected_current_season` | 2108 | 0.0% |
| post-fix | `resolved_prior_season` | 1818 | 100.0% |
| post-fix | `injected_current_season` | 290 | n/a |

**Post-fix, 100.0% of resolved 2026 rows now carry the pitcher's completed-2025 ERA** — the same leak-free prior-season path every other season uses.

## Verdict

- Original leak (pre-fix): **CONFIRMED** — every SP stat identical at every one of a pitcher's 2026 starts (217/217 pitchers), sourced from a retrain-time snapshot rather than a completed season.
- After the fix: **1818/2706 (67.2%)** of 2026 rows take the leak-free prior-season path.
- **Residual:** 290 rows still fall back to the current-season snapshot and retain look-ahead, at the highest sample weight (1.8). These are pitchers with no 2025 `pitcher_stats` row (2026 debuts, or 2025 absences). Deliberate trade-off — the alternative is league average — but a known residual, not a clean fix.
