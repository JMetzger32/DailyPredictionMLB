# Retrained artifact comparison

_Generated 2026-08-17T11:37:11 by `scripts/compare_retrained_artifact.py` (report-only)._

`pkl_BACKUP_preretrain.pkl` -> `mlb_model_artifacts.pkl`

Unlike the earlier validations, which held GB/XGB fixed to isolate the LR penalty, this compares **fully retrained artifacts** — LR, GB and the 50 bootstrap XGBs all rebuilt on leak-fixed features.

## Artifact metadata

| metric | old | new |
|---|---|---|
| model_version | 2c50e24e590d | 24c1e93ac246 |
| saved_at | 2026-08-13T15:42:46.201197 | 2026-08-17T11:36:05.168748 |
| holdout accuracy | 0.5269770879526977 | 0.5417590539541759 |
| holdout Brier | 0.25063278501448877 | 0.24862068456400954 |
| holdout log loss | 0.6946464660359826 | 0.6904207859680902 |
| train_size | 12148 | 12148 |
| val_size | 1353 | 1353 |

Reconstruction gate: the old artifact reproduces its own logged probabilities to max abs error **0.00049** on 49 rows.

## LR coefficients

Coefficients driven to exactly zero by the elastic-net penalty: **6/18**

- `diff_roll30_obp`
- `diff_roll30_runs_allowed`
- `diff_roll10_win_pct`
- `diff_roll7_bullpen_fatigue`
- `diff_sp_ip_gs`
- `diff_sp_k_bb`

| feature | old | new | delta |
|---|---|---|---|
| diff_roll30_opp_whip | -0.1685 | -0.1351 | 0.0334 |
| diff_sp_xfip | -0.0997 | -0.1166 | -0.0169 |
| diff_roll30_iso | 0.0859 | 0.0716 | -0.0143 |
| diff_sp_siera | -0.0676 | -0.0092 | 0.0585 |
| diff_bullpen_era | -0.0673 | -0.0652 | 0.0020 |
| diff_roll30_k_per_pa | -0.0593 | -0.0436 | 0.0157 |
| diff_roll30_opp_strikeouts | 0.0445 | 0.0383 | -0.0062 |
| diff_sp_k_bb | -0.0403 | 0.0000 | 0.0403 |
| diff_pyth_win_pct | 0.0403 | 0.0368 | -0.0035 |
| diff_roll10_runs_scored | 0.0318 | 0.0190 | -0.0127 |
| diff_sp_era | 0.0295 | 0.0046 | -0.0249 |
| diff_roll30_opp_hr_per9 | 0.0232 | 0.0132 | -0.0101 |
| diff_roll30_obp | -0.0216 | 0.0000 | 0.0216 |
| diff_roll30_runs_allowed | 0.0199 | 0.0000 | -0.0199 |
| diff_roll7_bullpen_fatigue | 0.0196 | 0.0000 | -0.0196 |
| diff_sp_ip_gs | -0.0099 | 0.0000 | 0.0099 |
| diff_roll10_win_pct | -0.0084 | 0.0000 | 0.0084 |
| diff_rest_days | -0.0023 | -0.0039 | -0.0016 |

## Served probabilities

| metric | old | new |
|---|---|---|
| sd of served prob | 0.09149 | 0.08897 |
| mean |p-0.5| | 0.08095 | 0.07865 |
| max |p-0.5| | 0.25368 | 0.26648 |
| mean |delta| vs old |  | 0.01495 |
| max |delta| vs old |  | 0.05892 |
| predicted winner flips |  | 20.00000 |

## True betting edges (n=362 priced games)

| metric | old | new | change |
|---|---|---|---|
| value bets (good) | 120.0000 | 113.0000 | -7.0000 |
| extreme (>0.12) | 46.0000 | 44.0000 | -2.0000 |
| bad (<-0.05) | 44.0000 | 51.0000 | 7.0000 |
| unsure | 152.0000 | 154.0000 | 2.0000 |
| mean edge (all) | 0.0440 | 0.0382 | -0.0058 |
| resolved value bets | 115.0000 | 109.0000 | -6.0000 |
| value-bet win% | 0.5739 | 0.5780 | 0.0041 |
| flat-bet ROI | 0.1167 | 0.1038 | -0.0129 |

- entering the value-bet band: **23**; leaving it: **30**

**Read the counts, not the ROI.** Resolved value-bet counts are ~100, far below the ~400/bucket CLAUDE.md notes are needed to resolve a 10pp win-rate gap, so win% and ROI differences here are not evidence either way. Whether the bet volume holds up is the question this n can answer.
