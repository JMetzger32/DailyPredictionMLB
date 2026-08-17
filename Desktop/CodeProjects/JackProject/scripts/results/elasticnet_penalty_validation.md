# Elastic-net penalty + SP leak-fix validation

_Generated 2026-08-17T11:08:09 by `scripts/validate_elasticnet_change.py` (report-only)._

## A — Leak-fix impact on the model (step 3)

Both fits use the shipped config (L2, C=0.5, `YEAR_WEIGHTS`, scaler windowed to 2023+). The only difference is the feature pipeline: pre-fix injects a current-season SP snapshot into every 2026 row; post-fix resolves 70.7% of them to retro IDs and uses completed-2025 stats instead.

### The SP features specifically

| feature | coef_pre | coef_post | coef_delta | VIF_pre | VIF_post | abs_change |
|---|---|---|---|---|---|---|
| diff_sp_xfip | -0.1003 | -0.1537 | -0.0535 | 11.5512 | 11.5118 | 0.0535 |
| diff_sp_siera | -0.0660 | 0.0034 | 0.0693 | 13.0405 | 13.0845 | 0.0693 |
| diff_sp_era | 0.0294 | 0.0234 | -0.0061 | 1.4662 | 1.4597 | 0.0061 |
| diff_sp_k_bb | -0.0390 | -0.0118 | 0.0272 | 2.2662 | 2.3069 | 0.0272 |
| diff_sp_ip_gs | -0.0098 | 0.0072 | 0.0170 | 1.0212 | 1.0190 | 0.0170 |

### All features, largest coefficient change first

| feature | coef_pre | coef_post | coef_delta | VIF_pre | VIF_post |
|---|---|---|---|---|---|
| diff_sp_siera | -0.0660 | 0.0034 | 0.0693 | 13.0405 | 13.0845 |
| diff_sp_xfip | -0.1003 | -0.1537 | -0.0535 | 11.5512 | 11.5118 |
| diff_sp_k_bb | -0.0390 | -0.0118 | 0.0272 | 2.2662 | 2.3069 |
| diff_sp_ip_gs | -0.0098 | 0.0072 | 0.0170 | 1.0212 | 1.0190 |
| diff_sp_era | 0.0294 | 0.0234 | -0.0061 | 1.4662 | 1.4597 |
| diff_roll30_opp_whip | -0.1669 | -0.1621 | 0.0048 | 4.7019 | 4.7034 |
| diff_roll30_runs_allowed | 0.0149 | 0.0123 | -0.0027 | 6.0244 | 6.0237 |
| diff_bullpen_era | -0.0675 | -0.0700 | -0.0024 | 1.2598 | 1.2566 |
| diff_roll30_obp | -0.0272 | -0.0292 | -0.0020 | 2.0235 | 2.0240 |
| diff_roll30_k_per_pa | -0.0574 | -0.0594 | -0.0020 | 1.2334 | 1.2341 |
| diff_roll30_opp_strikeouts | 0.0444 | 0.0428 | -0.0016 | 1.3913 | 1.3946 |
| diff_roll30_iso | 0.0842 | 0.0855 | 0.0012 | 1.6254 | 1.6240 |
| diff_roll30_opp_hr_per9 | 0.0263 | 0.0275 | 0.0012 | 1.7630 | 1.7649 |
| diff_pyth_win_pct | 0.0412 | 0.0423 | 0.0011 | 2.0790 | 2.0782 |
| diff_rest_days | -0.0120 | -0.0115 | 0.0005 | 1.1351 | 1.1351 |
| diff_roll10_win_pct | -0.0095 | -0.0092 | 0.0003 | 2.2983 | 2.2992 |
| diff_roll10_runs_scored | 0.0334 | 0.0334 | 0.0000 | 2.2331 | 2.2331 |
| diff_roll7_bullpen_fatigue | -0.0038 | -0.0038 | 0.0000 | 1.2266 | 1.2265 |

### LOSO metrics, pre-fix vs post-fix (same L2 config)

Note LOSO folds cover 2021-2025 only, so they exclude 2026 — the season the leak actually affects. A near-null result here is therefore expected and is **not** evidence the fix did nothing; it mainly reflects that the scaler (fit on 2023+, which includes 2026) shifted.

| season | auc_pre | logloss_pre | brier_pre | n_iter_pre | auc_post | logloss_post | brier_post | n_iter_post |
|---|---|---|---|---|---|---|---|---|
| 2021 | 0.5709 | 0.6830 | 0.2450 | 17 | 0.5709 | 0.6830 | 0.2450 | 17 |
| 2022 | 0.6126 | 0.6724 | 0.2397 | 16 | 0.6126 | 0.6724 | 0.2397 | 16 |
| 2023 | 0.5979 | 0.6783 | 0.2427 | 15 | 0.5979 | 0.6783 | 0.2427 | 15 |
| 2024 | 0.5934 | 0.6800 | 0.2434 | 17 | 0.5934 | 0.6800 | 0.2434 | 17 |
| 2025 | 0.5798 | 0.6800 | 0.2436 | 15 | 0.5798 | 0.6800 | 0.2436 | 15 |

## B — Elastic-net grid, re-run on FIXED data

eda_4 selected `l1_ratio=0.3, C=0.01` on leaked data, where SP features were artificially strong. Same pre-registered rule, re-applied here: **log loss neutral-or-better in >= 4/5 LOSO folds AND >= 1 coefficient driven to exactly zero.**

Grid points meeting both conditions: **4 of 42**.

Qualifying configurations, ranked by log loss (the gate is the filter, log loss the tiebreak):

| l1_ratio | C | mean_auc | mean_logloss | mean_brier | d_logloss | folds_better | n_zeroed | convergence_warnings | max_n_iter |
|---|---|---|---|---|---|---|---|---|---|
| 0.30000 | 0.01000 | 0.59269 | 0.67834 | 0.24269 | -0.00040 | 4 | 2 | 0 | 19 |
| 1.00000 | 0.03000 | 0.59261 | 0.67836 | 0.24270 | -0.00038 | 4 | 3 | 0 | 20 |
| 0.90000 | 0.03000 | 0.59255 | 0.67837 | 0.24270 | -0.00037 | 4 | 2 | 0 | 21 |
| 0.70000 | 0.03000 | 0.59238 | 0.67840 | 0.24271 | -0.00035 | 4 | 1 | 0 | 21 |

**Winner: l1_ratio=0.3, C=0.01** — 4/5 folds, 2 zeroed, mean dlogloss -0.00040.

eda_4's specific pick (`l1_ratio=0.3, C=0.01`) on fixed data: 4/5 folds, 2 zeroed, dlogloss -0.00040 — **still qualifies**.

Full grid:

| l1_ratio | C | mean_auc | mean_logloss | mean_brier | d_logloss | folds_better | n_zeroed | convergence_warnings | max_n_iter |
|---|---|---|---|---|---|---|---|---|---|
| 0.10000 | 0.01000 | 0.59212 | 0.67844 | 0.24274 | -0.00030 | 4 | 0 | 0 | 18 |
| 0.10000 | 0.03000 | 0.59143 | 0.67859 | 0.24281 | -0.00015 | 5 | 0 | 0 | 19 |
| 0.10000 | 0.10000 | 0.59108 | 0.67870 | 0.24286 | -0.00005 | 5 | 0 | 0 | 21 |
| 0.10000 | 0.30000 | 0.59096 | 0.67874 | 0.24288 | -0.00001 | 3 | 0 | 0 | 21 |
| 0.10000 | 0.50000 | 0.59093 | 0.67875 | 0.24288 | 0.00000 | 2 | 0 | 0 | 21 |
| 0.10000 | 1.00000 | 0.59090 | 0.67875 | 0.24289 | 0.00001 | 1 | 0 | 0 | 21 |
| 0.10000 | 3.00000 | 0.59089 | 0.67876 | 0.24289 | 0.00002 | 1 | 0 | 0 | 21 |
| 0.30000 | 0.01000 | 0.59269 | 0.67834 | 0.24269 | -0.00040 | 4 | 2 | 0 | 19 |
| 0.30000 | 0.03000 | 0.59183 | 0.67852 | 0.24277 | -0.00023 | 4 | 0 | 0 | 18 |
| 0.30000 | 0.10000 | 0.59123 | 0.67865 | 0.24284 | -0.00009 | 5 | 0 | 0 | 21 |
| 0.30000 | 0.30000 | 0.59101 | 0.67872 | 0.24287 | -0.00002 | 5 | 0 | 0 | 21 |
| 0.30000 | 0.50000 | 0.59097 | 0.67874 | 0.24288 | -0.00001 | 4 | 0 | 0 | 21 |
| 0.30000 | 1.00000 | 0.59093 | 0.67875 | 0.24288 | 0.00000 | 2 | 0 | 0 | 21 |
| 0.30000 | 3.00000 | 0.59088 | 0.67876 | 0.24289 | 0.00001 | 1 | 0 | 0 | 21 |
| 0.50000 | 0.01000 | 0.59303 | 0.67833 | 0.24268 | -0.00041 | 3 | 4 | 0 | 18 |
| 0.50000 | 0.03000 | 0.59219 | 0.67844 | 0.24273 | -0.00030 | 4 | 0 | 0 | 19 |
| 0.50000 | 0.10000 | 0.59136 | 0.67861 | 0.24282 | -0.00013 | 4 | 0 | 0 | 21 |
| 0.50000 | 0.30000 | 0.59106 | 0.67870 | 0.24286 | -0.00004 | 5 | 0 | 0 | 21 |
| 0.50000 | 0.50000 | 0.59099 | 0.67872 | 0.24287 | -0.00002 | 5 | 0 | 0 | 21 |
| 0.50000 | 1.00000 | 0.59094 | 0.67874 | 0.24288 | -0.00000 | 3 | 0 | 0 | 21 |
| 0.50000 | 3.00000 | 0.59090 | 0.67875 | 0.24289 | 0.00001 | 1 | 0 | 0 | 21 |
| 0.70000 | 0.01000 | 0.59307 | 0.67839 | 0.24271 | -0.00035 | 3 | 6 | 0 | 18 |
| 0.70000 | 0.03000 | 0.59238 | 0.67840 | 0.24271 | -0.00035 | 4 | 1 | 0 | 21 |
| 0.70000 | 0.10000 | 0.59152 | 0.67858 | 0.24280 | -0.00016 | 4 | 0 | 0 | 21 |
| 0.70000 | 0.30000 | 0.59114 | 0.67868 | 0.24285 | -0.00006 | 5 | 0 | 0 | 21 |
| 0.70000 | 0.50000 | 0.59102 | 0.67871 | 0.24287 | -0.00003 | 5 | 0 | 0 | 21 |
| 0.70000 | 1.00000 | 0.59098 | 0.67874 | 0.24288 | -0.00001 | 4 | 0 | 0 | 21 |
| 0.70000 | 3.00000 | 0.59091 | 0.67875 | 0.24288 | 0.00001 | 2 | 0 | 0 | 21 |
| 0.90000 | 0.01000 | 0.59297 | 0.67853 | 0.24278 | -0.00021 | 3 | 7 | 0 | 19 |
| 0.90000 | 0.03000 | 0.59255 | 0.67837 | 0.24270 | -0.00037 | 4 | 2 | 0 | 21 |
| 0.90000 | 0.10000 | 0.59167 | 0.67855 | 0.24279 | -0.00019 | 4 | 0 | 0 | 21 |
| 0.90000 | 0.30000 | 0.59118 | 0.67867 | 0.24285 | -0.00007 | 5 | 0 | 0 | 21 |
| 0.90000 | 0.50000 | 0.59106 | 0.67870 | 0.24286 | -0.00004 | 5 | 0 | 0 | 21 |
| 0.90000 | 1.00000 | 0.59098 | 0.67873 | 0.24287 | -0.00001 | 4 | 0 | 0 | 21 |
| 0.90000 | 3.00000 | 0.59092 | 0.67875 | 0.24288 | 0.00001 | 2 | 0 | 0 | 21 |
| 1.00000 | 0.01000 | 0.59287 | 0.67862 | 0.24282 | -0.00013 | 3 | 7 | 0 | 19 |
| 1.00000 | 0.03000 | 0.59261 | 0.67836 | 0.24270 | -0.00038 | 4 | 3 | 0 | 20 |
| 1.00000 | 0.10000 | 0.59171 | 0.67854 | 0.24278 | -0.00020 | 4 | 0 | 0 | 21 |
| 1.00000 | 0.30000 | 0.59122 | 0.67866 | 0.24284 | -0.00008 | 5 | 0 | 0 | 21 |
| 1.00000 | 0.50000 | 0.59109 | 0.67870 | 0.24286 | -0.00005 | 5 | 0 | 0 | 21 |
| 1.00000 | 1.00000 | 0.59098 | 0.67873 | 0.24287 | -0.00002 | 4 | 0 | 0 | 21 |
| 1.00000 | 3.00000 | 0.59092 | 0.67875 | 0.24288 | 0.00001 | 2 | 0 | 0 | 21 |

## C — Before/after on the fixed pipeline

### LOSO metrics

| season | auc_L2 | logloss_L2 | brier_L2 | n_iter_L2 | auc_EN | logloss_EN | brier_EN | n_iter_EN |
|---|---|---|---|---|---|---|---|---|
| 2021 | 0.5709 | 0.6830 | 0.2450 | 17 | 0.5777 | 0.6820 | 0.2445 | 18 |
| 2022 | 0.6126 | 0.6724 | 0.2397 | 16 | 0.6141 | 0.6723 | 0.2397 | 16 |
| 2023 | 0.5979 | 0.6783 | 0.2427 | 15 | 0.5986 | 0.6779 | 0.2425 | 17 |
| 2024 | 0.5934 | 0.6800 | 0.2434 | 17 | 0.5918 | 0.6803 | 0.2436 | 19 |
| 2025 | 0.5798 | 0.6800 | 0.2436 | 15 | 0.5813 | 0.6793 | 0.2432 | 17 |

Mean AUC 0.59092 -> 0.59269; mean log loss 0.67874 -> 0.67834; folds where EN log loss is neutral-or-better: **4/5**.

### Coefficients driven to exactly zero: **6**

- `diff_roll30_obp`
- `diff_roll30_runs_allowed`
- `diff_roll10_win_pct`
- `diff_roll7_bullpen_fatigue`
- `diff_sp_ip_gs`
- `diff_sp_k_bb`

SP features retained (non-zero): `diff_sp_era`, `diff_sp_xfip`, `diff_sp_siera`.

### Full coefficient comparison

| feature | L2_C0.5 | elasticnet | delta |
|---|---|---|---|
| diff_roll30_opp_whip | -0.1621 | -0.1351 | 0.0270 |
| diff_sp_xfip | -0.1537 | -0.1166 | 0.0371 |
| diff_roll30_iso | 0.0855 | 0.0716 | -0.0139 |
| diff_bullpen_era | -0.0700 | -0.0652 | 0.0047 |
| diff_roll30_k_per_pa | -0.0594 | -0.0436 | 0.0157 |
| diff_roll30_opp_strikeouts | 0.0428 | 0.0383 | -0.0044 |
| diff_pyth_win_pct | 0.0423 | 0.0368 | -0.0056 |
| diff_roll10_runs_scored | 0.0334 | 0.0190 | -0.0144 |
| diff_roll30_obp | -0.0292 | 0.0000 | 0.0292 |
| diff_roll30_opp_hr_per9 | 0.0275 | 0.0132 | -0.0143 |
| diff_sp_era | 0.0234 | 0.0046 | -0.0188 |
| diff_roll30_runs_allowed | 0.0123 | 0.0000 | -0.0123 |
| diff_sp_k_bb | -0.0118 | 0.0000 | 0.0118 |
| diff_rest_days | -0.0115 | -0.0039 | 0.0075 |
| diff_roll10_win_pct | -0.0092 | 0.0000 | 0.0092 |
| diff_sp_ip_gs | 0.0072 | 0.0000 | -0.0072 |
| diff_roll7_bullpen_fatigue | -0.0038 | 0.0000 | 0.0038 |
| diff_sp_siera | 0.0034 | -0.0092 | -0.0125 |

## D — Betting impact

Edge is `model_prob - devigged_market_prob` on the picked side (`Main/app.py:1583-1590`), so a shift in the served probability moves the edge one-for-one. The served probability is the mean of LR, GB and the 50-model bootstrap-XGB mean, then blended 4% toward `_HOME_PRIOR = 0.53` (`predict_games_batch`) — GB/XGB are held fixed from the shipped pkl, so this isolates the penalty change.

Evaluated on 1353 game rows from 2026.

| metric | L2 | elasticnet |
|---|---|---|
| sd of served prob | 0.07256 | 0.07150 |
| mean |p-0.5| | 0.06048 | 0.05957 |
| p90 |p-0.5| | 0.13167 | 0.13111 |
| max |p-0.5| | 0.32160 | 0.31794 |
| mean |Δ served prob| |  | 0.00230 |
| max |Δ served prob| |  | 0.01177 |

- predicted winner flips on **19/1353** games (1.4%).
- mean shift in |p-0.5| (i.e. in edge magnitude): **-0.00090**.

**Limitation — true edges could not be computed locally.** The local DB's 2026 games stop at 2026-07-07, while the only rows carrying stored odds in `predictions_log.json` run 2026-07-16 to 2026-08-13 — zero overlap. So the count of bets crossing `GOOD_EDGE = 0.05` cannot be measured here without refreshing the DB (CLAUDE.md rule 6: the local clone drifts behind production). The probability-shift figures above bound the effect: edge moves one-for-one with the served probability, so a mean shift of 0.00090 is the scale of the change to expect.
