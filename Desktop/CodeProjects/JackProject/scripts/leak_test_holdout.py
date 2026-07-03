#!/usr/bin/env python3
"""
Phase 2b: leak-test on the 2025 holdout.

Builds the model_df once via the MLBModel pipeline, then trains on 2021-2024 and
evaluates on 2025 with TWO feature sets (global FEATURE_COLS is never mutated):

  FULL    = current FEATURE_COLS
  REDUCED = FULL minus all season-long SP + bullpen-ERA features (the leaky ones):
            diff_sp_era, diff_sp_whip, diff_sp_xfip, diff_sp_siera, diff_sp_so9,
            diff_sp_bb9, diff_sp_hr9, diff_sp_ip_gs, diff_sp_k_bb, diff_bullpen_era
            (intersected with FEATURE_COLS)

Reports AUC / LogLoss / Brier for LR, GB, and the LR+GB ensemble (the ensemble the
production retrain ships), so the delta from dropping the leaky features is visible.
No 1.4x boost is applied (that lives only in MLBModel.__main__), so this is clean.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Main"))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from MLBModel import (
    load_data, build_team_game_log, compute_rolling_team_features,
    merge_bullpen_era, merge_sp_stats, assemble_features, FEATURE_COLS, DB_PATH,
    RANDOM_STATE,
)

LEAKY = ["diff_sp_era", "diff_sp_whip", "diff_sp_xfip", "diff_sp_siera", "diff_sp_so9",
         "diff_sp_bb9", "diff_sp_hr9", "diff_sp_ip_gs", "diff_sp_k_bb", "diff_bullpen_era"]


def evaluate(model_df, feature_cols, label):
    train = model_df[model_df["season"].between(2021, 2024)].dropna(subset=feature_cols)
    test = model_df[model_df["season"] == 2025].dropna(subset=feature_cols)
    Xtr, ytr = train[feature_cols], train["home_win"]
    Xte, yte = test[feature_cols], test["home_win"]

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)

    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE).fit(Xtr_sc, ytr)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=RANDOM_STATE).fit(Xtr, ytr)

    lr_p = lr.predict_proba(Xte_sc)[:, 1]
    gb_p = gb.predict_proba(Xte)[:, 1]
    ens_p = (lr_p + gb_p) / 2.0

    print(f"\n=== {label}  ({len(feature_cols)} features, "
          f"train={len(Xtr)}, test2025={len(Xte)}) ===")
    print(f"{'model':>12} | {'AUC':>7} | {'LogLoss':>8} | {'Brier':>7}")
    print("-" * 44)
    rows = {}
    for name, p in [("LR", lr_p), ("GB", gb_p), ("LR+GB ens", ens_p)]:
        auc = roc_auc_score(yte, p)
        ll = log_loss(yte, p)
        br = brier_score_loss(yte, p)
        rows[name] = (auc, ll, br)
        print(f"{name:>12} | {auc:>7.4f} | {ll:>8.4f} | {br:>7.4f}")
    return rows


def main():
    print("Loading + building features (2021-2025 pipeline)...")
    df, pitcher_stats, bullpen_stats = load_data(DB_PATH)
    tgl = build_team_game_log(df)
    tgl = compute_rolling_team_features(tgl)
    tgl = merge_bullpen_era(tgl, bullpen_stats)
    tgl = merge_sp_stats(tgl, pitcher_stats)
    model_df = assemble_features(df, tgl)

    full = list(FEATURE_COLS)
    reduced = [c for c in full if c not in LEAKY]
    dropped = [c for c in full if c in LEAKY]
    print(f"Dropped {len(dropped)} leaky feature(s): {dropped}")

    full_rows = evaluate(model_df, full, "FULL (current)")
    red_rows = evaluate(model_df, reduced, "REDUCED (SP+bullpen dropped)")

    print("\n=== DELTA (reduced - full) ===")
    print(f"{'model':>12} | {'dAUC':>8} | {'dLogLoss':>9} | {'dBrier':>8}")
    print("-" * 46)
    for name in full_rows:
        fa, fl, fb = full_rows[name]
        ra, rl, rb = red_rows[name]
        print(f"{name:>12} | {ra-fa:>+8.4f} | {rl-fl:>+9.4f} | {rb-fb:>+8.4f}")
    ens_dauc = red_rows["LR+GB ens"][0] - full_rows["LR+GB ens"][0]
    print(f"\nGate: Phase 4 (leak fix) triggers if FULL AUC drop > 0.01 when features removed.")
    print(f"Ensemble AUC change (reduced - full): {ens_dauc:+.4f}  "
          f"=> AUC drop = {-ens_dauc:.4f}  "
          f"({'>' if -ens_dauc > 0.01 else '<='} 0.01)")


if __name__ == "__main__":
    main()
