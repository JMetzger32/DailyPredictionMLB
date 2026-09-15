"""
train_away_runline.py
---------------------
Fit the away-side run line model (away wins by 2+) and add it to the existing
artifacts pkl as three new keys, WITHOUT retraining anything else.

Why additive rather than a full retrain: `compute_model_version` hashes the
moneyline LR + scaler only, so leaving those untouched keeps model_version and every
served moneyline probability byte-identical. A full retrain would churn the live
slate for a change that has nothing to do with the moneyline model.

Why the model is needed at all: the existing run-line model predicts only
home_covers = (home - visitor) > 1.5. Its `away_cover_prob` is 1 - that, i.e.
P(home does NOT win by 2+), which also counts 1-run games in either direction --
a different event from "away wins by 2+", which is what the market prices when the
away team is the one laying -1.5 (~42% of games). Without this model those games
have no comparable model probability at all.

Mirrors MLBModel.py's [7e] block exactly (same hyperparameters, same YEAR_WEIGHTS,
same feature set) so the source-of-truth training run and this one agree.

Usage:
    .venv/bin/python scripts/train_away_runline.py [--dry-run]
"""
import argparse
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Main"))

DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
PKL = os.path.join(_ROOT, "updates", "mlb_model_artifacts.pkl")

import MLBModel as M  # noqa: E402

# Must match MLBModel.py's training block
YEAR_WEIGHTS = {2021: 0.3, 2022: 1.1, 2023: 1.3, 2024: 1.5, 2025: 1.8, 2026: 1.8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="train and report, do not write the pkl")
    args = ap.parse_args()

    print("Building features via the canonical pipeline...", flush=True)
    df, pitcher_stats, bullpen_stats = M.load_data(DB)
    ip_lookup = M.load_boxscore_ip_lookup(DB, df)
    tgl = M.build_team_game_log(df, boxscore_ip_lookup=ip_lookup)
    tgl = M.compute_rolling_team_features(tgl)
    tgl = M.merge_sp_stats(tgl, pitcher_stats)
    tgl = M.merge_bullpen_era(tgl, bullpen_stats)
    model_df = M.assemble_features(df, tgl)

    rla_df = model_df.dropna(subset=M.FEATURE_COLS + ["away_covers"])
    print(f"training rows: {len(rla_df)}")
    print(f"away covers rate (all): {rla_df['away_covers'].astype(int).mean():.4f}")

    # Holdout check first (2021-2024 -> 2025), same split as the home side
    tr = rla_df[rla_df["season"].between(2021, 2024)]
    te = rla_df[rla_df["season"] == 2025]
    X_tr, X_te = tr[M.FEATURE_COLS], te[M.FEATURE_COLS]
    y_tr, y_te = tr["away_covers"].astype(int), te["away_covers"].astype(int)

    s = StandardScaler()
    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=M.RANDOM_STATE)
    lr.fit(s.fit_transform(X_tr), y_tr)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=M.RANDOM_STATE)
    gb.fit(X_tr, y_tr)
    p = (lr.predict_proba(s.transform(X_te))[:, 1] + gb.predict_proba(X_te)[:, 1]) / 2
    base = max(y_te.mean(), 1 - y_te.mean())
    print(f"\n2025 holdout  n={len(te)}")
    print(f"  away covers rate : {y_te.mean():.4f}")
    print(f"  baseline (majority): {base:.4f}")
    print(f"  ensemble accuracy: {accuracy_score(y_te, (p > 0.5).astype(int)):.4f}")
    print(f"  Brier            : {brier_score_loss(y_te, p):.5f}")
    print(f"  log loss         : {log_loss(y_te, p):.5f}")

    # Final fit on all seasons with recency weights
    sw = rla_df["season"].map(YEAR_WEIGHTS).fillna(1.0)
    X_all, y_all = rla_df[M.FEATURE_COLS], rla_df["away_covers"].astype(int)
    scaler = StandardScaler()
    final_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=M.RANDOM_STATE)
    final_lr.fit(scaler.fit_transform(X_all), y_all, sample_weight=sw)
    final_gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=M.RANDOM_STATE)
    final_gb.fit(X_all, y_all, sample_weight=sw)
    print(f"\nfinal fit on {len(X_all)} rows (weighted)")

    if args.dry_run:
        print("\n--dry-run: pkl NOT written")
        return 0

    with open(PKL, "rb") as fh:
        art = pickle.load(fh)
    before_version = art.get("model_version")
    art["lr_runline_away"]     = final_lr
    art["gb_runline_away"]     = final_gb
    art["scaler_runline_away"] = scaler
    art["runline_away_trained_at"] = datetime.now(timezone.utc).isoformat()
    with open(PKL, "wb") as fh:
        pickle.dump(art, fh)
    print(f"\nwrote 3 new keys to {os.path.relpath(PKL, _ROOT)}")
    print(f"model_version unchanged: {before_version} -> {art.get('model_version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
