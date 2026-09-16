"""
build_runline_oof.py
--------------------
Out-of-sample run-line probabilities for BOTH sides, for 2021-2026.

Companion to build_oof_predictions.py (moneyline), deliberately a separate script and
table: the moneyline OOF costs 50 bootstrap XGBs per fold, and nothing about adding
run-line numbers should force that to be recomputed. Reuses that script's fold
definitions verbatim (imported, not copied) so the two OOF sets are directly joinable
fold-for-fold.

Two targets, because they are NOT complements:
  home_covers  (home - visitor) > 1.5   -- prices the home team laying -1.5
  away_covers  (visitor - home) > 1.5   -- prices the away team laying -1.5
"1 - home_covers" also counts one-run games in either direction, so it cannot stand in
for away_covers. The market quotes whichever side is the favourite, so both are needed
to cover the whole slate.

Scoring the shipped artifact on its own training seasons would be in-sample and the
resulting edge meaningless -- same reasoning as build_oof_predictions.py's docstring.

Usage:
    .venv/bin/python scripts/build_runline_oof.py [--schemes walkforward,loso] [--dry-run]
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Main"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

import MLBModel as M                                    # noqa: E402
from build_oof_predictions import folds_for             # noqa: E402

# Must match MLBModel.py's [7d]/[7e] blocks exactly.
YEAR_WEIGHTS = {2021: 0.3, 2022: 1.1, 2023: 1.3, 2024: 1.5, 2025: 1.8, 2026: 1.8}

DDL = """
CREATE TABLE IF NOT EXISTS oof_runline_predictions (
    game_id         TEXT    NOT NULL,
    season          INTEGER NOT NULL,
    scheme          TEXT    NOT NULL,   -- 'walkforward' | 'loso'
    fold_season     INTEGER NOT NULL,
    home_cover_prob REAL    NOT NULL,   -- P(home wins by 2+)
    away_by2_prob   REAL    NOT NULL,   -- P(away wins by 2+)
    home_covers     INTEGER,            -- actual
    away_covers     INTEGER,            -- actual
    n_train         INTEGER,
    built_at        TEXT    NOT NULL,
    PRIMARY KEY (game_id, scheme)
)
"""


def fit_side(X_tr, y_tr, sw, X_va):
    """One run-line side, matching MLBModel's hyperparameters exactly."""
    scaler = StandardScaler()
    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=M.RANDOM_STATE)
    lr.fit(scaler.fit_transform(X_tr), y_tr, sample_weight=sw)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=M.RANDOM_STATE)
    gb.fit(X_tr, y_tr, sample_weight=sw)
    return (lr.predict_proba(scaler.transform(X_va))[:, 1]
            + gb.predict_proba(X_va)[:, 1]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="walkforward,loso")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("Building features via the canonical MLBModel pipeline...", flush=True)
    df, pitcher_stats, bullpen_stats = M.load_data(DB)
    ip_lookup = M.load_boxscore_ip_lookup(DB, df)
    tgl = M.build_team_game_log(df, boxscore_ip_lookup=ip_lookup)
    tgl = M.compute_rolling_team_features(tgl)
    tgl = M.merge_sp_stats(tgl, pitcher_stats)
    tgl = M.merge_bullpen_era(tgl, bullpen_stats)
    model_df = M.assemble_features(df, tgl)

    data = model_df.dropna(subset=M.FEATURE_COLS + ["home_covers", "away_covers"]).copy()
    seasons = sorted(int(s) for s in data["season"].unique() if 2021 <= s <= 2026)
    print(f"  usable rows: {len(data)}  seasons: {seasons}  ({time.time()-t0:.0f}s)")

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    if args.dry_run:
        for scheme in schemes:
            for hold, train in folds_for(scheme, seasons):
                print(f"  [{scheme}] hold {hold} <- train {train} "
                      f"({int(data['season'].isin(train).sum())} rows)")
        return 0

    conn = sqlite3.connect(DB)
    conn.execute(DDL)
    conn.commit()

    rows_out = []
    for scheme in schemes:
        for hold, train in folds_for(scheme, seasons):
            tr = data[data["season"].isin(train)]
            va = data[data["season"] == hold]
            if tr.empty or va.empty:
                continue
            assert not set(tr["game_id"]) & set(va["game_id"]), \
                f"LEAK: overlapping game_id in {scheme} fold {hold}"
            X_tr, X_va = tr[M.FEATURE_COLS], va[M.FEATURE_COLS]
            sw = tr["season"].map(YEAR_WEIGHTS).fillna(1.0)
            p_home = fit_side(X_tr, tr["home_covers"].astype(int), sw, X_va)
            p_away = fit_side(X_tr, tr["away_covers"].astype(int), sw, X_va)
            built = datetime.now(timezone.utc).isoformat()
            for gid, season, ph, pa, hc, ac in zip(
                    va["game_id"], va["season"], p_home, p_away,
                    va["home_covers"], va["away_covers"]):
                rows_out.append((str(gid), int(season), scheme, int(hold),
                                 float(ph), float(pa), int(hc), int(ac),
                                 len(tr), built))
            print(f"  [{scheme}] {hold}: {len(va)} games from {len(tr)} train rows",
                  flush=True)

    conn.executemany(
        "INSERT OR REPLACE INTO oof_runline_predictions "
        "(game_id, season, scheme, fold_season, home_cover_prob, away_by2_prob, "
        "home_covers, away_covers, n_train, built_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows_out)
    conn.commit()
    print(f"\nwrote {len(rows_out)} rows")

    for scheme, n, mph, mpa, ahc, aac in conn.execute(
            "SELECT scheme, COUNT(*), ROUND(AVG(home_cover_prob),4), "
            "ROUND(AVG(away_by2_prob),4), ROUND(AVG(home_covers),4), "
            "ROUND(AVG(away_covers),4) FROM oof_runline_predictions GROUP BY scheme"):
        print(f"  {scheme:<12} n={n:<6} mean P(home by2)={mph} (actual {ahc})  "
              f"mean P(away by2)={mpa} (actual {aac})")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
