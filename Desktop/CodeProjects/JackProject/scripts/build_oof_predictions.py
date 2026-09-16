"""
build_oof_predictions.py
------------------------
Generate OUT-OF-SAMPLE per-game win probabilities for 2021-2026, so historical odds
can be turned into an honest edge calibration.

Why this exists: the shipped artifact is trained on 2021-2026. Scoring those seasons
with it is in-sample, and the resulting "edge" is meaningless -- it measures memorized
outcomes, not forecasting skill. Two honest schemes are emitted:

  walkforward  (HEADLINE)  season S trained ONLY on seasons < S. The single scheme
               that matches how a bettor actually experiences a market: no future
               information, ever. 2021 gets no fold (nothing prior to train on).

  loso         (UPPER BOUND) season S trained on all OTHER seasons. Covers 2021 and
               gives each fold more data, but a 2021 fold trained on 2022-2025 knows
               future talent and market regimes, so it is optimistic BY CONSTRUCTION.
               Report it as a robustness check, never as the decision input. Where the
               two disagree on threshold ranking, walk-forward wins.

Reproduces the shipped ensemble per fold: LR (elastic net, scaler fit on the
recent-seasons window WITHIN the training fold) + GB + N bootstrap XGBs, averaged,
then blended toward the home prior via MLBModel.apply_home_prior_blend -- an offline
replay that skips that blend is not measuring the model the product ships.

GB hyperparameters are HARDCODED to the shipped values rather than re-grid-searched
per fold: MLBModel.__main__ tunes them against the 2025 holdout, so re-running that
search inside each fold would leak 2025 into every one of them.

Usage:
    .venv/bin/python scripts/build_oof_predictions.py                  # both schemes
    .venv/bin/python scripts/build_oof_predictions.py --quick          # 5 bootstraps
    .venv/bin/python scripts/build_oof_predictions.py --schemes walkforward
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Main"))
sys.path.insert(0, os.path.join(_ROOT, "updates"))

DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

import MLBModel as M  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Shipped GB config (MLBModel.cross_validate_loso / __main__). Hardcoded on purpose.
GB_KWARGS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, random_state=M.RANDOM_STATE)

DDL = """
CREATE TABLE IF NOT EXISTS oof_predictions (
    game_id       TEXT    NOT NULL,
    season        INTEGER NOT NULL,
    scheme        TEXT    NOT NULL,     -- 'walkforward' | 'loso'
    fold_season   INTEGER NOT NULL,     -- the held-out season this row was scored in
    home_win_prob REAL    NOT NULL,     -- AFTER the home-prior blend
    home_win      INTEGER,              -- actual outcome, for convenience
    n_train       INTEGER,
    n_bootstrap   INTEGER,
    built_at      TEXT    NOT NULL,
    PRIMARY KEY (game_id, scheme)
)"""


def fit_fold(X_tr, y_tr, seasons_tr, X_va, n_bootstrap, seed=M.RANDOM_STATE):
    """Fit the shipped ensemble on one fold, return blended probabilities for X_va."""
    parts = []

    # --- LR: scaler fit on the recent-seasons window WITHIN this fold's training
    # rows (mirrors cross_validate_loso, including its fallback when the window is
    # empty -- which genuinely fires for walk-forward 2022/2023, since the window
    # starts at 2023).
    window = seasons_tr >= M.SCALER_WINDOW_START_SEASON
    X_scaler_fit = X_tr[window.values] if window.any() else X_tr
    scaler = StandardScaler().fit(X_scaler_fit)
    sw = seasons_tr.map(M_YEAR_WEIGHTS).fillna(1.0).values
    lr = LogisticRegression(random_state=seed, **M.LR_PENALTY_KWARGS)
    lr.fit(scaler.transform(X_tr), y_tr, sample_weight=sw)
    parts.append(lr.predict_proba(scaler.transform(X_va))[:, 1])

    # --- GB (raw features)
    gb = GradientBoostingClassifier(**GB_KWARGS)
    gb.fit(X_tr, y_tr, sample_weight=sw)
    parts.append(gb.predict_proba(X_va)[:, 1])

    # --- bootstrap XGBs (raw features), averaged into ONE ensemble member, matching
    # predict_games_batch. Dropping these shifts the probability distribution, which
    # is the exact quantity being calibrated -- so they are not optional.
    try:
        from xgboost import XGBClassifier
        boot = []
        rng = np.random.RandomState(seed)
        n = len(X_tr)
        for b in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=seed + b, eval_metric="logloss",
                                verbosity=0)
            xgb.fit(X_tr.iloc[idx], y_tr.iloc[idx], sample_weight=sw[idx])
            boot.append(xgb.predict_proba(X_va)[:, 1])
        if boot:
            parts.append(np.mean(np.vstack(boot), axis=0))
    except Exception as e:
        print(f"    [warn] xgboost unavailable/failed ({e}) — LR+GB only", flush=True)

    return M.apply_home_prior_blend(np.mean(np.vstack(parts), axis=0))


M_YEAR_WEIGHTS = {2021: 0.3, 2022: 1.1, 2023: 1.3, 2024: 1.5, 2025: 1.8, 2026: 1.8}


def folds_for(scheme, seasons):
    """Yield (held_out_season, [training seasons])."""
    for s in seasons:
        train = [t for t in seasons if t < s] if scheme == "walkforward" \
            else [t for t in seasons if t != s]
        if not train:
            continue                      # walk-forward 2021 has no prior data
        yield s, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="walkforward,loso")
    ap.add_argument("--n-bootstrap", type=int, default=50,
                    help="bootstrap XGBs per fold (production uses 50)")
    ap.add_argument("--quick", action="store_true", help="5 bootstraps, for iteration")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    n_boot = 5 if args.quick else args.n_bootstrap

    t0 = time.time()
    print("Building features via the canonical MLBModel pipeline...", flush=True)
    df, pitcher_stats, bullpen_stats = M.load_data(DB)
    ip_lookup = M.load_boxscore_ip_lookup(DB, df)
    tgl = M.build_team_game_log(df, boxscore_ip_lookup=ip_lookup)
    tgl = M.compute_rolling_team_features(tgl)
    tgl = M.merge_sp_stats(tgl, pitcher_stats)
    tgl = M.merge_bullpen_era(tgl, bullpen_stats)
    model_df = M.assemble_features(df, tgl)
    print(f"  model_df: {len(model_df)} rows, {time.time()-t0:.0f}s", flush=True)

    need = set(M.FEATURE_COLS) | {"season", "home_win", "game_id"}
    missing = need - set(model_df.columns)
    assert not missing, f"model_df missing columns: {missing}"

    data = model_df.dropna(subset=M.FEATURE_COLS).copy()
    seasons = sorted(int(s) for s in data["season"].unique() if 2021 <= s <= 2026)
    print(f"  usable rows: {len(data)}  seasons: {seasons}", flush=True)
    for s in seasons:
        print(f"    {s}: {int((data['season']==s).sum())} games")

    if args.dry_run:
        for scheme in args.schemes.split(","):
            for hold, train in folds_for(scheme.strip(), seasons):
                n = int(data["season"].isin(train).sum())
                print(f"  [{scheme}] hold {hold} <- train {train} ({n} rows)")
        return 0

    conn = sqlite3.connect(DB)
    conn.execute(DDL)
    conn.commit()
    rows_out = []

    for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
        for hold, train in folds_for(scheme, seasons):
            tr = data[data["season"].isin(train)]
            va = data[data["season"] == hold]
            if va.empty:
                continue
            ts = time.time()
            probs = fit_fold(tr[M.FEATURE_COLS], tr["home_win"], tr["season"],
                             va[M.FEATURE_COLS], n_boot)
            # leak guard: a game must never appear in both sides of its own fold
            assert not (set(tr["game_id"]) & set(va["game_id"])), \
                f"LEAK: overlapping game_id in {scheme} fold {hold}"
            for gid, season, p, y in zip(va["game_id"], va["season"], probs, va["home_win"]):
                rows_out.append((str(gid), int(season), scheme, int(hold), float(p),
                                 int(y), len(tr), n_boot,
                                 datetime.now(timezone.utc).isoformat()))
            print(f"  [{scheme}] {hold}: {len(va)} games from {len(tr)} train rows "
                  f"(mean p={probs.mean():.4f}) {time.time()-ts:.0f}s", flush=True)

    conn.executemany(
        "INSERT OR REPLACE INTO oof_predictions "
        "(game_id, season, scheme, fold_season, home_win_prob, home_win, n_train, "
        " n_bootstrap, built_at) VALUES (?,?,?,?,?,?,?,?,?)", rows_out)
    conn.commit()
    summary = conn.execute(
        "SELECT scheme, COUNT(*), ROUND(AVG(home_win_prob),4), "
        "       ROUND(AVG(CASE WHEN (home_win_prob>0.5)=(home_win=1) THEN 1.0 ELSE 0 END),4) "
        "FROM oof_predictions GROUP BY scheme").fetchall()
    conn.close()
    print("\nscheme            n   mean_p   accuracy")
    for s, n, mp, acc in summary:
        print(f"  {s:<14} {n:>6}  {mp:.4f}   {acc:.4f}")
    print(f"\ntotal {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
