"""
feature_gap_vs_market.py
------------------------
The model ranks games worse than the market does (walk-forward AUC 0.5743 vs market
0.6037 at n=10,636). Recalibration cannot close that -- rescaling probabilities does
not reorder them -- so the gap is a FEATURE/information problem. This asks where
specifically.

Four questions, all on walk-forward out-of-fold probabilities joined to backfilled
odds, so nothing here is in-sample:

  1  Does the model add ANY information the market lacks? Fit
     logit(home_win) ~ market_logit + model_logit. If the model coefficient is ~0,
     every feature in FEATURE_COLS is already priced in and feature tweaks are futile.
  2  What does the market know that the model misses? Regress the MARKET's edge over
     the model (market_logit - model_logit) on the feature matrix. Features that
     predict this gap are ones the model UNDER-weights relative to the market.
  3  Per feature: incremental AUC over the market alone. This ranks features by the
     information they add to a market baseline, which is the only ranking that matters
     for beating a market -- not univariate strength, which the EDA already covered.
  4  Ceiling check: how good does a model get if it is allowed to USE the market line
     as a feature? That is the upper bound on what feature work alone can reach.

Report-only.

Usage: .venv/bin/python scripts/feature_gap_vs_market.py
"""
import os
import sqlite3
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "Main"))
DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

from calibrate_and_recompute_edge import devig  # noqa: E402
import MLBModel as M  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import statsmodels.api as sm  # noqa: E402


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    print("Building features via the canonical pipeline...", flush=True)
    df, pitcher_stats, bullpen_stats = M.load_data(DB)
    ip_lookup = M.load_boxscore_ip_lookup(DB, df)
    tgl = M.build_team_game_log(df, boxscore_ip_lookup=ip_lookup)
    tgl = M.compute_rolling_team_features(tgl)
    tgl = M.merge_sp_stats(tgl, pitcher_stats)
    tgl = M.merge_bullpen_era(tgl, bullpen_stats)
    model_df = M.assemble_features(df, tgl).dropna(subset=M.FEATURE_COLS)

    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT o.game_id, o.home_win_prob, o.home_win, o.season, s.away_ml, s.home_ml
        FROM oof_predictions o
        JOIN odds_game_link l ON l.game_id = o.game_id AND l.target='games'
        JOIN odds_snapshots s ON s.game_date_et = l.game_date_et
                             AND s.event_id = l.event_id
        WHERE o.scheme = 'walkforward' AND s.horizon_days = 0
          AND s.away_ml IS NOT NULL AND l.confidence = 'exact'
    """).fetchall()
    odds = {g: (p, w, s, a, h) for g, p, w, s, a, h in rows}

    # games.game_id is INTEGER but oof_predictions/odds_game_link store it as TEXT,
    # so the SQL join works (TEXT to TEXT) while a naive pandas isin does not.
    model_df = model_df.assign(game_id_str=model_df["game_id"].astype(str))
    sub = model_df[model_df["game_id_str"].isin(odds)].copy()
    model_p, market_p, y, season = [], [], [], []
    for gid in sub["game_id_str"]:
        p, w, s, aml, hml = odds[gid]
        _, mkt_home = devig(aml, hml)
        model_p.append(p); market_p.append(mkt_home); y.append(int(w)); season.append(s)
    model_p = np.array(model_p); market_p = np.array(market_p)
    y = np.array(y); season = np.array(season)
    X = sub[M.FEATURE_COLS].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    n = len(y)
    print(f"n = {n} priced, resolved games with a full feature vector\n")

    ml, kl = lg(model_p), lg(market_p)

    print("=" * 74)
    print("1) DOES THE MODEL ADD ANY INFORMATION THE MARKET LACKS?")
    print("=" * 74)
    r = sm.Logit(y, sm.add_constant(np.column_stack([kl, ml]))).fit(disp=0)
    for nm, c, se, pv in zip(("const", "market", "model"), r.params, r.bse, r.pvalues):
        print(f"  {nm:8s} coef={c:+.4f}  se={se:.4f}  p={pv:.4f}")
    print(f"\n  market alone AUC : {roc_auc_score(y, market_p):.4f}")
    print(f"  model  alone AUC : {roc_auc_score(y, model_p):.4f}")
    both = sm.Logit(y, sm.add_constant(np.column_stack([kl, ml]))).fit(disp=0)
    print(f"  market + model   : {roc_auc_score(y, both.predict(sm.add_constant(np.column_stack([kl, ml])))):.4f}")
    only_k = sm.Logit(y, sm.add_constant(kl)).fit(disp=0)
    print(f"  LR test, adding model to market: chi2={2*(both.llf-only_k.llf):.2f}  "
          f"p={r.pvalues[2]:.4f}")

    print("\n" + "=" * 74)
    print("2) WHAT DOES THE MARKET KNOW THAT THE MODEL MISSES?")
    print("=" * 74)
    print("   Regressing (market_logit - model_logit) on the standardized features.")
    print("   A large |coef| = the market moves on this feature and the model does not")
    print("   (or moves the wrong way) -- i.e. the model UNDER-weights it.\n")
    gap = kl - ml
    g = sm.OLS(gap, sm.add_constant(Xs)).fit()
    order = np.argsort(-np.abs(g.params[1:]))
    print(f"   R^2 = {g.rsquared:.4f}  (how much of the market-model disagreement the")
    print( "         model's OWN features explain; high = the model has the data and")
    print( "         mis-weights it, low = the market is using data the model lacks)\n")
    print(f"   {'feature':30s} {'coef':>9s} {'p':>8s}")
    for i in order[:10]:
        print(f"   {M.FEATURE_COLS[i]:30s} {g.params[i+1]:+9.4f} {g.pvalues[i+1]:8.4f}")

    print("\n" + "=" * 74)
    print("3) PER-FEATURE INCREMENTAL AUC OVER THE MARKET")
    print("=" * 74)
    print("   Baseline = market alone. Each row adds ONE feature to the market and")
    print("   re-scores. Positive = that feature carries information the market has")
    print("   not already priced. This is the only ranking that matters for beating a")
    print("   market; univariate strength (covered in eda_3/eda_4) does not.\n")
    base_auc = roc_auc_score(y, market_p)
    res = []
    for i, name in enumerate(M.FEATURE_COLS):
        Z = sm.add_constant(np.column_stack([kl, Xs[:, i]]))
        try:
            fit = sm.Logit(y, Z).fit(disp=0)
            auc = roc_auc_score(y, fit.predict(Z))
            res.append((name, auc - base_auc, fit.pvalues[2]))
        except Exception:
            continue
    res.sort(key=lambda t: -t[1])
    print(f"   {'feature':30s} {'dAUC':>9s} {'p':>8s}")
    for name, d, pv in res:
        flag = "  <-- adds beyond the market" if (d > 0.001 and pv < 0.05) else ""
        print(f"   {name:30s} {d:+9.5f} {pv:8.4f}{flag}")

    print("\n" + "=" * 74)
    print("4) CEILING CHECK — how good can this feature set get?")
    print("=" * 74)
    Zf = sm.add_constant(np.column_stack([kl, Xs]))
    full = sm.Logit(y, Zf).fit(disp=0)
    print(f"\n   market alone                  AUC {base_auc:.4f}")
    print(f"   market + ALL 18 features      AUC {roc_auc_score(y, full.predict(Zf)):.4f}")
    Zo = sm.add_constant(Xs)
    feats_only = sm.Logit(y, Zo).fit(disp=0)
    print(f"   ALL 18 features, no market    AUC {roc_auc_score(y, feats_only.predict(Zo)):.4f}")
    print(f"   walk-forward ensemble (live)  AUC {roc_auc_score(y, model_p):.4f}")
    print("\n   The gap between rows 2 and 1 is the TOTAL headroom these 18 features")
    print("   have over the market. If it is ~0, no re-weighting, no penalty change,")
    print("   and no amount of retraining on these inputs will beat the market --")
    print("   only NEW information will.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
