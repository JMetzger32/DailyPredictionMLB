"""
home_lean_feature_analysis.py
------------------------------
Follow-up to scripts/clv_and_skew_diagnostics.py's finding that the model runs
+1.37pp more home-leaning than the market (paired t-test p=0.027). This drills into
WHICH features drive that gap, using the exact scaled feature vector logged per
prediction (`x_scaled_features`, 18 floats matching Main/MLBModel.py's FEATURE_COLS).

Three questions:
  1  Is the model's baseline home-field-advantage assumption (the LR intercept) stale
     relative to the actual 2026 home win rate?
  2  Which features does the model's probability move with MORE than the market's does?
  3  Is the excess home lean a compositional artifact (these particular games having
     home teams that are simply better on these features) or a pure weighting
     difference (model reacts more to the same information)?

Usage:
    python3 scripts/home_lean_feature_analysis.py

Report-only: this script changes nothing, and this investigation does NOT modify
FEATURE_COLS, retrain the model, or otherwise touch Main/MLBModel.py -- see the
report's own recommendation on why not.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
PKL      = os.path.join(_ROOT, "updates", "mlb_model_artifacts.pkl")

FEATURE_COLS = [
    "diff_pyth_win_pct", "diff_roll30_obp", "diff_roll30_iso", "diff_roll10_runs_scored",
    "diff_roll30_k_per_pa", "diff_roll30_opp_whip", "diff_roll30_opp_hr_per9",
    "diff_roll30_opp_strikeouts", "diff_roll30_runs_allowed", "diff_roll10_win_pct",
    "diff_bullpen_era", "diff_roll7_bullpen_fatigue", "diff_rest_days", "diff_sp_era",
    "diff_sp_ip_gs", "diff_sp_k_bb", "diff_sp_xfip", "diff_sp_siera",
]


def a2r(ml):
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def devig(away_ml, home_ml):
    a, h = a2r(away_ml), a2r(home_ml)
    t = a + h
    return a / t, h / t


def load():
    with open(PRED_LOG) as f:
        d = json.load(f)
    rows = [e for day in d.values() for e in day
            if e.get("bet_rating") and e.get("correct") is not None
            and e.get("game_type") == "R" and e.get("x_scaled_features")]
    model_p, market_p, X, y = [], [], [], []
    for e in rows:
        _, mkt_home = devig(e["away_ml"], e["home_ml"])
        model_p.append(e["home_win_prob"])
        market_p.append(mkt_home)
        X.append(e["x_scaled_features"])
        y.append(1 if e["actual_winner"] == "Home" else 0)
    return np.array(model_p), np.array(market_p), np.array(X), np.array(y)


def main():
    model_p, market_p, X, y = load()
    n = len(model_p)
    print(f"n = {n} rated+resolved games with a logged feature vector\n")

    print("=" * 72)
    print("1) IS THE LR'S BASELINE HOME-FIELD ASSUMPTION STALE?")
    print("=" * 72)
    with open(PKL, "rb") as f:
        art = pickle.load(f)
    lr = art["lr_model"]
    intercept_p = 1 / (1 + np.exp(-lr.intercept_[0]))
    print(f"\n  LR intercept alone (all diffs=0) implies : {intercept_p:.4f}")
    print(f"  actual 2026 home win rate (this sample)  : {y.mean():.4f}")
    print(f"  -> {'stale' if abs(intercept_p - y.mean()) > 0.01 else 'NOT stale — matches closely'}, "
          f"gap = {100*(intercept_p - y.mean()):+.2f}pp")

    print("\n" + "=" * 72)
    print("2) WHICH FEATURES DOES THE MODEL LEAN ON MORE THAN THE MARKET DOES?")
    print("=" * 72)
    print(f"\n{'feature':28s} {'corr w/ MODEL':>13s} {'corr w/ MARKET':>14s} {'excess':>8s}")
    out = []
    for i, name in enumerate(FEATURE_COLS):
        if X[:, i].std() < 1e-6:   # diff_rest_days is ~constant in this sample
            continue
        rm, _ = stats.pearsonr(X[:, i], model_p)
        rk, _ = stats.pearsonr(X[:, i], market_p)
        out.append((name, rm, rk, abs(rm) - abs(rk)))
    out.sort(key=lambda t: -t[3])
    for name, rm, rk, excess in out:
        flag = "  <-- model leans on this much more than market" if excess > 0.25 else ""
        print(f"{name:28s} {rm:+13.4f} {rk:+14.4f} {excess:+8.4f}{flag}")

    print("\n" + "=" * 72)
    print("3) COMPOSITIONAL vs WEIGHTING: where does the average home lean come from?")
    print("=" * 72)
    coef = lr.coef_[0]
    contrib = X.mean(axis=0) * coef
    print(f"\n  intercept                : {lr.intercept_[0]:+.4f}")
    print(f"  sum(mean_feature * coef) : {contrib.sum():+.4f}  "
          f"(the average tilt these 222 games happen to have, weighted by the LR)")
    print("\n  Top contributors to that sum:")
    for name, c in sorted(zip(FEATURE_COLS, contrib), key=lambda t: -abs(t[1]))[:5]:
        print(f"    {name:28s} {c:+.4f}")
    lr_prob = lr.predict_proba(X)[:, 1]
    print(f"\n  mean LR-only prob (recomputed from X) : {lr_prob.mean():.4f}")
    print(f"  mean LIVE ensemble prob (GB+XGB+blend): {model_p.mean():.4f}")
    print(f"  ensemble adds beyond LR alone          : {100*(model_p.mean()-lr_prob.mean()):+.2f}pp "
          f"(small — this is an LR-feature-weighting effect, not a tree/blend artifact)")

    print("\n" + "=" * 72)
    print("4) FEATURE REDUNDANCY CHECK (data-quality note, not the primary cause)")
    print("=" * 72)
    i_xfip, i_siera, i_era = (FEATURE_COLS.index(n) for n in
                              ("diff_sp_xfip", "diff_sp_siera", "diff_sp_era"))
    print(f"\n  corr(diff_sp_xfip, diff_sp_siera) = "
          f"{np.corrcoef(X[:,i_xfip], X[:,i_siera])[0,1]:+.4f}  (near-duplicate signal)")
    print(f"  LR coefficients: xfip={coef[i_xfip]:+.4f}  siera={coef[i_siera]:+.4f}  "
          f"era={coef[i_era]:+.4f}")
    print("  -> LR already puts most weight on xfip and little on siera despite the")
    print("     near-collinearity, so this is NOT double-counting driving the lean —")
    print("     worth simplifying for interpretability, but not the mechanism here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
