"""
train_joint_edge_model.py
-------------------------
Fit the final second-stage "joint edge" model and store it in the artifacts pkl.

The second stage maps four logits -> P(home win):
    [logit(model_ml), logit(market_ml), logit(model_rl), logit(market_rl)]

It is trained on WALK-FORWARD OOF model probabilities, not on the shipped artifact's
own in-sample outputs. That matters: at serve time the first-stage models see a game
they were not trained on, so the second stage has to learn the relationship as it
looks out-of-sample. Training it on in-sample first-stage probabilities would teach it
to trust the model far more than it deserves.

Evidence and caveats: scripts/results/joint_edge_research.md. Headline -- this metric
is NOT demonstrably profitable after vig (best cell 51.8% vs 52.4% breakeven); what it
fixes is that the shipped moneyline-only edge is actively anti-predictive.

Usage:
    .venv/bin/python scripts/train_joint_edge_model.py [--dry-run]
"""
import argparse
import os
import pickle
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
PKL = os.path.join(_ROOT, "updates", "mlb_model_artifacts.pkl")

sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from joint_edge_research import load, lg, auc   # noqa: E402

# Thresholds calibrated on the new metric's scale in joint_edge_research.md sec. 6.
# The shipped 0.05/0.12 are meaningless here -- the joint edge is ~6x smaller.
GOOD_EDGE_JOINT = 0.010
EXTREME_EDGE_JOINT = 0.020


def build_matrix(rows):
    mdl_ml = np.array([r[2] for r in rows], dtype=float)
    home_win = np.array([r[3] for r in rows], dtype=int)
    mkt_ml = np.array([r[7] for r in rows], dtype=float)
    p_home_by2 = np.array([r[8] for r in rows], dtype=float)
    p_away_by2 = np.array([r[9] for r in rows], dtype=float)
    home_point = np.array([r[12] for r in rows], dtype=float)
    mkt_rl = np.array([r[14] for r in rows], dtype=float)
    mdl_rl = np.where(home_point == -1.5, p_home_by2, 1.0 - p_away_by2)
    X = np.column_stack([lg(mdl_ml), lg(mkt_ml), lg(mdl_rl), lg(mkt_rl)])
    return X, home_win, mkt_ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load("walkforward")
    X, y, mkt_ml = build_matrix(rows)
    print(f"training rows (walk-forward OOF, priced both markets): {len(y)}")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    names = ["model_ml", "market_ml", "model_rl", "market_rl"]
    print("\ncoefficients (on logits):")
    for n, c in zip(names, lr.coef_[0]):
        print(f"  {n:<11} {c:+.4f}")
    print(f"  {'intercept':<11} {lr.intercept_[0]:+.4f}")
    print(f"\nin-sample AUC {auc(y, p):.4f}   market AUC {auc(y, mkt_ml):.4f}")
    edge = p - mkt_ml
    print(f"edge: mean {edge.mean():+.4f}  mean|edge| {np.abs(edge).mean():.4f}  "
          f"sd {edge.std():.4f}")
    for t in (GOOD_EDGE_JOINT, EXTREME_EDGE_JOINT):
        print(f"  |edge| >= {t:.3f}: {(np.abs(edge) >= t).sum()} "
              f"({(np.abs(edge) >= t).mean()*100:.1f}%)")

    if args.dry_run:
        print("\n--dry-run: pkl NOT written")
        return 0

    with open(PKL, "rb") as fh:
        art = pickle.load(fh)
    before = art.get("model_version")
    art["joint_edge_model"] = lr
    art["joint_edge_feature_order"] = names
    art["joint_edge_thresholds"] = {"good": GOOD_EDGE_JOINT,
                                    "extreme": EXTREME_EDGE_JOINT}
    art["joint_edge_trained_at"] = datetime.now(timezone.utc).isoformat()
    art["joint_edge_n_train"] = int(len(y))
    with open(PKL, "wb") as fh:
        pickle.dump(art, fh)
    print(f"\nwrote joint_edge_model to {os.path.relpath(PKL, _ROOT)}")
    print(f"model_version unchanged: {before} -> {art.get('model_version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
