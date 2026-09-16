"""
home_lean_at_scale.py
---------------------
Re-test the finding in scripts/results/clv_and_home_skew.md -- "the model runs
+1.37pp more home-leaning than the market (paired t-test p=0.027)" -- at the scale
the historical backfill unlocked.

CLAUDE.md flags that result as measured at n=222 on model_version b9133b95d2ec and
"unverified against" later versions. p=0.027 at n=222 is exactly the regime where a
finding can be real or can be one lucky sample; it is also one of only two
significant results in the whole original investigation, so it is worth confirming
rather than inheriting.

This uses walk-forward out-of-fold probabilities (each season predicted by a model
trained only on PRIOR seasons) joined to backfilled odds, so the comparison is
leak-free and spans 2022-2026 instead of one month of one season.

Three questions:
  1  Does the model still run more home-leaning than the market, pooled?
  2  Is that lean stable per season, or driven by one or two years?
  3  Is the model's home lean an ACCURACY problem (is it wrong) or purely a
     CALIBRATION-scale one (right direction, overconfident magnitude)?

Report-only: changes nothing.

Usage: .venv/bin/python scripts/home_lean_at_scale.py [--scheme walkforward]
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

from calibrate_and_recompute_edge import devig  # noqa: E402


def load(conn, scheme):
    rows = conn.execute("""
        SELECT o.home_win_prob, o.home_win, o.season, s.away_ml, s.home_ml
        FROM oof_predictions o
        JOIN odds_game_link l ON l.game_id = o.game_id AND l.target='games'
        JOIN odds_snapshots s ON s.game_date_et = l.game_date_et
                             AND s.event_id = l.event_id
        WHERE o.scheme = ? AND s.horizon_days = 0
          AND s.away_ml IS NOT NULL AND l.confidence = 'exact'
    """, (scheme,)).fetchall()
    model, market, actual, season = [], [], [], []
    for p, w, s, aml, hml in rows:
        _, mkt_home = devig(aml, hml)
        model.append(p)
        market.append(mkt_home)
        actual.append(1 if w else 0)
        season.append(s)
    return (np.array(model), np.array(market), np.array(actual), np.array(season))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="walkforward")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    model, market, actual, season = load(conn, args.scheme)
    n = len(model)
    print(f"scheme: {args.scheme}")
    print(f"n = {n} priced, resolved games (vs n=222 in the 2026-08-07 analysis)\n")

    print("=" * 74)
    print("1) DOES THE MODEL RUN MORE HOME-LEANING THAN THE MARKET? (pooled)")
    print("=" * 74)
    diff = model - market
    t, p = stats.ttest_rel(model, market)
    print(f"\n  mean model home prob  : {model.mean():.4f}")
    print(f"  mean market home prob : {market.mean():.4f}")
    print(f"  mean ACTUAL home rate : {actual.mean():.4f}")
    print(f"\n  model - market        : {100*diff.mean():+.2f}pp   "
          f"paired t={t:+.2f}  p={p:.2e}")
    print(f"  model - actual        : {100*(model.mean()-actual.mean()):+.2f}pp")
    print(f"  market - actual       : {100*(market.mean()-actual.mean()):+.2f}pp")
    print(f"\n  original finding (n=222): +1.37pp, p=0.027")
    verdict = "CONFIRMED" if (p < 0.05 and diff.mean() > 0) else \
              ("REVERSED" if (p < 0.05 and diff.mean() < 0) else "NOT SIGNIFICANT")
    print(f"  -> {verdict} at n={n}")

    print("\n" + "=" * 74)
    print("2) IS THE LEAN STABLE PER SEASON, OR DRIVEN BY ONE OR TWO YEARS?")
    print("=" * 74)
    print(f"\n| season | n | model | market | actual | model-market | p |")
    print("|---|---|---|---|---|---|---|")
    signs = []
    for s in sorted(set(season.tolist())):
        m = season == s
        d = model[m] - market[m]
        tt, pp = stats.ttest_rel(model[m], market[m])
        signs.append(1 if d.mean() > 0 else -1)
        print(f"| {s} | {int(m.sum())} | {model[m].mean():.4f} | {market[m].mean():.4f} "
              f"| {actual[m].mean():.4f} | {100*d.mean():+.2f}pp | {pp:.3f} |")
    agree = sum(1 for x in signs if x > 0)
    print(f"\n  seasons with a POSITIVE (home-leaning) gap: {agree}/{len(signs)}")

    print("\n" + "=" * 74)
    print("3) IS IT AN ACCURACY PROBLEM OR A CALIBRATION-SCALE ONE?")
    print("=" * 74)
    # Brier decomposition: who is closer to truth, model or market?
    bm = np.mean((model - actual) ** 2)
    bk = np.mean((market - actual) ** 2)
    print(f"\n  Brier  model  : {bm:.4f}")
    print(f"  Brier  market : {bk:.4f}")
    print(f"  -> {'market is closer to truth' if bk < bm else 'model is closer to truth'} "
          f"(diff {bm-bk:+.4f})")

    from sklearn.metrics import roc_auc_score
    print(f"\n  AUC    model  : {roc_auc_score(actual, model):.4f}")
    print(f"  AUC    market : {roc_auc_score(actual, market):.4f}")
    print("  (AUC is scale-free: if the model's AUC matches the market's, the model")
    print("   ranks games as well and the gap is purely one of probability SCALE.)")

    # split the lean into home-side and away-side to see if it is symmetric
    home_pick = model > 0.5
    print(f"\n  games where model picks HOME : {int(home_pick.sum())} "
          f"({100*home_pick.mean():.1f}%)")
    print(f"  games where market favors HOME: {int((market>0.5).sum())} "
          f"({100*(market>0.5).mean():.1f}%)")
    print(f"  actual home win rate           : {100*actual.mean():.1f}%")

    # Score each game on the side the model actually picked: home games use the home
    # probability, away games use (1 - home probability). Indexing np.where(mask, ...)
    # by the same mask would hand the AWAY rows their home-side numbers.
    claimed_side = np.where(home_pick, model, 1 - model)
    won_side = np.where(home_pick, actual, 1 - actual)
    for label, mask in (("model picks HOME", home_pick), ("model picks AWAY", ~home_pick)):
        claimed = claimed_side[mask]
        won = won_side[mask]
        print(f"    {label:18s} n={int(mask.sum()):5d}  "
              f"claimed {100*claimed.mean():.1f}%  actual {100*won.mean():.1f}%  "
              f"overconfidence {100*(claimed.mean()-won.mean()):+.1f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
