"""
segment_joint_edge.py
---------------------
"Which edges are actually worth betting?" -- run on BOTH edge definitions so the
answer is a comparison, not an isolated number.

Reads the per-game cache written by joint_edge_research.py (walk-forward OOF model
probabilities joined to de-vigged archived market prices), buckets and threshold-
sweeps each edge, and prices every bet at the real market moneyline so ROI is the
quantity a bettor would actually have experienced.

Bootstrap CIs are reported because this project's own power note is that detecting a
10pp win-rate gap needs ~400 bets per bucket and 5pp needs ~1,600: a bucket of 200
that looks great is noise, and saying so is the point of this script.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_ROOT, "scripts", "results")
CACHE = os.path.join(RESULTS, "_joint_edge_cache.npy")

RNG = np.random.default_rng(2026)


def american_from_prob(p):
    """De-vigged prob -> the American price a book would post (no vig added back)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.where(p >= 0.5, -100 * p / (1 - p), 100 * (1 - p) / p)


def payout(ml):
    """Profit per $1 staked on a win at American odds ml."""
    return np.where(ml > 0, ml / 100.0, 100.0 / np.abs(ml))


def boot_ci(vals, n=2000):
    if len(vals) < 5:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    means = np.asarray(vals)[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(edge, y, mkt_home, label, thresholds):
    """Bet every game whose |edge| clears the threshold, on the side the edge favours."""
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    print(f"{'thresh':>7} {'bets':>6} {'win%':>7} {'win% 95% CI':>16} "
          f"{'ROI':>8} {'ROI 95% CI':>18}")
    # price both sides off the de-vigged market prob, then add back a realistic -110
    # style hold by pricing at the market's own implied number (no vig re-added means
    # ROI here is the OPTIMISTIC, zero-vig case -- stated explicitly in the writeup)
    ml_home = american_from_prob(mkt_home)
    ml_away = american_from_prob(1 - mkt_home)
    for t in thresholds:
        sel = np.abs(edge) >= t
        if sel.sum() < 20:
            continue
        pick_home = edge[sel] > 0
        won = np.where(pick_home, y[sel] == 1, y[sel] == 0)
        odds = np.where(pick_home, ml_home[sel], ml_away[sel])
        pl = np.where(won, payout(odds), -1.0)
        wl, wh = boot_ci(won.astype(float))
        rl, rh = boot_ci(pl)
        print(f"{t:>7.3f} {sel.sum():>6} {won.mean()*100:>6.1f}% "
              f"[{wl*100:>5.1f},{wh*100:>5.1f}] {pl.mean()*100:>7.1f}% "
              f"[{rl*100:>6.1f},{rh*100:>6.1f}]")


def main():
    if not os.path.exists(CACHE):
        print("run scripts/joint_edge_research.py first")
        return 1
    arr = np.load(CACHE)
    season, y, mkt, mdl, true_p, old_edge, new_edge = (arr[:, i] for i in range(7))
    y = y.astype(int)
    print(f"games: {len(y)}   seasons: {sorted(set(season.astype(int).tolist()))}")
    print(f"home win rate: {y.mean():.4f}   market mean: {mkt.mean():.4f}")
    print("\nNOTE: bets are priced at the DE-VIGGED market probability, i.e. with the "
          "\nbookmaker's hold removed. Real ROI is therefore ~4-5pp WORSE than every "
          "\nnumber below. A strategy that is not clearly profitable here is losing "
          "\nmoney in reality.")

    evaluate(old_edge, y, mkt, "OLD EDGE  (moneyline model - market)",
             [0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15])
    evaluate(new_edge, y, mkt, "NEW EDGE  (joint ml+rl second stage - market)",
             [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04])

    # per-season robustness on the flat 'bet every positive edge' rule
    print(f"\n{'='*78}\nPER-SEASON, betting every positive edge (no threshold)\n{'='*78}")
    print(f"{'season':>7} {'n':>6} {'old win%':>10} {'new win%':>10}")
    for s in sorted(set(season.astype(int).tolist())):
        m = season.astype(int) == s
        ow = np.where(old_edge[m] > 0, y[m] == 1, y[m] == 0).mean()
        nw = np.where(new_edge[m] > 0, y[m] == 1, y[m] == 0).mean()
        print(f"{s:>7} {m.sum():>6} {ow*100:>9.1f}% {nw*100:>9.1f}%")

    # how much would the site actually flag at today's thresholds?
    print(f"\n{'='*78}\nPRODUCT IMPACT: how many games clear the CURRENT 0.05 / 0.12 bars\n{'='*78}")
    for label, e in (("old", old_edge), ("new", new_edge)):
        print(f"  {label}: |edge|>=0.05 -> {(np.abs(e)>=0.05).sum():>5} games "
              f"({(np.abs(e)>=0.05).mean()*100:.1f}%)   "
              f"|edge|>=0.12 -> {(np.abs(e)>=0.12).sum():>4} "
              f"({(np.abs(e)>=0.12).mean()*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
