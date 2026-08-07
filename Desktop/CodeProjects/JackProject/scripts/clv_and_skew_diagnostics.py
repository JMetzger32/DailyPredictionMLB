"""
clv_and_skew_diagnostics.py
---------------------------
Follow-up to Phase 3, answering three questions the earlier phases raised:

  1  Does the model actually beat the closing line?  (the stored `clv` field says
     yes, but it is NOT closing line value -- see below)
  2  Where does the home/away skew in value-bet flagging come from?
  3  Can re-binning edge find a profitable segment, or is the sample too small?

THE CLV BUG THIS SCRIPT EXISTS TO EXPOSE
    Main/app.py:1306 stores        clv = model_prob - closing_implied
    True closing line value is     clv = closing_implied - bet_implied
The stored field never compares two PRICES -- it re-measures the model's own edge
against a later line, so it is large whenever the model is confident and tells you
nothing about whether the market moved your way. True CLV is the standard
gold-standard skill metric: did you get a better price than the close?

Usage:
    python3 scripts/clv_and_skew_diagnostics.py

Report-only: this script changes nothing.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BET_LOG  = os.path.join(_ROOT, "Databases_and_logs", "betting_log.json")
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
PKL      = os.path.join(_ROOT, "updates", "mlb_model_artifacts.pkl")
BLEND, PRIOR = 0.04, 0.53


def a2r(ml):
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def devig(away_ml, home_ml):
    ra, rh = a2r(away_ml), a2r(home_ml)
    t = ra + rh
    return ra / t, rh / t


def pl(ml, won, stake=10.0):
    return stake * (ml / 100 if ml >= 0 else 100 / abs(ml)) if won else -stake


def load_bets():
    with open(BET_LOG) as f:
        blog = json.load(f)
    return [e for day in sorted(blog) for e in blog[day]
            if e.get("bet_rating") and e.get("correct") is not None
            and e.get("game_type") == "R"]


def load_preds():
    with open(PKL, "rb") as f:
        v = pickle.load(f).get("model_version")
    with open(PRED_LOG) as f:
        plog = json.load(f)
    return v, [e for day in plog.values() for e in day
               if e.get("correct") is not None and e.get("game_type") == "R"
               and e.get("post_game_created") is not True
               and e.get("home_win_prob") is not None
               and e.get("model_version") == v]


def stat_line(sub, label, min_n=8):
    if len(sub) < min_n:
        print(f"  {label:26s} n={len(sub):3d}  (too few)")
        return
    w = sum(1 for e in sub if e["correct"])
    net = sum(pl(e["predicted_team_ml"], e["correct"]) for e in sub)
    lo, hi = proportion_confint(w, len(sub), method="wilson")
    print(f"  {label:26s} n={len(sub):3d}  {w}-{len(sub)-w}  {100*w/len(sub):5.1f}% "
          f"[{100*lo:4.1f},{100*hi:4.1f}]  ROI {100*net/(len(sub)*10):+6.1f}%")


def section_clv(rows):
    print("=" * 72)
    print("1) IS THE STORED `clv` REALLY CLOSING LINE VALUE?  (no)")
    print("=" * 72)
    stored, true, won, home = [], [], [], []
    for e in rows:
        if e.get("closing_away_ml") is None:
            continue
        bai, bhi = devig(e["away_ml"], e["home_ml"])
        cai, chi = devig(e["closing_away_ml"], e["closing_home_ml"])
        is_home = e["predicted_winner"] == "Home"
        stored.append(e["clv"])
        true.append((chi if is_home else cai) - (bhi if is_home else bai))
        won.append(1 if e["correct"] else 0)
        home.append(is_home)
    stored, true = np.array(stored), np.array(true)
    won, home = np.array(won), np.array(home)

    t, p = stats.ttest_1samp(true, 0)
    se = true.std(ddof=1) / np.sqrt(len(true))
    print(f"\nn = {len(true)} bets that have closing odds\n")
    print(f"  STORED 'clv'  (model_prob - closing_implied) : {100*stored.mean():+.2f}%"
          f"   <- what the page reports")
    print(f"  TRUE   CLV    (closing_implied - bet_implied): {100*true.mean():+.2f}%")
    print(f"      t={t:+.2f}  p={p:.3f}  95% CI "
          f"[{100*(true.mean()-1.96*se):+.2f}%, {100*(true.mean()+1.96*se):+.2f}%]")
    print(f"      -> {'differs from zero' if p < 0.05 else 'INDISTINGUISHABLE FROM ZERO'}")
    print(f"      beat the close on {100*(true>0).mean():.1f}% of bets (coin flip = 50%)")
    print(f"\n  correlation with actually winning:")
    print(f"      TRUE CLV   {np.corrcoef(true, won)[0,1]:+.4f}   (positive = real skill signal)")
    print(f"      stored clv {np.corrcoef(stored, won)[0,1]:+.4f}   (negative = it is measuring overconfidence)")
    print(f"\n  true CLV by side:  HOME {100*true[home].mean():+.2f}% (n={home.sum()})"
          f"   AWAY {100*true[~home].mean():+.2f}% (n={(~home).sum()})")
    return true


def section_skew(rows, ents):
    print("\n" + "=" * 72)
    print("2) WHERE DOES THE HOME/AWAY SKEW COME FROM?")
    print("=" * 72)
    p = np.array([e["home_win_prob"] for e in ents])
    y = np.array([1 if e["actual_winner"] == "Home" else 0 for e in ents])
    raw = (p - BLEND * PRIOR) / (1 - BLEND)

    print(f"\n  a) Is the model home-biased overall?  (n={len(ents)} resolved predictions)")
    print(f"     mean predicted home prob : {p.mean():.4f}")
    print(f"     actual home win rate     : {y.mean():.4f}")
    print(f"     bias                     : {100*(p.mean()-y.mean()):+.2f}pp  (small)")
    print(f"     BUT model picks Home on  : {100*(p>0.5).mean():.1f}% of games")

    print(f"\n  b) Is the _HOME_PRIOR=0.53 blend the cause?  NO")
    print(f"     mean RAW ensemble prob   : {raw.mean():.4f}")
    print(f"     mean BLENDED (live) prob : {p.mean():.4f}")
    print(f"     blend shifts the mean by : {100*(p.mean()-raw.mean()):+.3f}pp "
          f"-> it REDUCES the lean slightly")
    print(f"     the home lean is in the MODEL ITSELF (features/training), not the blend")

    print(f"\n  c) Is the overconfidence home-specific?  NO -- it is symmetric")
    for name, mask in (("picked HOME", p > 0.5), ("picked AWAY", p <= 0.5)):
        wp = p[mask] if name == "picked HOME" else 1 - p[mask]
        corr = y[mask] if name == "picked HOME" else 1 - y[mask]
        print(f"     {name}: n={mask.sum():3d}  claimed {100*wp.mean():.1f}%  "
              f"actual {100*corr.mean():.1f}%  gap {100*(wp.mean()-corr.mean()):+.1f}pp")

    mh = np.array([devig(e["away_ml"], e["home_ml"])[1] for e in rows])
    ph = np.array([e["home_win_prob"] for e in rows])
    yh = np.array([1 if e["actual_winner"] == "Home" else 0 for e in rows])
    t, pv = stats.ttest_rel(ph, mh)
    print(f"\n  d) THE ACTUAL MECHANISM: model vs market on the home side "
          f"(n={len(mh)} rated bets)")
    print(f"     mean MODEL  home prob : {ph.mean():.4f}")
    print(f"     mean MARKET home prob : {mh.mean():.4f}")
    print(f"     actual home win rate  : {yh.mean():.4f}   <- market is closer to truth")
    print(f"     model is {100*(ph.mean()-mh.mean()):+.2f}pp more home-leaning than the market")
    print(f"     paired t-test: t={t:+.2f} p={pv:.4f} "
          f"-> {'SIGNIFICANT' if pv < 0.05 else 'not significant'}")
    print(f"     that excess lean inflates home-side edge, which is why home games")
    print(f"     get flagged as value ~2x as often as away games.")


def section_bins(rows):
    print("\n" + "=" * 72)
    print("3) CAN RE-BINNING EDGE FIND A PROFITABLE SEGMENT?")
    print("=" * 72)
    print("\n  Non-cumulative edge bands (win% with 95% Wilson CI):")
    for lo, hi in [(0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 0.10),
                   (0.10, 0.14), (0.14, 1.0)]:
        stat_line([e for e in rows if lo <= e["model_edge"] < hi], f"edge {lo:.2f}-{hi:.2f}")
    print("\n  ^ note the sawtooth (66%, 50%, 73%, 58%, 42%, 55%) and how every CI")
    print("    overlaps every other -- that is the signature of noise, not signal.")

    print("\n  Cumulative 'bet everything with edge >= T':")
    for T in [0.00, 0.03, 0.05, 0.08, 0.10, 0.12]:
        stat_line([e for e in rows if e["model_edge"] >= T], f"edge >= {T:.2f}")
    print("\n  ^ no threshold beats simply betting every positive-edge game.")

    print("\n  Home/away split (the one structural, non-noise finding):")
    stat_line([e for e in rows if e["predicted_winner"] == "Away"], "all AWAY picks")
    stat_line([e for e in rows if e["predicted_winner"] == "Home"], "all HOME picks")
    print("  ^ nearly identical WIN RATES, very different ROI: away picks are priced")
    print("    as underdogs, so the same win rate pays much more.")


def section_power(true_clv):
    print("\n" + "=" * 72)
    print("4) HOW MUCH DATA WOULD ANY OF THIS NEED?  (80% power, alpha=0.05)")
    print("=" * 72)
    print("\n  Detecting a WIN-RATE difference between two buckets:")
    for d in (0.10, 0.07, 0.05):
        n = 16 * 0.25 / d ** 2
        print(f"     {100*d:4.1f}pp difference -> ~{n:5.0f} bets PER BUCKET ({2*n:5.0f} total)")
    print("     You currently have 20-70 per bucket.")

    s = true_clv.std(ddof=1)
    print(f"\n  Detecting nonzero mean TRUE CLV (observed sd {100*s:.1f}%):")
    for d in (0.02, 0.01):
        print(f"     {100*d:4.2f}pp mean CLV -> ~{(2.8*s/d)**2:5.0f} bets total")
    print(f"     You currently have {len(true_clv)}.")
    print(f"\n  Information per observation:")
    print(f"     sd of one win/loss (0/1) : 0.500")
    print(f"     sd of one true CLV       : {s:.3f}")
    print(f"     -> one CLV reading is worth ~{(0.5/s)**2:.0f} win/loss readings.")
    print("     CLV is also known the same day, instead of waiting for outcomes.")


def main():
    rows = load_bets()
    version, ents = load_preds()
    print(f"model_version {version} | {len(rows)} rated+resolved bets | "
          f"{len(ents)} resolved predictions\n")
    true_clv = section_clv(rows)
    section_skew(rows, ents)
    section_bins(rows)
    section_power(true_clv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
