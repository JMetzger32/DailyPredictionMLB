"""
calibrate_and_recompute_edge.py
-------------------------------
Phase 3 of the edge-calibration investigation: if model_edge were computed from a
CALIBRATED probability instead of the production flat-blend one, would the
Value-Bet / Toss-Up inversion resolve, and would Quarter-Kelly stop returning ~$0?

    3a  fit Platt scaling on resolved predictions for the deployed model_version
    3b  recompute model_edge for every resolved bet from the calibrated probability
    3c  re-classify Value Bet / Toss-Up / No Value at the same thresholds
    3d  recompute Quarter-Kelly net P/L using the recalibrated edge for sizing

Usage:
    python3 scripts/calibrate_and_recompute_edge.py [--stake 10]

Honesty of the fit: calibrated probabilities used for evaluation are OUT-OF-FOLD
(KFold CV), so no row is scored by a calibrator that saw its own outcome. Method
mirrors scripts/calibration_live_check.py — un-blend production's flat 4%-toward-0.53
recalibration to recover the raw probability, then fit LogisticRegression on its logit.

Market probabilities are re-derived from away_ml/home_ml with the same de-vig as
updates/schedule_fetcher.py; Phase 1 verified this reproduces the stored model_edge
exactly on all 222 rows, so any change here comes purely from the probability swap.

Report-only: this script changes nothing.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
BET_LOG  = os.path.join(_ROOT, "Databases_and_logs", "betting_log.json")
PKL      = os.path.join(_ROOT, "updates", "mlb_model_artifacts.pkl")

BLEND, PRIOR = 0.04, 0.53      # must match MLBModel.py:940-944
EXTREME_EDGE, GOOD_EDGE = 0.12, 0.05   # must match Main/app.py:1482-1490
# live /api/betting defaults (Main/app.py:2474-2476)
BANKROLL, KELLY_FRACTION, MAX_STAKE_PCT = 100.0, 0.25, 0.15


def american_to_raw(ml):
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def devig(away_ml, home_ml):
    a, h = american_to_raw(away_ml), american_to_raw(home_ml)
    t = a + h
    return round(a / t, 4), round(h / t, 4)


def kelly_stake(win_prob, ml, bankroll, fraction, max_stake_pct,
                edge=None, max_edge_for_sizing=0.10):
    """Faithful copy of Main/app.py:2292-2316 INCLUDING the edge cap — note
    scripts/backtest_kelly.py's older copy omits it."""
    if win_prob is None or ml is None:
        return None
    if edge is not None and abs(edge) > max_edge_for_sizing:
        capped_edge = max_edge_for_sizing if edge > 0 else -max_edge_for_sizing
        win_prob = win_prob - edge + capped_edge
    b = (ml / 100) if ml >= 0 else (100 / abs(ml))
    f = (win_prob * b - (1 - win_prob)) / b
    if f <= 0:
        return 0.0
    return round(bankroll * min(f * fraction, max_stake_pct), 2)


def pl_for(ml, won, stake):
    if not won:
        return -stake
    return stake * (ml / 100 if ml >= 0 else 100 / abs(ml))


def rate(edge):
    """Main/app.py:1482-1490."""
    if edge > EXTREME_EDGE:
        return "extreme"
    if edge > GOOD_EDGE:
        return "good"
    if edge < -GOOD_EDGE:
        return "bad"
    return "unsure"


def reliability_ece(probs_home, entries, bins=10):
    """Winner-side reliability, identical to calibration_live_check.py:44-59."""
    b = [{"mid": (i * 10 + 5) / 100, "w": 0, "n": 0} for i in range(bins)]
    for p, e in zip(probs_home, entries):
        wp = p if e.get("predicted_winner") == "Home" else 1 - p
        idx = min(int(wp * bins), bins - 1)
        b[idx]["n"] += 1
        b[idx]["w"] += 1 if e["correct"] else 0
    total = sum(x["n"] for x in b)
    return sum(abs(x["w"] / x["n"] - x["mid"]) * x["n"] / total for x in b if x["n"])


def load():
    with open(PKL, "rb") as f:
        version = pickle.load(f).get("model_version")
    with open(PRED_LOG) as f:
        plog = json.load(f)
    fit_rows = [e for day in plog.values() for e in day
                if e.get("correct") is not None
                and e.get("game_type") == "R"
                and e.get("post_game_created") is not True
                and e.get("home_win_prob") is not None
                and e.get("model_version") == version]
    with open(BET_LOG) as f:
        blog = json.load(f)
    bet_rows = [e for day in sorted(blog) for e in blog[day]
                if e.get("bet_rating") and e.get("correct") is not None
                and e.get("game_type") == "R"]
    return version, fit_rows, bet_rows


def summarize(rows, stake, key="bet_rating"):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    res = {}
    for label, sub in out.items():
        w = sum(1 for e in sub if e["correct"])
        net = sum(pl_for(e["predicted_team_ml"], e["correct"], stake) for e in sub)
        res[label] = (len(sub), w, len(sub) - w, 100 * w / len(sub), net,
                      100 * net / (len(sub) * stake))
    return res


def print_cat(title, res, order=("good", "extreme", "unsure", "bad")):
    print(title)
    print("| category | n | W-L | win% | net P/L | ROI |")
    print("|---|---|---|---|---|---|")
    for k in order:
        if k not in res:
            continue
        n, w, l, pct, net, roi = res[k]
        print(f"| {k} | {n} | {w}-{l} | {pct:.1f}% | ${net:+.2f} | {roi:+.1f}% |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=float, default=10.0)
    args = ap.parse_args()

    version, fit_rows, bet_rows = load()
    print(f"Deployed model_version: {version}")
    print(f"Calibration fit sample: {len(fit_rows)} resolved predictions")
    print(f"Evaluation sample:      {len(bet_rows)} rated+resolved bets\n")

    # ---- 3a: fit Platt ------------------------------------------------------
    raw = np.array([min(max((e["home_win_prob"] - BLEND * PRIOR) / (1 - BLEND), 1e-6),
                        1 - 1e-6) for e in fit_rows])
    home_win = np.array([1 if e["actual_winner"] == "Home" else 0 for e in fit_rows])
    logit_raw = np.log(raw / (1 - raw)).reshape(-1, 1)
    blended = np.array([e["home_win_prob"] for e in fit_rows])

    full = LogisticRegression().fit(logit_raw, home_win)
    slope, intercept = float(full.coef_[0][0]), float(full.intercept_[0])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(fit_rows))
    for tr, te in kf.split(logit_raw):
        lr = LogisticRegression().fit(logit_raw[tr], home_win[tr])
        oof[te] = lr.predict_proba(logit_raw[te])[:, 1]

    print("## 3a. Fitted Platt scaling")
    print(f"  calibrated_logit = {intercept:+.4f} + {slope:.4f} * logit(raw_prob)")
    print(f"  slope < 1 means the model is overconfident (spread gets shrunk)")
    print(f"  ECE  production flat blend : {reliability_ece(blended, fit_rows):.4f}")
    print(f"  ECE  Platt (out-of-fold)   : {reliability_ece(oof, fit_rows):.4f}")
    print(f"  Brier production           : {np.mean((blended - home_win) ** 2):.4f}")
    print(f"  Brier Platt (out-of-fold)  : {np.mean((oof - home_win) ** 2):.4f}")
    print(f"  prob range production      : [{blended.min():.3f}, {blended.max():.3f}]")
    print(f"  prob range calibrated      : [{oof.min():.3f}, {oof.max():.3f}]")

    cal_by_pk = {e["game_pk"]: p for e, p in zip(fit_rows, oof)}

    # ---- 3b: recompute edge -------------------------------------------------
    rows = []
    for e in bet_rows:
        pk = e["game_pk"]
        if pk not in cal_by_pk:
            continue
        ai, hi = devig(e["away_ml"], e["home_ml"])
        is_home = e["predicted_winner"] == "Home"
        mkt = hi if is_home else ai
        cal_home = cal_by_pk[pk]
        cal_prob = cal_home if is_home else 1 - cal_home
        rows.append({**e,
                     "cal_prob": cal_prob,
                     "old_edge": e["model_edge"],
                     "new_edge": round(cal_prob - mkt, 4),
                     "old_rating": e["bet_rating"],
                     "consistent_rating": rate(e["model_edge"]),
                     "new_rating": rate(cal_prob - mkt)})

    print(f"\n## 3b. Edge recomputed on {len(rows)} bets")
    old_e = np.array([r["old_edge"] for r in rows])
    new_e = np.array([r["new_edge"] for r in rows])
    print(f"  mean |edge|  production {np.abs(old_e).mean():.4f} -> calibrated {np.abs(new_e).mean():.4f}")
    print(f"  max  edge    production {old_e.max():.4f} -> calibrated {new_e.max():.4f}")
    print(f"  bets with edge > 0.12: {int((old_e > 0.12).sum())} -> {int((new_e > 0.12).sum())}")
    print(f"  bets with edge > 0.05: {int((old_e > 0.05).sum())} -> {int((new_e > 0.05).sum())}")

    # ---- 3c: re-classify ----------------------------------------------------
    print("\n## 3c. Classification before vs after")
    print_cat("\nA) AS-LABELED (what the betting page shows today)",
              summarize(rows, args.stake, "old_rating"))
    print_cat("\nB) PRODUCTION EDGE, thresholds applied consistently",
              summarize(rows, args.stake, "consistent_rating"))
    print_cat("\nC) CALIBRATED EDGE, same thresholds",
              summarize(rows, args.stake, "new_rating"))

    def gap(res):
        v = res.get("good")
        t = res.get("unsure")
        if not v or not t:
            return None
        return v[3] - t[3]

    for name, key in (("A as-labeled", "old_rating"),
                      ("B consistent", "consistent_rating"),
                      ("C calibrated", "new_rating")):
        res = summarize(rows, args.stake, key)
        g = gap(res)
        print(f"  {name:14s} Value − TossUp win% gap: "
              f"{g:+.1f}pp" if g is not None else f"  {name}: n/a")

    # ---- 3d: Quarter-Kelly --------------------------------------------------
    print("\n## 3d. Quarter-Kelly net P/L "
          f"(bankroll ${BANKROLL:.0f}, fraction {KELLY_FRACTION}, cap {MAX_STAKE_PCT:.0%})")
    print("| sizing basis | bet set | bets | staked | net P/L | ROI |")
    print("|---|---|---|---|---|---|")

    def run_kelly(subset, prob_key, edge_key, label, setname):
        bets = staked = net = 0.0
        nb = 0
        for r in subset:
            wp = r[prob_key] if prob_key == "cal_prob" else (
                r["home_win_prob"] if r["predicted_winner"] == "Home" else r["away_win_prob"])
            stake = kelly_stake(wp, r["predicted_team_ml"], BANKROLL, KELLY_FRACTION,
                                MAX_STAKE_PCT, edge=r[edge_key])
            if not stake:
                continue
            nb += 1
            staked += stake
            net += pl_for(r["predicted_team_ml"], r["correct"], stake)
        roi = 100 * net / staked if staked else 0.0
        print(f"| {label} | {setname} | {nb} | ${staked:.2f} | ${net:+.2f} | {roi:+.1f}% |")
        return net

    as_labeled_good = [r for r in rows if r["old_rating"] == "good"]
    consistent_good = [r for r in rows if r["consistent_rating"] == "good"]
    calibrated_good = [r for r in rows if r["new_rating"] == "good"]

    run_kelly(as_labeled_good, "prod", "old_edge",
              "production prob + production edge", "as-labeled good (live baseline)")
    run_kelly(consistent_good, "prod", "old_edge",
              "production prob + production edge", "consistent good")
    run_kelly(calibrated_good, "cal_prob", "new_edge",
              "CALIBRATED prob + calibrated edge", "calibrated good")
    run_kelly(rows, "cal_prob", "new_edge",
              "CALIBRATED prob + calibrated edge", "all rated bets")

    # ---- 3e: is any of this distinguishable from noise? ---------------------
    from statsmodels.stats.proportion import proportions_ztest, proportion_confint
    import statsmodels.api as sm

    print("\n## 3e. Significance of the Value-vs-TossUp gap")
    print("| scenario | Value | 95% CI | Toss-Up | 95% CI | gap | p |")
    print("|---|---|---|---|---|---|---|")
    for name, key in (("A as-labeled", "old_rating"),
                      ("B consistent", "consistent_rating"),
                      ("C calibrated", "new_rating")):
        res = summarize(rows, args.stake, key)
        (n1, w1, _, p1, _, _) = res["good"]
        (n2, w2, _, p2, _, _) = res["unsure"]
        z, pv = proportions_ztest([w1, w2], [n1, n2])
        c1 = proportion_confint(w1, n1, method="wilson")
        c2 = proportion_confint(w2, n2, method="wilson")
        print(f"| {name} | {w1}/{n1} = {p1:.1f}% | [{100*c1[0]:.1f}, {100*c1[1]:.1f}] "
              f"| {w2}/{n2} = {p2:.1f}% | [{100*c2[0]:.1f}, {100*c2[1]:.1f}] "
              f"| {p1-p2:+.1f}pp | {pv:.3f} |")

    def lg(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return np.log(p / (1 - p))

    won = np.array([1 if r["correct"] else 0 for r in rows])
    ed = np.array([r["old_edge"] for r in rows])
    r2 = sm.Logit(won, sm.add_constant(ed)).fit(disp=0)
    print(f"\nDoes edge predict the pick winning?  logit(win) ~ const + model_edge")
    print(f"  edge coef = {r2.params[1]:+.4f}  se={r2.bse[1]:.4f}  p={r2.pvalues[1]:.3f}")

    y = np.array([1 if r["actual_winner"] == "Home" else 0 for r in rows])
    X = sm.add_constant(np.array([[lg(devig(r["away_ml"], r["home_ml"])[1]),
                                   lg(r["home_win_prob"])] for r in rows]))
    r3 = sm.Logit(y, X).fit(disp=0)
    print("Does the model add information beyond the market?  "
          "logit(home win) ~ const + market + model")
    for nm, c, se, pv in zip(("const", "market", "model"),
                             r3.params, r3.bse, r3.pvalues):
        print(f"  {nm:7s} coef={c:+.4f}  se={se:.4f}  p={pv:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
