"""
calibrate_edge_at_scale.py
---------------------------
Re-run the calibrate_and_recompute_edge.py test (2026-08-06, rejected: "better
calibration did not produce better betting") at the scale the historical backfill
unlocked. The original test fit on n=283 and evaluated on n=222 -- below the
~400/bucket power CLAUDE.md itself says is needed to resolve a 10pp gap. This fits
on up to 11,072 walk-forward out-of-sample predictions and evaluates on the subset
priced against real historical odds (n=10,629).

Reuses rate()/kelly_stake()/pl_for()/devig()/reliability_ece() from
calibrate_and_recompute_edge.py unchanged -- only the data loader differs (OOF table
+ backfilled odds instead of predictions_log.json + betting_log.json). Fit sample is
the FULL walk-forward set (odds not required for fitting a probability calibrator);
eval/edge/ROI sample is restricted to the priced subset, mirroring the original
script's fit_rows > bet_rows structure.

Report-only: changes nothing.

Usage: .venv/bin/python scripts/calibrate_edge_at_scale.py [--stake 10] [--scheme walkforward]
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "Main"))
DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

from calibrate_and_recompute_edge import (  # noqa: E402
    devig, kelly_stake, pl_for, rate, reliability_ece,
    BANKROLL, KELLY_FRACTION, MAX_STAKE_PCT,
)
from MLBModel import HOME_PRIOR, RECAL_BLEND  # noqa: E402


def load_fit_rows(conn, scheme):
    """Every OOF prediction with a known outcome -- odds not required to fit."""
    rows = conn.execute(
        "SELECT game_id, home_win_prob, home_win, season FROM oof_predictions "
        "WHERE scheme = ?", (scheme,)).fetchall()
    out = []
    for g, p, w, s in rows:
        home_pick = p > 0.5
        out.append({
            "game_id": g, "home_win_prob": p, "season": s,
            "actual_winner": "Home" if w else "Away",
            # reliability_ece() scores the WINNER side, so it needs both of these
            "predicted_winner": "Home" if home_pick else "Away",
            "correct": bool(w) == home_pick,
        })
    return out


def load_priced_rows(conn, scheme):
    """OOF predictions joined to primary backfilled odds -- the eval/edge subset."""
    rows = conn.execute("""
        SELECT o.game_id, o.home_win_prob, o.home_win, o.season, s.away_ml, s.home_ml
        FROM oof_predictions o
        JOIN odds_game_link l ON l.game_id = o.game_id AND l.target='games'
        JOIN odds_snapshots s ON s.game_date_et = l.game_date_et
                             AND s.event_id = l.event_id
        WHERE o.scheme = ? AND s.horizon_days = 0
          AND s.away_ml IS NOT NULL AND l.confidence = 'exact'
    """, (scheme,)).fetchall()
    out = []
    for gid, p, w, season, aml, hml in rows:
        home_pick = p > 0.5
        predicted_team_ml = hml if home_pick else aml
        ai, hi = devig(aml, hml)
        mkt = hi if home_pick else ai
        old_edge = (p if home_pick else 1 - p) - mkt
        out.append({
            "game_id": gid, "home_win_prob": p, "away_ml": aml, "home_ml": hml,
            "predicted_winner": "Home" if home_pick else "Away",
            "predicted_team_ml": predicted_team_ml,
            "actual_winner": "Home" if w else "Away",
            "correct": bool(w) == home_pick,
            "old_edge": round(old_edge, 4),
            "consistent_rating": rate(old_edge),
            "season": season,
        })
    return out


def summarize(rows, stake, key):
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
    ap.add_argument("--scheme", default="walkforward")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    fit_rows = load_fit_rows(conn, args.scheme)
    priced = load_priced_rows(conn, args.scheme)
    print(f"scheme: {args.scheme}")
    print(f"Calibration fit sample: {len(fit_rows)} out-of-fold predictions "
          f"(vs n=283 in the 2026-08-06 test)")
    print(f"Evaluation sample:      {len(priced)} priced, resolved games "
          f"(vs n=222 in the 2026-08-06 test)\n")

    # ---- 3a: fit Platt (un-blend the home-prior nudge first, then refit OOF) ----
    raw = np.array([min(max((e["home_win_prob"] - RECAL_BLEND * HOME_PRIOR) /
                            (1 - RECAL_BLEND), 1e-6), 1 - 1e-6) for e in fit_rows])
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

    cal_by_gid = {e["game_id"]: p for e, p in zip(fit_rows, oof)}

    # ---- 3b: recompute edge on the priced subset ----
    rows = []
    for e in priced:
        gid = e["game_id"]
        if gid not in cal_by_gid:
            continue
        ai, hi = devig(e["away_ml"], e["home_ml"])
        is_home = e["predicted_winner"] == "Home"
        mkt = hi if is_home else ai
        cal_home = cal_by_gid[gid]
        cal_prob = cal_home if is_home else 1 - cal_home
        rows.append({**e,
                     "cal_prob": cal_prob,
                     "new_edge": round(cal_prob - mkt, 4),
                     "new_rating": rate(cal_prob - mkt)})

    print(f"\n## 3b. Edge recomputed on {len(rows)} priced games")
    old_e = np.array([r["old_edge"] for r in rows])
    new_e = np.array([r["new_edge"] for r in rows])
    print(f"  mean |edge|  walk-forward {np.abs(old_e).mean():.4f} -> calibrated {np.abs(new_e).mean():.4f}")
    print(f"  max  edge    walk-forward {old_e.max():.4f} -> calibrated {new_e.max():.4f}")
    print(f"  bets with edge > 0.12: {int((old_e > 0.12).sum())} -> {int((new_e > 0.12).sum())}")
    print(f"  bets with edge > 0.05: {int((old_e > 0.05).sum())} -> {int((new_e > 0.05).sum())}")

    print("\n## 3c. Classification before vs after")
    print_cat("\nB) WALK-FORWARD EDGE, consistent thresholds (the honest baseline)",
              summarize(rows, args.stake, "consistent_rating"))
    print_cat("\nC) CALIBRATED EDGE, same thresholds",
              summarize(rows, args.stake, "new_rating"))

    def gap(res):
        v, t = res.get("good"), res.get("unsure")
        return (v[3] - t[3]) if (v and t) else None

    for name, key in (("B walk-forward", "consistent_rating"),
                      ("C calibrated", "new_rating")):
        res = summarize(rows, args.stake, key)
        g = gap(res)
        print(f"  {name:16s} Value - TossUp win% gap: "
              f"{g:+.1f}pp" if g is not None else f"  {name}: n/a")

    # per-season stability of the "good" bucket -- the check that actually matters
    print("\n## 3c-bis. Per-season stability of the 'good' bucket, before vs after")
    print("| season | n(walk-fwd good) | win% | n(cal good) | win% |")
    print("|---|---|---|---|---|")
    for s in sorted({r["season"] for r in rows}):
        sr = [r for r in rows if r["season"] == s]
        wf = [r for r in sr if r["consistent_rating"] == "good"]
        cal = [r for r in sr if r["new_rating"] == "good"]
        wf_w = f"{100*sum(1 for r in wf if r['correct'])/len(wf):.1f}%" if wf else "-"
        cal_w = f"{100*sum(1 for r in cal if r['correct'])/len(cal):.1f}%" if cal else "-"
        print(f"| {s} | {len(wf)} | {wf_w} | {len(cal)} | {cal_w} |")

    print("\n## 3d. Quarter-Kelly net P/L "
          f"(bankroll ${BANKROLL:.0f}, fraction {KELLY_FRACTION}, cap {MAX_STAKE_PCT:.0%})")
    print("| sizing basis | bet set | bets | staked | net P/L | ROI |")
    print("|---|---|---|---|---|---|")

    def run_kelly(subset, prob_key, edge_key, label, setname):
        staked = net = 0.0
        nb = 0
        for r in subset:
            wp = r[prob_key] if prob_key == "cal_prob" else (
                r["home_win_prob"] if r["predicted_winner"] == "Home" else 1 - r["home_win_prob"])
            stake = kelly_stake(wp, r["predicted_team_ml"], BANKROLL, KELLY_FRACTION,
                                MAX_STAKE_PCT, edge=r[edge_key])
            if not stake:
                continue
            nb += 1
            staked += stake
            net += pl_for(r["predicted_team_ml"], r["correct"], stake)
        roi = 100 * net / staked if staked else 0.0
        print(f"| {label} | {setname} | {nb} | ${staked:.2f} | ${net:+.2f} | {roi:+.1f}% |")

    consistent_good = [r for r in rows if r["consistent_rating"] == "good"]
    calibrated_good = [r for r in rows if r["new_rating"] == "good"]
    run_kelly(consistent_good, "prod", "old_edge",
              "walk-forward prob + walk-forward edge", "consistent good")
    run_kelly(calibrated_good, "cal_prob", "new_edge",
              "CALIBRATED prob + calibrated edge", "calibrated good")
    run_kelly(rows, "cal_prob", "new_edge",
              "CALIBRATED prob + calibrated edge", "all priced games")

    from statsmodels.stats.proportion import proportions_ztest, proportion_confint
    import statsmodels.api as sm

    print("\n## 3e. Significance of the Value-vs-TossUp gap")
    print("| scenario | Value | 95% CI | Toss-Up | 95% CI | gap | p |")
    print("|---|---|---|---|---|---|---|")
    for name, key in (("B walk-forward", "consistent_rating"),
                      ("C calibrated", "new_rating")):
        res = summarize(rows, args.stake, key)
        if "good" not in res or "unsure" not in res:
            continue
        (n1, w1, _, p1, _, _) = res["good"]
        (n2, w2, _, p2, _, _) = res["unsure"]
        z, pv = proportions_ztest([w1, w2], [n1, n2])
        c1 = proportion_confint(w1, n1, method="wilson")
        c2 = proportion_confint(w2, n2, method="wilson")
        print(f"| {name} | {w1}/{n1} = {p1:.1f}% | [{100*c1[0]:.1f}, {100*c1[1]:.1f}] "
              f"| {w2}/{n2} = {p2:.1f}% | [{100*c2[0]:.1f}, {100*c2[1]:.1f}] "
              f"| {p1-p2:+.1f}pp | {pv:.3f} |")

    won = np.array([1 if r["correct"] else 0 for r in rows])
    ed_old = np.array([r["old_edge"] for r in rows])
    ed_new = np.array([r["new_edge"] for r in rows])
    r2 = sm.Logit(won, sm.add_constant(ed_old)).fit(disp=0)
    print(f"\nDoes WALK-FORWARD edge predict winning?  logit(win) ~ const + model_edge  (n={len(rows)})")
    print(f"  edge coef = {r2.params[1]:+.4f}  se={r2.bse[1]:.4f}  p={r2.pvalues[1]:.3f}")
    r2b = sm.Logit(won, sm.add_constant(ed_new)).fit(disp=0)
    print(f"Does CALIBRATED edge predict winning?  logit(win) ~ const + calibrated_edge  (n={len(rows)})")
    print(f"  edge coef = {r2b.params[1]:+.4f}  se={r2b.bse[1]:.4f}  p={r2b.pvalues[1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
