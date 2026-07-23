#!/usr/bin/env python3
"""
Live-calibration check for the CURRENTLY DEPLOYED model_version.

Why this exists: MLBModel.py's recalibration step (a flat 4% blend toward a
0.53 home prior) is a fixed constant, not fit from data. A data-driven fit
(Platt/isotonic) can only be trusted on data that reflects real prediction-time
conditions for the model actually in production — which means predictions_log
entries tagged with the CURRENT model_version, not the 2021-2025 DB (the
deployed model is trained on that data, so scoring it there leaks) and not
older log entries from a superseded model_version.

Run:  python3 scripts/calibration_live_check.py

Decision rule: only recommend shipping a fitted calibrator once 5-fold
cross-validated ECE for Platt scaling beats the current flat-blend ECE by a
real margin. At small n (tens of games) CV usually shows fitting HURTS —
that's the sample telling you it's not enough data yet, not a bug.
"""
import os
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
PKL = os.path.join(_ROOT, "updates", "mlb_model_artifacts.pkl")

BLEND, PRIOR = 0.04, 0.53          # must match MLBModel.py's current recalibration
MIN_MARGIN = 0.01                   # CV ECE must beat baseline by this much to recommend shipping


def current_model_version():
    import pickle
    with open(PKL, "rb") as f:
        art = pickle.load(f)
    return art.get("model_version")


def reliability_ece(probs_home, entries, bins=10):
    b = [{"lo": i * 10, "hi": i * 10 + 10, "mid": (i * 10 + 5) / 100, "w": 0, "n": 0} for i in range(bins)]
    for p, e in zip(probs_home, entries):
        wp = p if e.get("predicted_winner") == "Home" else 1 - p
        idx = min(int(wp * bins), bins - 1)
        b[idx]["n"] += 1
        b[idx]["w"] += 1 if e["correct"] else 0
    total = sum(x["n"] for x in b)
    ece, rows = 0.0, []
    for x in b:
        if x["n"] == 0:
            continue
        act = x["w"] / x["n"]
        ece += abs(act - x["mid"]) * x["n"] / total
        rows.append((x["lo"], x["hi"], x["mid"], act, x["n"]))
    return ece, rows


def fmt(title, rows, ece, brier):
    out = [title, f"{'bin':>10} | {'pred_mid':>8} | {'actual':>7} | {'count':>6}", "-" * 42]
    for lo, hi, mid, act, n in rows:
        out.append(f"{lo:>3}-{hi:<3}% | {mid:>8.2f} | {act:>7.3f} | {n:>6}")
    out.append(f"ECE = {ece:.4f}   Brier = {brier:.4f}")
    return "\n".join(out)


def main():
    version = current_model_version()
    log = json.load(open(LOG))
    entries = [e for day in log.values() for e in day
               if e.get("correct") is not None
               and e.get("game_type") == "R"
               and e.get("post_game_created") is not True
               and e.get("home_win_prob") is not None
               and e.get("model_version") == version]

    print(f"Deployed model_version: {version}")
    print(f"Live resolved games for this version: {len(entries)}\n")
    if len(entries) < 20:
        print("Too few games (<20) to say anything — skipping fit.")
        return

    raw, home_win = [], []
    for e in entries:
        blended = e["home_win_prob"]
        r = (blended - BLEND * PRIOR) / (1 - BLEND)
        raw.append(min(max(r, 1e-6), 1 - 1e-6))
        home_win.append(1 if e["actual_winner"] == "Home" else 0)
    raw = np.array(raw)
    home_win = np.array(home_win)
    logit_raw = np.log(raw / (1 - raw))

    blended_arr = np.array([e["home_win_prob"] for e in entries])
    ece_before, rows_before = reliability_ece(blended_arr, entries)
    print(fmt("=== CURRENT (flat 4% blend, actual production output) ===",
              rows_before, ece_before, np.mean((blended_arr - home_win) ** 2)))

    n_splits = min(5, len(entries) // 10) if len(entries) >= 50 else min(5, max(2, len(entries) // 15))
    kf = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)

    platt_cv = np.zeros_like(raw)
    for tr, te in kf.split(logit_raw):
        lr = LogisticRegression()
        lr.fit(logit_raw[tr].reshape(-1, 1), home_win[tr])
        platt_cv[te] = lr.predict_proba(logit_raw[te].reshape(-1, 1))[:, 1]
    ece_platt, rows_platt = reliability_ece(platt_cv, entries)
    print()
    print(fmt("=== PLATT SCALING (cross-validated, honest out-of-sample) ===",
              rows_platt, ece_platt, np.mean((platt_cv - home_win) ** 2)))

    iso_cv = np.zeros_like(raw)
    for tr, te in kf.split(raw):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw[tr], home_win[tr])
        iso_cv[te] = iso.predict(raw[te])
    ece_iso, rows_iso = reliability_ece(iso_cv, entries)
    print()
    print(fmt("=== ISOTONIC (cross-validated, honest out-of-sample) ===",
              rows_iso, ece_iso, np.mean((iso_cv - home_win) ** 2)))

    print(f"\nSUMMARY (n={len(entries)}):")
    print(f"  current flat blend : ECE={ece_before:.4f}")
    print(f"  Platt (CV)         : ECE={ece_platt:.4f}")
    print(f"  Isotonic (CV)      : ECE={ece_iso:.4f}")

    best_fit_ece = min(ece_platt, ece_iso)
    if ece_before - best_fit_ece > MIN_MARGIN:
        which = "Platt" if ece_platt <= ece_iso else "Isotonic"
        print(f"\nRECOMMENDATION: {which} beats the flat blend by "
              f"{ece_before - best_fit_ece:.4f} (> {MIN_MARGIN} margin) — safe to fit "
              f"and ship a real calibrator now.")
    else:
        print(f"\nRECOMMENDATION: no fit beats the flat blend by the {MIN_MARGIN} margin yet "
              f"(best gain: {ece_before - best_fit_ece:+.4f}). Keep the flat blend. "
              f"Re-run this script as more games accumulate for model_version {version}.")


if __name__ == "__main__":
    main()
