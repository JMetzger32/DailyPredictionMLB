#!/usr/bin/env python3
"""
validate_true_edge.py — measure the REAL betting impact of the elastic-net penalty
change: how many games clear the value-bet threshold, and how those bets would have
performed, before vs after.

Why this script exists separately from validate_elasticnet_change.py: that script
could not compute true edges, because the local DB's 2026 games stop at 2026-07-07
while every row carrying stored odds runs 2026-07-16 onward — zero overlap, so
there were no features for the games that have prices.

The way around it: `predictions_log.json` stores `x_scaled_features`, the exact
18-float scaled vector used at prediction time. Multiplying by the scaler's
scale_ and adding mean_ recovers the RAW features — no DB needed. Rows logged
under an older model_version were scaled by that version's scaler, which is
recoverable from git history (the pkl is committed), so every odds row is usable
rather than just the current version's.

Reconstruction is verified against the logged probabilities before any comparison
is made; if it does not round-trip, the script refuses to report.

Report-only. Reads logs + pkls, writes one markdown report.

Usage:
    .venv/bin/python scripts/validate_true_edge.py
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Main"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from MLBModel import FEATURE_COLS, LR_PENALTY_KWARGS  # noqa: E402

from validate_elasticnet_change import BASELINE, build_model_df, fit_lr, md_table  # noqa: E402

warnings.resetwarnings()
warnings.simplefilter("ignore")

PRED_LOG = PROJECT_ROOT / "Databases_and_logs" / "predictions_log.json"
CUR_PKL = PROJECT_ROOT / "updates" / "mlb_model_artifacts.pkl"
OUT_PATH = PROJECT_ROOT / "scripts" / "results" / "true_edge_validation.md"

# model_version -> git commit holding the pkl that produced it. Needed to recover
# the scaler each logged x_scaled_features vector was produced with.
HISTORICAL_PKLS = {"b9133b95d2ec": "83fda83"}
PKL_REPO_PATH = "Desktop/CodeProjects/JackProject/updates/mlb_model_artifacts.pkl"

HOME_PRIOR, RECAL_BLEND = 0.53, 0.04
GOOD_EDGE, EXTREME_EDGE = 0.05, 0.12


def implied_probs(away_ml, home_ml):
    """De-vigged implied probabilities, mirroring Main/app.py::_implied_probs."""
    def raw(ml):
        ml = float(ml)
        return (100.0 / (ml + 100.0)) if ml > 0 else ((-ml) / ((-ml) + 100.0))
    a, h = raw(away_ml), raw(home_ml)
    t = a + h
    return a / t, h / t


def rate_edge(edge):
    """Mirrors Main/app.py::_rate_edge."""
    if edge is None or edge != edge:
        return None
    if edge > EXTREME_EDGE:
        return "extreme"
    if edge > GOOD_EDGE:
        return "good"
    if edge < -GOOD_EDGE:
        return "bad"
    return "unsure"


def load_scalers():
    """Scaler per model_version: current pkl plus any recoverable from git."""
    with open(CUR_PKL, "rb") as f:
        cur = pickle.load(f)
    scalers = {cur["model_version"]: cur["scaler"]}
    tmp = Path(sys.argv[0]).parent / "_tmp_hist.pkl"
    for mv, commit in HISTORICAL_PKLS.items():
        try:
            blob = subprocess.run(["git", "show", f"{commit}:{PKL_REPO_PATH}"],
                                   cwd=PROJECT_ROOT, capture_output=True, check=True).stdout
            tmp.write_bytes(blob)
            art = pickle.load(open(tmp, "rb"))
            if art.get("model_version") == mv:
                scalers[mv] = art["scaler"]
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not recover scaler for {mv} from {commit}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    return cur, scalers


def main() -> int:
    cur, scalers = load_scalers()
    print(f"scalers available for model_versions: {sorted(scalers)}")

    log = json.load(open(PRED_LOG))
    rows = [g for d in log.values() for g in d
            if g.get("away_ml") is not None and g.get("x_scaled_features")
            and g.get("model_version") in scalers]
    print(f"usable odds rows: {len(rows)}")

    # Recover raw features per row using its own version's scaler.
    Xraw = np.vstack([
        np.array(g["x_scaled_features"], float) * scalers[g["model_version"]].scale_
        + scalers[g["model_version"]].mean_
        for g in rows
    ])

    # --- Gate: reconstruction must reproduce the logged probability ---
    gb, boots = cur.get("gb_model"), cur.get("xgb_bootstrap_models") or []
    tree_parts = []
    if gb is not None:
        tree_parts.append(gb.predict_proba(Xraw)[:, 1])
    if boots:
        tree_parts.append(np.mean(np.vstack([b.predict_proba(Xraw)[:, 1] for b in boots]), axis=0))

    cur_mv = cur["model_version"]
    same = np.array([g["model_version"] == cur_mv for g in rows])
    p_shipped = cur["lr_model"].predict_proba(cur["scaler"].transform(Xraw))[:, 1]
    recon = np.mean(np.vstack([p_shipped] + tree_parts), axis=0)
    recon = (1 - RECAL_BLEND) * recon + RECAL_BLEND * HOME_PRIOR
    logged = np.array([g["home_win_prob"] for g in rows], float)
    err = float(np.abs(recon[same] - logged[same]).max()) if same.any() else float("nan")
    print(f"reconstruction check on {int(same.sum())} current-version rows: "
          f"max|recon-logged| = {err:.5f}")
    if not (err < 0.002):
        print("ABORT: reconstruction does not round-trip; refusing to report.")
        return 1

    # --- Candidate LRs, both trained on leak-fixed data ---
    print("training candidate LRs on leak-fixed data...")
    md = build_model_df(gate=True)
    d = md.dropna(subset=FEATURE_COLS)
    lr_l2, sc_l2 = fit_lr(d[FEATURE_COLS], d["home_win"], d["season"], **BASELINE)
    lr_en, sc_en = fit_lr(d[FEATURE_COLS], d["home_win"], d["season"],
                          penalty="elasticnet", C=LR_PENALTY_KWARGS["C"],
                          l1_ratio=LR_PENALTY_KWARGS["l1_ratio"])

    def served(lr, sc):
        p_lr = lr.predict_proba(sc.transform(Xraw))[:, 1]
        p = np.mean(np.vstack([p_lr] + tree_parts), axis=0)
        return (1 - RECAL_BLEND) * p + RECAL_BLEND * HOME_PRIOR

    variants = {"L2 (C=0.5)": served(lr_l2, sc_l2), "elastic net": served(lr_en, sc_en)}

    # --- True edges against the stored market prices ---
    frames = {}
    for name, p_home in variants.items():
        recs = []
        for i, g in enumerate(rows):
            a_imp, h_imp = implied_probs(g["away_ml"], g["home_ml"])
            home_pick = p_home[i] > 0.5
            model_p = p_home[i] if home_pick else 1 - p_home[i]
            mkt_p = h_imp if home_pick else a_imp
            ml = g["home_ml"] if home_pick else g["away_ml"]
            edge = model_p - mkt_p
            correct = g.get("correct")
            if correct is not None:
                aw = g.get("actual_winner")
                won = (aw == "Home") if home_pick else (aw == "Away")
            else:
                won = None
            recs.append({"date": g["date"], "edge": edge, "rating": rate_edge(edge),
                         "ml": float(ml), "won": won})
        frames[name] = pd.DataFrame(recs)

    def summarize(df):
        val = df[df["rating"] == "good"]          # value bets exclude "extreme"
        res = val[val["won"].notna()]
        if len(res):
            wins = res["won"].sum()
            profit = sum((r.ml / 100.0) if r.ml > 0 else (100.0 / -r.ml)
                          for r in res.itertuples() if r.won) - (len(res) - wins)
            roi = profit / len(res)
            wr = wins / len(res)
        else:
            roi, wr, wins = float("nan"), float("nan"), 0
        return {
            "value bets (good)": len(val),
            "extreme (>0.12)": int((df["rating"] == "extreme").sum()),
            "bad (<-0.05)": int((df["rating"] == "bad").sum()),
            "unsure": int((df["rating"] == "unsure").sum()),
            "mean edge (all)": df["edge"].mean(),
            "mean edge (value)": val["edge"].mean() if len(val) else float("nan"),
            "resolved value bets": len(res),
            "value-bet win%": wr,
            "flat-bet ROI": roi,
        }

    summ = pd.DataFrame({k: summarize(v) for k, v in frames.items()})
    summ["change"] = summ["elastic net"] - summ["L2 (C=0.5)"]

    a, b = frames["L2 (C=0.5)"], frames["elastic net"]
    moved_in = int(((a["rating"] != "good") & (b["rating"] == "good")).sum())
    moved_out = int(((a["rating"] == "good") & (b["rating"] != "good")).sum())

    L = ["# True edge validation — elastic net vs L2", "",
         f"_Generated {datetime.now().isoformat(timespec='seconds')} by "
         "`scripts/validate_true_edge.py` (report-only)._", "",
         "## Method", "",
         "The local DB's 2026 games stop at 2026-07-07 while every odds-carrying row runs "
         "2026-07-16 onward, so features for priced games could not be rebuilt. Instead the "
         "raw features are **recovered by inverting `x_scaled_features`** from "
         "`predictions_log.json` (`raw = scaled * scaler.scale_ + scaler.mean_`), using each "
         "row's own model_version scaler — the historical one recovered from git "
         f"(`{HISTORICAL_PKLS}`). No DB needed.", "",
         f"Reconstruction gate: replaying the shipped model through this path reproduces the "
         f"logged probability to **max abs error {err:.5f}** on "
         f"{int(same.sum())} current-version rows (the residual is the log's "
         "`round(prob, 3)`). The script aborts rather than report if this fails.", "",
         f"Both candidate LRs are trained on **leak-fixed** data; GB and the 50-model "
         f"bootstrap XGB are held fixed from the shipped pkl, isolating the penalty change. "
         f"Edges use the de-vigged stored prices and `_rate_edge`'s live thresholds "
         f"(good {GOOD_EDGE}-{EXTREME_EDGE}, extreme >{EXTREME_EDGE}).", "",
         f"n = **{len(rows)}** odds rows "
         f"({', '.join(f'{v} under {k}' for k, v in pd.Series([g['model_version'] for g in rows]).value_counts().items())}).",
         "", "## Results", "",
         md_table(summ.reset_index().rename(columns={"index": "metric"}), "{:.4f}"), "",
         f"- games entering the value-bet band under elastic net: **{moved_in}**",
         f"- games leaving it: **{moved_out}**", ""]

    n_res = summ.loc["resolved value bets", "elastic net"]
    L += ["## Reading this honestly", "",
          f"The resolved value-bet counts here are small (n={int(summ.loc['resolved value bets', 'L2 (C=0.5)'])} "
          f"vs {int(n_res)}), far below the ~400 bets/bucket CLAUDE.md notes are needed to "
          "resolve a 10pp win-rate gap. **Win% and ROI differences at this n are not "
          "evidence** — they are reported because they are the quantities that matter "
          "operationally, not because they discriminate between the two models.",
          "",
          "The load-bearing numbers are the **counts**: whether the change moves games "
          "across the 0.05 threshold in bulk. That is a mechanical property of the "
          "probability shift and is measurable at this n.", ""]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(L))

    print()
    print(summ.round(4).to_string())
    print(f"\nvalue-bet band: +{moved_in} in, -{moved_out} out")
    print(f"report -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
