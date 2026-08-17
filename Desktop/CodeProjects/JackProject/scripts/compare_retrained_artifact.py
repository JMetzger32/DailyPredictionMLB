#!/usr/bin/env python3
"""
compare_retrained_artifact.py — diff a freshly retrained model artifact against a
previous one, on the axes that actually matter: coefficients, served probabilities,
and true betting edges.

Motivation: validate_elasticnet_change.py and validate_true_edge.py both held the
tree models FIXED from the shipped pkl in order to isolate the LR penalty change.
A real retrain also rebuilds GB and the 50 bootstrap XGBs — on leak-fixed features —
so the combined effect is larger than either script measured. This script measures
the whole artifact, end to end.

Edges are computed the same way as validate_true_edge.py: raw features are recovered
by inverting `x_scaled_features` from predictions_log.json (the local DB has no rows
for the dates that carry odds), using the scaler of the model_version each row was
scored under.

Report-only: reads two pkls and the logs, writes one markdown report. Does not
modify any artifact.

Usage:
    .venv/bin/python scripts/compare_retrained_artifact.py OLD.pkl [NEW.pkl]
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Main"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from MLBModel import FEATURE_COLS  # noqa: E402

from validate_true_edge import (  # noqa: E402
    HOME_PRIOR, RECAL_BLEND, implied_probs, load_scalers, rate_edge,
)
from validate_elasticnet_change import md_table  # noqa: E402

warnings.resetwarnings()
warnings.simplefilter("ignore")

PRED_LOG = PROJECT_ROOT / "Databases_and_logs" / "predictions_log.json"
OUT_PATH = PROJECT_ROOT / "scripts" / "results" / "retrain_artifact_comparison.md"


def ensemble_prob(art, Xraw):
    """Reproduce predict_games_batch: mean(LR, GB, mean-of-bootstrap-XGB), then
    blend RECAL_BLEND toward HOME_PRIOR. LR uses scaled features, trees raw."""
    parts = [art["lr_model"].predict_proba(art["scaler"].transform(Xraw))[:, 1]]
    if art.get("gb_model") is not None:
        parts.append(art["gb_model"].predict_proba(Xraw)[:, 1])
    boots = art.get("xgb_bootstrap_models") or []
    if boots:
        parts.append(np.mean(np.vstack([b.predict_proba(Xraw)[:, 1] for b in boots]), axis=0))
    elif art.get("xgb_model") is not None:
        parts.append(art["xgb_model"].predict_proba(Xraw)[:, 1])
    p = np.mean(np.vstack(parts), axis=0)
    return (1 - RECAL_BLEND) * p + RECAL_BLEND * HOME_PRIOR


def edge_frame(p_home, rows):
    recs = []
    for i, g in enumerate(rows):
        a_imp, h_imp = implied_probs(g["away_ml"], g["home_ml"])
        home_pick = p_home[i] > 0.5
        model_p = p_home[i] if home_pick else 1 - p_home[i]
        mkt_p = h_imp if home_pick else a_imp
        ml = g["home_ml"] if home_pick else g["away_ml"]
        won = None
        if g.get("correct") is not None:
            won = (g.get("actual_winner") == "Home") if home_pick else (g.get("actual_winner") == "Away")
        recs.append({"edge": model_p - mkt_p, "rating": rate_edge(model_p - mkt_p),
                     "ml": float(ml), "won": won})
    return pd.DataFrame(recs)


def summarize(df):
    val = df[df["rating"] == "good"]
    res = val[val["won"].notna()]
    if len(res):
        wins = int(res["won"].sum())
        profit = sum((r.ml / 100.0) if r.ml > 0 else (100.0 / -r.ml)
                      for r in res.itertuples() if r.won) - (len(res) - wins)
        wr, roi = wins / len(res), profit / len(res)
    else:
        wr = roi = float("nan")
    return {"value bets (good)": len(val),
            "extreme (>0.12)": int((df["rating"] == "extreme").sum()),
            "bad (<-0.05)": int((df["rating"] == "bad").sum()),
            "unsure": int((df["rating"] == "unsure").sum()),
            "mean edge (all)": df["edge"].mean(),
            "resolved value bets": len(res),
            "value-bet win%": wr, "flat-bet ROI": roi}


def main() -> int:
    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "updates" / "mlb_model_artifacts.pkl"
    old = pickle.load(open(old_path, "rb"))
    new = pickle.load(open(new_path, "rb"))

    _, scalers = load_scalers()
    scalers.setdefault(old.get("model_version"), old["scaler"])

    log = json.load(open(PRED_LOG))
    rows = [g for d in log.values() for g in d
            if g.get("away_ml") is not None and g.get("x_scaled_features")
            and g.get("model_version") in scalers]
    Xraw = np.vstack([
        np.array(g["x_scaled_features"], float) * scalers[g["model_version"]].scale_
        + scalers[g["model_version"]].mean_ for g in rows])

    # Gate: the OLD artifact must reproduce the logged probabilities for rows it scored.
    p_old = ensemble_prob(old, Xraw)
    same = np.array([g["model_version"] == old.get("model_version") for g in rows])
    logged = np.array([g["home_win_prob"] for g in rows], float)
    err = float(np.abs(p_old[same] - logged[same]).max()) if same.any() else float("nan")
    if same.any() and not err < 0.002:
        print(f"ABORT: old artifact does not reproduce logged probs (max err {err:.5f})")
        return 1
    p_new = ensemble_prob(new, Xraw)

    c_old = pd.Series(old["lr_model"].coef_[0], index=old["feature_cols"]).reindex(FEATURE_COLS)
    c_new = pd.Series(new["lr_model"].coef_[0], index=new["feature_cols"]).reindex(FEATURE_COLS)
    zeroed = [f for f in FEATURE_COLS if abs(c_new[f]) < 1e-6]

    f_old, f_new = edge_frame(p_old, rows), edge_frame(p_new, rows)
    summ = pd.DataFrame({"old": summarize(f_old), "new": summarize(f_new)})
    summ["change"] = summ["new"] - summ["old"]
    moved_in = int(((f_old["rating"] != "good") & (f_new["rating"] == "good")).sum())
    moved_out = int(((f_old["rating"] == "good") & (f_new["rating"] != "good")).sum())

    rm_old, rm_new = old.get("retrain_metrics") or {}, new.get("retrain_metrics") or {}
    meta = pd.DataFrame({
        "metric": ["model_version", "saved_at", "holdout accuracy", "holdout Brier",
                   "holdout log loss", "train_size", "val_size"],
        "old": [old.get("model_version"), old.get("saved_at"), rm_old.get("accuracy"),
                rm_old.get("brier_score"), rm_old.get("log_loss"),
                rm_old.get("train_size"), rm_old.get("val_size")],
        "new": [new.get("model_version"), new.get("saved_at"), rm_new.get("accuracy"),
                rm_new.get("brier_score"), rm_new.get("log_loss"),
                rm_new.get("train_size"), rm_new.get("val_size")],
    })

    L = ["# Retrained artifact comparison", "",
         f"_Generated {datetime.now().isoformat(timespec='seconds')} by "
         "`scripts/compare_retrained_artifact.py` (report-only)._", "",
         f"`{old_path.name}` -> `{new_path.name}`", "",
         "Unlike the earlier validations, which held GB/XGB fixed to isolate the LR "
         "penalty, this compares **fully retrained artifacts** — LR, GB and the 50 "
         "bootstrap XGBs all rebuilt on leak-fixed features.", "",
         "## Artifact metadata", "", md_table(meta), "",
         f"Reconstruction gate: the old artifact reproduces its own logged probabilities "
         f"to max abs error **{err:.5f}** on {int(same.sum())} rows.", "",
         "## LR coefficients", "",
         f"Coefficients driven to exactly zero by the elastic-net penalty: "
         f"**{len(zeroed)}/{len(FEATURE_COLS)}**", "",
         ("- " + "\n- ".join(f"`{z}`" for z in zeroed)) if zeroed else "_(none)_", "",
         md_table(pd.DataFrame({"old": c_old, "new": c_new, "delta": c_new - c_old})
                  .reindex(c_old.abs().sort_values(ascending=False).index)
                  .reset_index().rename(columns={"index": "feature"})), "",
         "## Served probabilities", "",
         md_table(pd.DataFrame({
             "metric": ["sd of served prob", "mean |p-0.5|", "max |p-0.5|",
                        "mean |delta| vs old", "max |delta| vs old",
                        "predicted winner flips"],
             "value": [p_new.std(), np.abs(p_new - 0.5).mean(), np.abs(p_new - 0.5).max(),
                       np.abs(p_new - p_old).mean(), np.abs(p_new - p_old).max(),
                       int(((p_old > 0.5) != (p_new > 0.5)).sum())],
             "old": [p_old.std(), np.abs(p_old - 0.5).mean(), np.abs(p_old - 0.5).max(),
                     np.nan, np.nan, np.nan],
         })[["metric", "old", "value"]].rename(columns={"value": "new"}), "{:.5f}"), "",
         f"## True betting edges (n={len(rows)} priced games)", "",
         md_table(summ.reset_index().rename(columns={"index": "metric"}), "{:.4f}"), "",
         f"- entering the value-bet band: **{moved_in}**; leaving it: **{moved_out}**", "",
         "**Read the counts, not the ROI.** Resolved value-bet counts are ~100, far below "
         "the ~400/bucket CLAUDE.md notes are needed to resolve a 10pp win-rate gap, so "
         "win% and ROI differences here are not evidence either way. Whether the bet "
         "volume holds up is the question this n can answer.", ""]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(L))

    print(meta.to_string(index=False))
    print(f"\nzeroed ({len(zeroed)}): {zeroed}")
    print(f"\nserved prob: mean|delta|={np.abs(p_new - p_old).mean():.5f}  "
          f"max={np.abs(p_new - p_old).max():.5f}  "
          f"flips={int(((p_old > 0.5) != (p_new > 0.5)).sum())}/{len(rows)}")
    print(f"\n{summ.round(4).to_string()}")
    print(f"\nvalue band: +{moved_in} in, -{moved_out} out")
    print(f"report -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
