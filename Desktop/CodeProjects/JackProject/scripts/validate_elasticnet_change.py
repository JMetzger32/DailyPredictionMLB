#!/usr/bin/env python3
"""
validate_elasticnet_change.py — validate (1) the 2026 SP leak fix's effect on the
model, and (2) the proposed elastic-net penalty, before either ships.

Report-only: fits models in memory and writes one markdown report. Never touches
the shipped pkl, the DB, FEATURE_COLS, or any tracked data file.

Sections:
  A  Leak-fix impact — coefficients and VIF, pre-fix vs post-fix pipeline (step 3)
  B  Elastic-net grid re-run on FIXED data, against eda_4's pre-registered rule
  C  Before/after LOSO metrics and the list of zeroed coefficients
  D  Betting impact — served-probability change through the real ensemble

Usage:
    .venv/bin/python scripts/validate_elasticnet_change.py
"""
from __future__ import annotations

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

from MLBModel import (  # noqa: E402
    FEATURE_COLS,
    SCALER_WINDOW_START_SEASON,
    assemble_features,
    compute_vif,
    load_data,
)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from verify_sp_leak import build_tgl_as_retrain_does  # noqa: E402

# MLBModel.py does warnings.filterwarnings("ignore") at import; undo so saga's
# ConvergenceWarning is visible rather than silently swallowed.
warnings.resetwarnings()
warnings.simplefilter("default")

RANDOM_STATE = 42
LOSO_SEASONS = [2021, 2022, 2023, 2024, 2025]
YEAR_WEIGHTS = {2021: 0.3, 2022: 1.1, 2023: 1.3, 2024: 1.5, 2025: 1.8, 2026: 1.8}
BASELINE = {"penalty": "l2", "C": 0.5}
OUT_PATH = PROJECT_ROOT / "scripts" / "results" / "elasticnet_penalty_validation.md"
ARTIFACTS_PATH = PROJECT_ROOT / "updates" / "mlb_model_artifacts.pkl"
HOME_PRIOR, RECAL_BLEND = 0.53, 0.04


def fit_lr(X_df, y, seasons, penalty="l2", C=0.5, l1_ratio=None, max_iter=5000):
    """Replicate the shipped fit: scaler on seasons >= SCALER_WINDOW_START_SEASON,
    YEAR_WEIGHTS sample weights, then LR."""
    window = seasons >= SCALER_WINDOW_START_SEASON
    scaler = StandardScaler().fit(X_df[window.values] if window.any() else X_df)
    Xs = scaler.transform(X_df)
    sw = seasons.map(YEAR_WEIGHTS).fillna(1.0).values
    kwargs = {"C": C, "max_iter": max_iter, "random_state": RANDOM_STATE}
    if penalty == "elasticnet":
        kwargs.update(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio)
    lr = LogisticRegression(**kwargs)
    lr.fit(Xs, y, sample_weight=sw)
    return lr, scaler


def loso(model_df, penalty="l2", C=0.5, l1_ratio=None):
    """Per-fold metrics + coefficients, replicating cross_validate_loso's folds."""
    tr_all = model_df[model_df["season"].between(2021, 2025)].dropna(subset=FEATURE_COLS)
    rows, coefs = [], {}
    for s in LOSO_SEASONS:
        tr, va = tr_all[tr_all["season"] != s], tr_all[tr_all["season"] == s]
        lr, sc = fit_lr(tr[FEATURE_COLS], tr["home_win"], tr["season"],
                        penalty=penalty, C=C, l1_ratio=l1_ratio)
        p = lr.predict_proba(sc.transform(va[FEATURE_COLS]))[:, 1]
        rows.append({"season": s, "auc": roc_auc_score(va["home_win"], p),
                     "logloss": log_loss(va["home_win"], p),
                     "brier": brier_score_loss(va["home_win"], p),
                     "n_iter": int(np.max(lr.n_iter_))})
        coefs[s] = pd.Series(lr.coef_[0], index=FEATURE_COLS)
    return pd.DataFrame(rows).set_index("season"), pd.DataFrame(coefs).T


def build_model_df(gate: bool) -> pd.DataFrame:
    tgl, _ = build_tgl_as_retrain_does(gate_injection=gate)
    df, _, _ = load_data(str(PROJECT_ROOT / "Databases_and_logs" / "mlb_allseasons.db"))
    md = assemble_features(df, tgl)
    md[FEATURE_COLS] = md[FEATURE_COLS].fillna(0)
    return md


def md_table(df, fmt="{:.4f}"):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return ("| " + " | ".join(map(str, d.columns)) + " |\n"
            + "|" + "|".join(["---"] * len(d.columns)) + "|\n"
            + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values))


def main() -> int:
    L = ["# Elastic-net penalty + SP leak-fix validation", "",
         f"_Generated {datetime.now().isoformat(timespec='seconds')} by "
         "`scripts/validate_elasticnet_change.py` (report-only)._", ""]

    print("Building pre-fix and post-fix feature sets...")
    md_pre, md_post = build_model_df(gate=False), build_model_df(gate=True)

    # ---------- A: leak-fix impact ----------
    print("A: leak-fix impact on coefficients / VIF...")
    res = {}
    for name, md in (("pre-fix", md_pre), ("post-fix", md_post)):
        d = md.dropna(subset=FEATURE_COLS)
        lr, _ = fit_lr(d[FEATURE_COLS], d["home_win"], d["season"], **BASELINE)
        vif = compute_vif(d, FEATURE_COLS).set_index("feature")["VIF"]
        res[name] = {"coef": pd.Series(lr.coef_[0], index=FEATURE_COLS), "vif": vif}

    comp = pd.DataFrame({
        "coef_pre": res["pre-fix"]["coef"], "coef_post": res["post-fix"]["coef"],
        "coef_delta": res["post-fix"]["coef"] - res["pre-fix"]["coef"],
        "VIF_pre": res["pre-fix"]["vif"], "VIF_post": res["post-fix"]["vif"],
    })
    comp["abs_change"] = comp["coef_delta"].abs()

    L += ["## A — Leak-fix impact on the model (step 3)", "",
          "Both fits use the shipped config (L2, C=0.5, `YEAR_WEIGHTS`, scaler windowed "
          "to 2023+). The only difference is the feature pipeline: pre-fix injects a "
          "current-season SP snapshot into every 2026 row; post-fix resolves 70.7% of "
          "them to retro IDs and uses completed-2025 stats instead.", "",
          "### The SP features specifically", "",
          md_table(comp.loc[["diff_sp_xfip", "diff_sp_siera", "diff_sp_era",
                             "diff_sp_k_bb", "diff_sp_ip_gs"]]
                   .reset_index().rename(columns={"index": "feature"})), "",
          "### All features, largest coefficient change first", "",
          md_table(comp.sort_values("abs_change", ascending=False)
                   .drop(columns="abs_change").reset_index()
                   .rename(columns={"index": "feature"})), ""]

    m_pre, _ = loso(md_pre, **BASELINE)
    m_post, _ = loso(md_post, **BASELINE)
    L += ["### LOSO metrics, pre-fix vs post-fix (same L2 config)", "",
          "Note LOSO folds cover 2021-2025 only, so they exclude 2026 — the season the "
          "leak actually affects. A near-null result here is therefore expected and is "
          "**not** evidence the fix did nothing; it mainly reflects that the scaler "
          "(fit on 2023+, which includes 2026) shifted.", "",
          md_table(pd.concat([m_pre.add_suffix("_pre"), m_post.add_suffix("_post")],
                             axis=1).reset_index()), ""]

    # ---------- B: elastic-net grid on FIXED data ----------
    print("B: re-running the pre-registered elastic-net grid on fixed data...")
    base_m, _ = m_post, None
    grid = []
    for l1r in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for C in [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m, cf = loso(md_post, penalty="elasticnet", C=C, l1_ratio=l1r)
                n_conv = sum(1 for w in caught
                             if issubclass(w.category, ConvergenceWarning))
            grid.append({
                "l1_ratio": l1r, "C": C, "mean_auc": m["auc"].mean(),
                "mean_logloss": m["logloss"].mean(), "mean_brier": m["brier"].mean(),
                "d_logloss": m["logloss"].mean() - base_m["logloss"].mean(),
                "folds_better": int((m["logloss"] <= base_m["logloss"] + 1e-12).sum()),
                "n_zeroed": int((cf.abs() < 1e-6).all(axis=0).sum()),
                "convergence_warnings": n_conv, "max_n_iter": int(m["n_iter"].max()),
            })
    G = pd.DataFrame(grid)
    qual = G[(G["folds_better"] >= 4) & (G["n_zeroed"] >= 1)]
    passes = len(qual) > 0

    L += ["## B — Elastic-net grid, re-run on FIXED data", "",
          "eda_4 selected `l1_ratio=0.3, C=0.01` on leaked data, where SP features were "
          "artificially strong. Same pre-registered rule, re-applied here: **log loss "
          "neutral-or-better in >= 4/5 LOSO folds AND >= 1 coefficient driven to exactly "
          "zero.**", "",
          f"Grid points meeting both conditions: **{len(qual)} of {len(G)}**.", ""]
    if passes:
        # The pre-registered rule is the FILTER; log loss is the tiebreak among
        # configs that pass it. Ranking by n_zeroed instead would pick whichever
        # config discards the most features, which the rule never asked for and
        # which tends to strip real signal (l1=1.0 additionally zeroes sp_siera).
        qs = qual.sort_values("mean_logloss")
        L += ["Qualifying configurations, ranked by log loss (the gate is the filter, "
              "log loss the tiebreak):", "",
              md_table(qs.head(10), "{:.5f}"), ""]
        win = qs.iloc[0]
        L += [f"**Winner: l1_ratio={win['l1_ratio']}, C={win['C']}** — "
              f"{int(win['folds_better'])}/5 folds, {int(win['n_zeroed'])} zeroed, "
              f"mean dlogloss {win['d_logloss']:+.5f}.", ""]
    else:
        L += ["**No configuration qualifies on fixed data.** The eda_4 recommendation "
              "does not survive the leak fix — keep L2.", ""]
        win = None

    orig = G[(G["l1_ratio"] == 0.3) & (G["C"] == 0.01)].iloc[0]
    L += [f"eda_4's specific pick (`l1_ratio=0.3, C=0.01`) on fixed data: "
          f"{int(orig['folds_better'])}/5 folds, {int(orig['n_zeroed'])} zeroed, "
          f"dlogloss {orig['d_logloss']:+.5f} — "
          f"**{'still qualifies' if orig['folds_better'] >= 4 and orig['n_zeroed'] >= 1 else 'NO LONGER qualifies'}**.", "",
          "Full grid:", "", md_table(G, "{:.5f}"), ""]

    # ---------- C: before/after + zeroed coefficients ----------
    if win is not None:
        print("C: before/after with the winning config...")
        d = md_post.dropna(subset=FEATURE_COLS)
        lr_old, sc_old = fit_lr(d[FEATURE_COLS], d["home_win"], d["season"], **BASELINE)
        lr_new, sc_new = fit_lr(d[FEATURE_COLS], d["home_win"], d["season"],
                                penalty="elasticnet", C=float(win["C"]),
                                l1_ratio=float(win["l1_ratio"]))
        c_old = pd.Series(lr_old.coef_[0], index=FEATURE_COLS)
        c_new = pd.Series(lr_new.coef_[0], index=FEATURE_COLS)
        zeroed = c_new.index[c_new.abs() < 1e-6].tolist()
        m_new, _ = loso(md_post, penalty="elasticnet", C=float(win["C"]),
                        l1_ratio=float(win["l1_ratio"]))

        L += ["## C — Before/after on the fixed pipeline", "",
              "### LOSO metrics", "",
              md_table(pd.concat([m_post.add_suffix("_L2"), m_new.add_suffix("_EN")],
                                 axis=1).reset_index()), "",
              f"Mean AUC {m_post['auc'].mean():.5f} -> {m_new['auc'].mean():.5f}; "
              f"mean log loss {m_post['logloss'].mean():.5f} -> "
              f"{m_new['logloss'].mean():.5f}; folds where EN log loss is "
              f"neutral-or-better: **{int((m_new['logloss'] <= m_post['logloss'] + 1e-12).sum())}/5**.",
              "",
              f"### Coefficients driven to exactly zero: **{len(zeroed)}**", "",
              ("- " + "\n- ".join(f"`{z}`" for z in zeroed)) if zeroed else "_(none)_", "",
              "SP features retained (non-zero): "
              + ", ".join(f"`{f}`" for f in FEATURE_COLS
                          if f.startswith("diff_sp_") and f not in zeroed) + ".", "",
              "### Full coefficient comparison", "",
              md_table(pd.DataFrame({"L2_C0.5": c_old, "elasticnet": c_new,
                                     "delta": c_new - c_old})
                       .reindex(c_old.abs().sort_values(ascending=False).index)
                       .reset_index().rename(columns={"index": "feature"})), ""]

        # ---------- D: betting impact ----------
        print("D: betting impact through the real ensemble...")
        with open(ARTIFACTS_PATH, "rb") as f:
            art = pickle.load(f)
        gb, boots = art.get("gb_model"), art.get("xgb_bootstrap_models") or []
        live = d[d["season"] == 2026]
        Xr = live[FEATURE_COLS]
        tree = [gb.predict_proba(Xr)[:, 1]] if gb is not None else []
        if boots:
            tree.append(np.mean(np.vstack([b.predict_proba(Xr)[:, 1] for b in boots]), axis=0))

        def served(lr_model, scaler):
            """Reproduce predict_games_batch: mean(LR, GB, mean-of-bootstrap-XGB),
            then blend 4% toward _HOME_PRIOR. LR uses scaled features, trees raw."""
            p_lr = lr_model.predict_proba(scaler.transform(Xr))[:, 1]
            p = np.mean(np.vstack([p_lr] + tree), axis=0)
            return (1 - RECAL_BLEND) * p + RECAL_BLEND * HOME_PRIOR

        s_old, s_new = served(lr_old, sc_old), served(lr_new, sc_new)
        e_old, e_new = np.abs(s_old - 0.5), np.abs(s_new - 0.5)
        flips = int(((s_old > 0.5) != (s_new > 0.5)).sum())

        L += ["## D — Betting impact", "",
              "Edge is `model_prob - devigged_market_prob` on the picked side "
              "(`Main/app.py:1583-1590`), so a shift in the served probability moves the "
              "edge one-for-one. The served probability is the mean of LR, GB and the "
              "50-model bootstrap-XGB mean, then blended 4% toward `_HOME_PRIOR = 0.53` "
              "(`predict_games_batch`) — GB/XGB are held fixed from the shipped pkl, so "
              "this isolates the penalty change.", "",
              f"Evaluated on {len(live)} game rows from 2026.", "",
              md_table(pd.DataFrame({
                  "metric": ["sd of served prob", "mean |p-0.5|", "p90 |p-0.5|",
                             "max |p-0.5|", "mean |Δ served prob|", "max |Δ served prob|"],
                  "L2": [s_old.std(), e_old.mean(), np.quantile(e_old, 0.9), e_old.max(),
                         np.nan, np.nan],
                  "elasticnet": [s_new.std(), e_new.mean(), np.quantile(e_new, 0.9),
                                 e_new.max(), np.abs(s_new - s_old).mean(),
                                 np.abs(s_new - s_old).max()],
              }), "{:.5f}"), "",
              f"- predicted winner flips on **{flips}/{len(live)}** games "
              f"({flips / len(live):.1%}).",
              f"- mean shift in |p-0.5| (i.e. in edge magnitude): "
              f"**{e_new.mean() - e_old.mean():+.5f}**.", "",
              "**Limitation — true edges could not be computed locally.** The local DB's "
              "2026 games stop at 2026-07-07, while the only rows carrying stored odds in "
              "`predictions_log.json` run 2026-07-16 to 2026-08-13 — zero overlap. So the "
              "count of bets crossing `GOOD_EDGE = 0.05` cannot be measured here without "
              "refreshing the DB (CLAUDE.md rule 6: the local clone drifts behind "
              "production). The probability-shift figures above bound the effect: edge "
              "moves one-for-one with the served probability, so a mean shift of "
              f"{abs(e_new.mean() - e_old.mean()):.5f} is the scale of the change to expect.",
              ""]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(L))
    print(f"\nreport -> {OUT_PATH.relative_to(PROJECT_ROOT)}")

    print("\n=== SP features, pre-fix -> post-fix (L2 C=0.5) ===")
    print(comp.loc[["diff_sp_xfip", "diff_sp_siera", "diff_sp_era", "diff_sp_k_bb"],
                   ["coef_pre", "coef_post", "coef_delta", "VIF_pre", "VIF_post"]]
          .round(4).to_string())
    if win is not None:
        print(f"\n=== Elastic net winner: l1_ratio={win['l1_ratio']}, C={win['C']} ===")
        print(f"  zeroed ({len(zeroed)}): {zeroed}")
        print(f"  AUC {m_post['auc'].mean():.5f} -> {m_new['auc'].mean():.5f}   "
              f"logloss {m_post['logloss'].mean():.5f} -> {m_new['logloss'].mean():.5f}")
        print(f"  served-prob mean |Δ| {np.abs(s_new - s_old).mean():.5f}, "
              f"winner flips {flips}/{len(live)}")
    else:
        print("\n=== No elastic-net config qualifies on fixed data — keep L2 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
