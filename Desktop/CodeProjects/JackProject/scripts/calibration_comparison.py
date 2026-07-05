#!/usr/bin/env python3
"""
Step 1: calibration deep-dive.

Produces 10-bin reliability tables (bin, predicted mid, actual home-win rate, count) + ECE
for THREE views:

  A) 2025 holdout, LEAKED features   (SP/bullpen same-season aggregates; train 2021-2024)
  B) 2025 holdout, LEAK-FIXED features (SP/bullpen prior-season S-1;    train 2021-2024)
  C) 2026 same-games approximation   (leak-fixed model trained 2021-2025, scored on 2026 rows)
  D) Live-2026 reference             (OLD model's real predictions from predictions_log.json)

A vs B is the apples-to-apples verdict (same 2025 rows). C/D compare on the same 2026 games,
with the caveat that C uses pipeline prior-season SP for 2026 while D used the live snapshot.

All binning is standard reliability: bin home_win_prob into 10 even bins, report the actual
home-win rate per bin. Ensemble = mean(LR_scaled, GB_raw), matching the production retrain.
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Main"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from MLBModel import (
    load_data, build_team_game_log, compute_rolling_team_features,
    merge_bullpen_era, assemble_features, FEATURE_COLS, DB_PATH, RANDOM_STATE,
)

LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")


# ---------------------------------------------------------------------------
# SP + bullpen merge in two modes (leaked = same-season first, fixed = prior-season)
# Mirrors MLBModel.merge_sp_stats / merge_bullpen_era; the only difference is lookup order.
# ---------------------------------------------------------------------------
def merge_sp_two_ways(tgl, pitcher_stats, mode):
    has_xfip = "xfip" in pitcher_stats.columns
    has_siera = "siera" in pitcher_stats.columns
    stat_cols = ["era", "whip", "so9", "bb9", "hr9", "innings_pitched", "games_started"]
    if has_xfip:  stat_cols.append("xfip")
    if has_siera: stat_cols.append("siera")

    ps = pitcher_stats.sort_values("games_started", ascending=False).drop_duplicates(
        subset=["retro_pitcher_id", "season"], keep="first")
    sp_lookup = ps.set_index(["retro_pitcher_id", "season"])[stat_cols].to_dict("index")
    league_avg = pitcher_stats.groupby("season")[stat_cols].mean().to_dict("index")
    overall_avg = pitcher_stats[stat_cols].mean().to_dict()

    cols = {k: [] for k in ["sp_era", "sp_whip", "sp_xfip", "sp_siera", "sp_so9",
                            "sp_bb9", "sp_hr9", "sp_ip_gs", "sp_k_bb"]}
    for _, row in tgl.iterrows():
        pid, season = row["starting_pitcher_id"], row["season"]
        cur, prev = (pid, season), (pid, season - 1)
        if mode == "leaked":
            stats = sp_lookup.get(cur) or sp_lookup.get(prev) or league_avg.get(season, overall_avg)
        else:  # fixed: prior season only, then league avg (no same-season)
            stats = sp_lookup.get(prev) or league_avg.get(season - 1, league_avg.get(season, overall_avg))
        cols["sp_era"].append(stats.get("era", overall_avg["era"]))
        cols["sp_whip"].append(stats.get("whip", overall_avg["whip"]))
        cols["sp_so9"].append(stats.get("so9", overall_avg["so9"]))
        cols["sp_bb9"].append(stats.get("bb9", overall_avg["bb9"]))
        cols["sp_hr9"].append(stats.get("hr9", overall_avg["hr9"]))
        cols["sp_xfip"].append(stats.get("xfip", stats.get("era", overall_avg["era"])))
        cols["sp_siera"].append(stats.get("siera", stats.get("era", overall_avg["era"])))
        ip = stats.get("innings_pitched", overall_avg.get("innings_pitched", 0))
        gs = stats.get("games_started", overall_avg.get("games_started", 1))
        cols["sp_ip_gs"].append(ip / gs if gs > 0 else 5.8)
        bb9 = stats.get("bb9", overall_avg["bb9"]); so9 = stats.get("so9", overall_avg["so9"])
        cols["sp_k_bb"].append(so9 / bb9 if bb9 > 0.5 else so9 / 0.5)
    for c, v in cols.items():
        tgl[c] = pd.to_numeric(pd.Series(v, index=tgl.index), errors="coerce")
        tgl[c] = tgl[c].fillna(tgl[c].mean())
    return tgl


def merge_bullpen_two_ways(tgl, bullpen_stats, mode):
    if bullpen_stats is None or len(bullpen_stats) == 0:
        tgl["bullpen_era"] = 4.20
        return tgl
    bp = bullpen_stats.set_index(["retro_team", "season"])["bullpen_era"].to_dict()
    if mode == "leaked":
        tgl["bullpen_era"] = tgl.apply(
            lambda r: bp.get((r["team"], r["season"]), bp.get((r["team"], r["season"] - 1), 4.20)), axis=1)
    else:
        tgl["bullpen_era"] = tgl.apply(
            lambda r: bp.get((r["team"], r["season"] - 1), 4.20), axis=1)
    return tgl


def build_model_df(mode):
    df, pitcher_stats, bullpen_stats = load_data(DB_PATH)
    tgl = build_team_game_log(df)
    tgl = compute_rolling_team_features(tgl)
    tgl = merge_bullpen_two_ways(tgl, bullpen_stats, mode)
    tgl = merge_sp_two_ways(tgl, pitcher_stats, mode)
    return assemble_features(df, tgl)


# ---------------------------------------------------------------------------
def reliability(probs, actuals):
    """Return (rows, ece). rows: list of (lo,hi,mid,actual,count). ece weighted by count."""
    bins = [{"lo": i * 10, "hi": i * 10 + 10, "mid": (i * 10 + 5) / 100, "w": 0, "n": 0}
            for i in range(10)]
    for p, a in zip(probs, actuals):
        idx = min(int(p * 10), 9)
        bins[idx]["n"] += 1
        bins[idx]["w"] += int(a)
    total = sum(b["n"] for b in bins)
    rows, ece = [], 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        act = b["w"] / b["n"]
        ece += abs(act - b["mid"]) * b["n"] / total
        rows.append((b["lo"], b["hi"], b["mid"], act, b["n"]))
    return rows, ece


def fmt_table(title, rows, ece):
    out = [f"{title}", f"{'bin':>9} | {'pred_mid':>8} | {'actual':>7} | {'count':>6}", "-" * 40]
    for lo, hi, mid, act, n in rows:
        out.append(f"{lo:>3}-{hi:<3}% | {mid:>8.2f} | {act:>7.3f} | {n:>6}")
    out.append(f"ECE = {ece:.4f}   (lower = better calibrated)")
    return "\n".join(out)


def train_and_score(train_df, test_df, feats):
    tr = train_df.dropna(subset=feats)
    te = test_df.dropna(subset=feats)
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[feats]); Xte = sc.transform(te[feats])
    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE).fit(Xtr, tr["home_win"])
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=RANDOM_STATE).fit(tr[feats], tr["home_win"])
    ens = (lr.predict_proba(Xte)[:, 1] + gb.predict_proba(te[feats])[:, 1]) / 2.0
    return ens, te["home_win"].values


def main():
    feats = list(FEATURE_COLS)
    blocks = []

    # A + B: 2025 holdout, leaked vs leak-fixed (same rows)
    df_leak = build_model_df("leaked")
    df_fix = build_model_df("fixed")
    pA, aA = train_and_score(df_leak[df_leak.season.between(2021, 2024)],
                             df_leak[df_leak.season == 2025], feats)
    pB, aB = train_and_score(df_fix[df_fix.season.between(2021, 2024)],
                             df_fix[df_fix.season == 2025], feats)
    rA, eA = reliability(pA, aA)
    rB, eB = reliability(pB, aB)
    blocks.append(fmt_table("A) 2025 HOLDOUT — LEAKED (same-season SP/bullpen)", rA, eA))
    blocks.append(fmt_table("B) 2025 HOLDOUT — LEAK-FIXED (prior-season SP/bullpen)", rB, eB))

    # C: 2026 same-games approximation (leak-fixed, train 2021-2025, score 2026)
    tr_c = df_fix[df_fix.season.between(2021, 2025)]
    te_c = df_fix[df_fix.season == 2026]
    if len(te_c.dropna(subset=feats)) > 0:
        pC, aC = train_and_score(tr_c, te_c, feats)
        rC, eC = reliability(pC, aC)
        blocks.append(fmt_table("C) 2026 GAMES — LEAK-FIXED approximation "
                                "(pipeline prior-season SP, NOT the live snapshot)", rC, eC))
    else:
        blocks.append("C) 2026 GAMES — LEAK-FIXED approximation: no complete-feature 2026 rows")

    # D: Live-2026 reference from predictions_log
    log = json.load(open(LOG))
    live = [e for day in log.values() for e in day
            if e.get("correct") is not None and e.get("game_type") == "R"
            and e.get("post_game_created") is not True and e.get("home_win_prob") is not None]
    pD = np.array([e["home_win_prob"] for e in live])
    aD = np.array([1 if e["actual_winner"] == "Home" else 0 for e in live])
    rD, eD = reliability(pD, aD)
    blocks.append(fmt_table("D) LIVE 2026 REFERENCE — OLD model real predictions (predictions_log)", rD, eD))

    verdict = (f"VERDICT (A vs B, same 2025 rows): ECE leaked={eA:.4f} -> leak-fixed={eB:.4f}  "
               f"({'IMPROVED' if eB < eA else 'WORSE' if eB > eA else 'UNCHANGED'} "
               f"by {abs(eB-eA):.4f}).")

    report = "\n\n".join(blocks) + "\n\n" + verdict
    print(report)
    return report, dict(eA=eA, eB=eB, eD=eD)


if __name__ == "__main__":
    main()
