#!/usr/bin/env python3
"""
Step 2: data-integrity check on the Phase 4a prior-season SP substitution.

For historical team-game rows (2021-2025), reports:
  - % of rows that fall back to league-average because the pitcher has no (pid, season-1)
    row in pitcher_stats — overall and per season.
  - mean / median / std of the sp_era feature under the OLD (same-season) vs NEW
    (prior-season) lookup.

Granularity = tgl rows (one per team per game = the row the feature is attached to).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Main"))

import numpy as np
import pandas as pd
from MLBModel import load_data, build_team_game_log, DB_PATH


def main():
    df, pitcher_stats, _ = load_data(DB_PATH)
    tgl = build_team_game_log(df)

    ps = pitcher_stats.sort_values("games_started", ascending=False).drop_duplicates(
        subset=["retro_pitcher_id", "season"], keep="first")
    sp_era = ps.set_index(["retro_pitcher_id", "season"])["era"].to_dict()
    league_era = pitcher_stats.groupby("season")["era"].mean().to_dict()
    overall_era = float(pitcher_stats["era"].mean())

    hist = tgl[tgl["season"].between(2021, 2025)].copy()

    old_vals, new_vals, fell_back, seasons = [], [], [], []
    for _, r in hist.iterrows():
        pid, s = r["starting_pitcher_id"], int(r["season"])
        cur, prev = (pid, s), (pid, s - 1)
        # OLD (leaked): same-season first, else prior, else league avg
        old = sp_era.get(cur, sp_era.get(prev, league_era.get(s, overall_era)))
        # NEW (fixed): prior only, else league avg
        if prev in sp_era:
            new = sp_era[prev]; fb = 0
        else:
            new = league_era.get(s - 1, league_era.get(s, overall_era)); fb = 1
        old_vals.append(old); new_vals.append(new); fell_back.append(fb); seasons.append(s)

    hist = hist.assign(_old=old_vals, _new=new_vals, _fb=fell_back, _s=seasons)
    n = len(hist)
    fb_total = int(hist["_fb"].sum())

    lines = []
    lines.append(f"Historical tgl rows (2021-2025): {n}")
    lines.append(f"Fell back to league-average (no prior-season row): "
                 f"{fb_total} ({100*fb_total/n:.1f}%)\n")

    lines.append("Per-season fallback rate:")
    lines.append(f"{'season':>7} | {'rows':>6} | {'fallback':>9} | {'pct':>6}")
    lines.append("-" * 36)
    for s in sorted(hist["_s"].unique()):
        sub = hist[hist["_s"] == s]
        lines.append(f"{s:>7} | {len(sub):>6} | {int(sub['_fb'].sum()):>9} | "
                     f"{100*sub['_fb'].mean():>5.1f}%")

    def stats(col):
        v = hist[col].astype(float)
        return v.mean(), v.median(), v.std()

    om, omd, osd = stats("_old")
    nm, nmd, nsd = stats("_new")
    lines.append("\nsp_era feature distribution (OLD same-season vs NEW prior-season):")
    lines.append(f"{'variant':>18} | {'mean':>7} | {'median':>7} | {'std':>7}")
    lines.append("-" * 48)
    lines.append(f"{'OLD (same-season)':>18} | {om:>7.3f} | {omd:>7.3f} | {osd:>7.3f}")
    lines.append(f"{'NEW (prior-season)':>18} | {nm:>7.3f} | {nmd:>7.3f} | {nsd:>7.3f}")
    lines.append(f"{'delta (NEW-OLD)':>18} | {nm-om:>+7.3f} | {nmd-omd:>+7.3f} | {nsd-osd:>+7.3f}")

    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    main()
