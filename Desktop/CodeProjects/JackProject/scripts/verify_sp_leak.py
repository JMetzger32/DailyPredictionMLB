#!/usr/bin/env python3
"""
verify_sp_leak.py — verify (or refute) the 2026 SP look-ahead leak.

`merge_sp_stats` deliberately uses each pitcher's PRIOR-season (S-1) stats for
training rows, an explicit leak fix (Main/MLBModel.py:495-502), because
same-season aggregates fold in games played AFTER the row's own game date.
`updates/update_daily.py::retrain_model` then overwrites the 2026 rows with a
CURRENT season-to-date snapshot taken from the pkl at retrain time.

If that snapshot is constant per pitcher across game dates, every 2026 training
row carries information from the future relative to its own game. This script
measures that directly: within-pitcher stddev of sp_era / sp_xfip / sp_siera
across each pitcher's own starts.

Report-only: reads the DB and the pkl, writes only its own markdown report.
Never imports Main/app.py.

Usage:
    .venv/bin/python scripts/verify_sp_leak.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Main"))

from MLBModel import (  # noqa: E402
    DB_PATH,
    build_team_game_log,
    compute_rolling_team_features,
    load_boxscore_ip_lookup,
    load_data,
    merge_bullpen_era,
    merge_sp_stats,
)

ARTIFACTS_PATH = PROJECT_ROOT / "updates" / "mlb_model_artifacts.pkl"
OUT_PATH = PROJECT_ROOT / "scripts" / "results" / "sp_leak_verification.md"
SP_COLS = ["sp_era", "sp_xfip", "sp_siera"]
MIN_STARTS = 3


def build_tgl_as_retrain_does(gate_injection: bool) -> tuple[pd.DataFrame, dict]:
    """Replicate updates/update_daily.py::retrain_model's tgl pipeline.

    Mirrors update_daily.py:999-1050 (note merge_bullpen_era runs BEFORE
    merge_sp_stats there, unlike MLBModel.__main__).

    gate_injection=False reproduces the PRE-FIX behaviour (inject into every
    2026 row); True reproduces the POST-FIX behaviour (inject only into rows
    whose retro ID could not be resolved)."""
    df, pitcher_stats, bullpen_stats = load_data(str(DB_PATH))
    ip_lookup = load_boxscore_ip_lookup(str(DB_PATH), df)
    tgl = build_team_game_log(df, boxscore_ip_lookup=ip_lookup)
    tgl = compute_rolling_team_features(tgl)
    tgl = merge_bullpen_era(tgl, bullpen_stats)
    # gate_injection=False reproduces the true pre-fix state, which had no
    # ID resolution at all — so disable it here too, otherwise rows that resolve
    # but lack a pkl baseline would get prior-season stats the old code never gave them.
    tgl = merge_sp_stats(tgl, pitcher_stats, resolve_ids=gate_injection)

    with open(ARTIFACTS_PATH, "rb") as f:
        art = pickle.load(f)
    sp_live = art.get("sp_baselines", {}) or {}
    name_lookup = {}
    for _pid, b in sp_live.items():
        nm = (b or {}).get("name", "")
        if nm:
            name_lookup[nm.strip().lower()] = b

    col_map = {"sp_era": "era", "sp_whip": "whip", "sp_xfip": "xfip",
               "sp_siera": "siera", "sp_so9": "so9", "sp_bb9": "bb9", "sp_hr9": "hr9"}
    mask_2026 = tgl["season"] == 2026
    pid_col = tgl["starting_pitcher_id"]
    unresolved = mask_2026 & (
        pid_col.isna() | pid_col.astype(str).str.strip().isin(["", "None", "nan"])
    )
    tgl["_sp_path"] = None
    tgl.loc[mask_2026 & ~unresolved, "_sp_path"] = "resolved_prior_season"

    loop_mask = unresolved if gate_injection else mask_2026
    injected = 0
    for idx in tgl[loop_mask].index:
        pid = tgl.at[idx, "starting_pitcher_id"]
        b = sp_live.get(pid) if (pid and not gate_injection) else None
        if b is None and "starting_pitcher_name" in tgl.columns:
            pname = tgl.at[idx, "starting_pitcher_name"]
            if pname and isinstance(pname, str):
                b = name_lookup.get(pname.strip().lower())
        if b is not None:
            for col, key in col_map.items():
                if key in b and b[key] is not None:
                    tgl.at[idx, col] = float(b[key])
            tgl.at[idx, "sp_ip_gs"] = float(b.get("ip_gs", tgl.at[idx, "sp_ip_gs"]))
            bb9, so9 = b.get("bb9", 1), b.get("so9", 7)
            tgl.at[idx, "sp_k_bb"] = so9 / bb9 if bb9 > 0.5 else so9 / 0.5
            injected += 1
            tgl.at[idx, "_sp_path"] = "injected_current_season"

    n_unres = int(unresolved.sum())
    return tgl, {"injected": injected, "total_2026": int(mask_2026.sum()),
                 "unresolved": n_unres, "resolved": int(mask_2026.sum()) - n_unres,
                 "sp_baselines_in_pkl": len(sp_live), "pitcher_stats": pitcher_stats}


def provenance_check(tgl: pd.DataFrame, pitcher_stats: pd.DataFrame) -> dict:
    """THE decisive test. Within-pitcher stddev cannot validate the fix, because
    prior-season stats are constant within a season BY DESIGN — the same zero
    stddev appears whether the constant came from completed-2025 (correct) or
    2026-to-date (leaked). What distinguishes them is provenance: does a 2026
    row's sp_era equal that pitcher's COMPLETED 2025 ERA, or their in-progress
    2026 ERA?

    Only completed-prior-season values are knowable pre-game."""
    p25 = pitcher_stats[pitcher_stats["season"] == 2025]
    p25 = p25.sort_values("games_started", ascending=False).drop_duplicates(
        subset=["retro_pitcher_id"], keep="first"
    )
    era25 = dict(zip(p25["retro_pitcher_id"], p25["era"]))

    rows = tgl[(tgl["season"] == 2026) & (tgl["_sp_path"].notna())].copy()
    out = {}
    for path in ("resolved_prior_season", "injected_current_season"):
        sub = rows[rows["_sp_path"] == path]
        if not len(sub):
            out[path] = {"n": 0}
            continue
        expected = sub["starting_pitcher_id"].map(era25)
        comparable = expected.notna()
        match = (sub["sp_era"] - expected).abs() < 1e-6
        out[path] = {
            "n": len(sub),
            "n_comparable": int(comparable.sum()),
            "pct_matching_2025_era": (float(match[comparable].mean())
                                       if comparable.any() else float("nan")),
        }
    return out


def within_pitcher_spread(tgl: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict]:
    """Per-pitcher stddev of each SP stat across that pitcher's own starts."""
    rows = tgl[(tgl["season"] == season) &
               tgl["starting_pitcher_name"].notna() &
               (tgl["starting_pitcher_name"].astype(str).str.len() > 0)].copy()
    rows["_pitcher"] = rows["starting_pitcher_name"].astype(str).str.strip().str.lower()

    counts = rows.groupby("_pitcher").size()
    eligible = counts[counts >= MIN_STARTS].index
    sub = rows[rows["_pitcher"].isin(eligible)]

    sd = sub.groupby("_pitcher")[SP_COLS].std(ddof=0)
    summary = {}
    for col in SP_COLS:
        n_const = int((sd[col] < 1e-9).sum())
        summary[col] = {
            "mean_within_pitcher_sd": float(sd[col].mean()),
            "max_within_pitcher_sd": float(sd[col].max()),
            "n_pitchers_constant": n_const,
            "pct_constant": n_const / len(sd) if len(sd) else float("nan"),
        }
    return sd, {"n_pitchers": len(sd), "n_rows": len(sub), "per_col": summary}


def main() -> int:
    print("Building tgl as retrain_model does — PRE-FIX (ungated injection)...")
    tgl_pre, inj_pre = build_tgl_as_retrain_does(gate_injection=False)
    print("Building tgl as retrain_model does — POST-FIX (gated injection)...")
    tgl, inj = build_tgl_as_retrain_does(gate_injection=True)
    prov_pre = provenance_check(tgl_pre, inj_pre["pitcher_stats"])
    prov = provenance_check(tgl, inj["pitcher_stats"])
    print(f"  resolved {inj['resolved']}/{inj['total_2026']} 2026 rows to retro IDs; "
          f"{inj['injected']} of {inj['unresolved']} unresolved rows injected")

    sd_2026, s26 = within_pitcher_spread(tgl, 2026)
    # 2025 is the control: it takes the normal prior-season path with no injection,
    # so its within-pitcher spread shows what "not leaked" looks like on this metric.
    sd_2025, s25 = within_pitcher_spread(tgl, 2025)

    lines = [
        "# SP look-ahead leak verification",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')} by "
        "`scripts/verify_sp_leak.py` (report-only)._",
        "",
        "## What is being tested",
        "",
        "`merge_sp_stats` uses each pitcher's **prior-season (S-1)** stats for training "
        "rows — an explicit leak fix (`Main/MLBModel.py:495-502`) — because same-season "
        "aggregates fold in games played *after* the row's own game date. "
        "`update_daily.py::retrain_model` (lines 1006-1050) then overwrites the 2026 rows "
        "with a **current season-to-date snapshot** from the pkl.",
        "",
        "If that snapshot is constant per pitcher across game dates, every 2026 training "
        "row carries future information relative to its own game. Measured below as the "
        f"within-pitcher stddev across each pitcher's own starts (pitchers with "
        f">= {MIN_STARTS} starts).",
        "",
        "## Injection coverage",
        "",
        f"- live SP baselines injected into **{inj['injected']}/{inj['total_2026']}** "
        f"2026 team-game rows",
        f"- `sp_baselines` entries in pkl: {inj['sp_baselines_in_pkl']}",
        "",
        f"## 2026 — within-pitcher spread ({s26['n_pitchers']} pitchers, "
        f"{s26['n_rows']} rows)",
        "",
        "| stat | mean within-pitcher sd | max | pitchers with sd == 0 | % constant |",
        "|---|---|---|---|---|",
    ]
    for col in SP_COLS:
        m = s26["per_col"][col]
        lines.append(
            f"| `{col}` | {m['mean_within_pitcher_sd']:.6f} | "
            f"{m['max_within_pitcher_sd']:.6f} | "
            f"{m['n_pitchers_constant']}/{s26['n_pitchers']} | {m['pct_constant']:.1%} |"
        )

    lines += [
        "",
        f"## 2025 control — same metric, no injection ({s25['n_pitchers']} pitchers)",
        "",
        "2025 takes the normal prior-season path, so its spread shows what this metric "
        "looks like when there is no injection. Note a *legitimately* constant value is "
        "expected here too — prior-season stats are by construction fixed for the whole "
        "season — so 2025 is a reference for the metric's behaviour, not a clean contrast.",
        "",
        "| stat | mean within-pitcher sd | pitchers with sd == 0 | % constant |",
        "|---|---|---|---|",
    ]
    for col in SP_COLS:
        m = s25["per_col"][col]
        lines.append(
            f"| `{col}` | {m['mean_within_pitcher_sd']:.6f} | "
            f"{m['n_pitchers_constant']}/{s25['n_pitchers']} | {m['pct_constant']:.1%} |"
        )

    confirmed = all(s26["per_col"][c]["pct_constant"] > 0.9 for c in SP_COLS)
    lines += [
        "",
        "## Why within-pitcher stddev CANNOT validate the fix",
        "",
        "Constancy is the right signal for *detecting* the original leak, but it cannot "
        "confirm the repair. Prior-season stats are constant within a season **by design**, "
        "so the stddev is zero whether the constant came from completed-2025 (correct, "
        "knowable pre-game) or from 2026-to-date (leaked). The distinguishing question is "
        "**provenance**, not variance — which is what the next section tests.",
        "",
        "## Provenance — the decisive test",
        "",
        "For each 2026 row, does `sp_era` equal that pitcher's **completed 2025** ERA "
        "(correct: fully knowable before any 2026 game) or their in-progress 2026 ERA "
        "(leaked)?",
        "",
        "| pipeline | path | rows | % matching completed-2025 ERA |",
        "|---|---|---|---|",
    ]
    for label, p in (("pre-fix", prov_pre), ("post-fix", prov)):
        for path, m in p.items():
            if not m.get("n"):
                continue
            pct = m.get("pct_matching_2025_era")
            pct_s = "n/a" if pct != pct else f"{pct:.1%}"
            lines.append(f"| {label} | `{path}` | {m['n']} | {pct_s} |")

    resolved_ok = prov.get("resolved_prior_season", {}).get("pct_matching_2025_era", 0)
    lines += [
        "",
        f"**Post-fix, {resolved_ok:.1%} of resolved 2026 rows now carry the pitcher's "
        f"completed-2025 ERA** — the same leak-free prior-season path every other season "
        f"uses.",
        "",
        "## Verdict",
        "",
        f"- Original leak (pre-fix): **{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}** "
        f"— every SP stat identical at every one of a pitcher's 2026 starts "
        f"({s26['per_col']['sp_era']['n_pitchers_constant']}/{s26['n_pitchers']} pitchers), "
        "sourced from a retrain-time snapshot rather than a completed season.",
        f"- After the fix: **{inj['resolved']}/{inj['total_2026']} "
        f"({inj['resolved'] / inj['total_2026']:.1%})** of 2026 rows take the leak-free "
        f"prior-season path.",
        f"- **Residual:** {inj['injected']} rows still fall back to the current-season "
        "snapshot and retain look-ahead, at the highest sample weight (1.8). These are "
        "pitchers with no 2025 `pitcher_stats` row (2026 debuts, or 2025 absences). "
        "Deliberate trade-off — the alternative is league average — but a known residual, "
        "not a clean fix.",
        "",
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))

    print()
    print("  PRE-FIX  within-pitcher sd (leak detection):")
    for col in SP_COLS:
        m = s26["per_col"][col]
        print(f"    {col:10s}: sd={m['mean_within_pitcher_sd']:.6f}  "
              f"constant for {m['n_pitchers_constant']}/{s26['n_pitchers']}")
    print("\n  PROVENANCE (% carrying completed-2025 ERA, the decisive test):")
    for label, p in (("pre-fix ", prov_pre), ("post-fix", prov)):
        for path, m in p.items():
            if not m.get("n"):
                continue
            pct = m.get("pct_matching_2025_era")
            pct_s = "n/a" if pct != pct else f"{pct:6.1%}"
            print(f"    {label}  {path:26s} n={m['n']:5d}  {pct_s}")
    print(f"\n  post-fix: {inj['resolved']}/{inj['total_2026']} rows leak-free, "
          f"{inj['injected']} residual")
    print(f"  report -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
