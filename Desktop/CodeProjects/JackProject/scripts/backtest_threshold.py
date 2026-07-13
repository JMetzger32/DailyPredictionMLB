"""
backtest_threshold.py
---------------------
Backtests different edge thresholds against historical betting_log data
to find the ROI-maximizing threshold for "value bet" classification.

Usage:
    python3 scripts/backtest_threshold.py [--min-sample N] [--stake DOLLARS]

Data source: SQLite betting_log table in Databases_and_logs/mlb_allseasons.db
(deviation from the original pseudocode, which read betting_log.json — the DB
is the authoritative store; JSON is only its backup). Falls back to the JSON
if the table is unreadable.

This script only REPORTS. It never edits the live threshold in Main/app.py
(currently 0.05) — changing that is a separate, deliberate decision.
"""
import argparse
import json
import os
import sqlite3
import sys
from statistics import mean, stdev

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB   = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
_JSON = os.path.join(_ROOT, "Databases_and_logs", "betting_log.json")

# Below ~200 resolved bets the ROI ranking is mostly variance; still print the
# table, but say loudly that it isn't decision-grade.
DECISION_GRADE_N = 200


def load_resolved():
    """Resolved regular-season entries with an edge and odds, oldest first."""
    try:
        conn = sqlite3.connect(_DB)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT date, model_edge, predicted_team_ml, correct FROM betting_log "
            "WHERE correct IS NOT NULL AND model_edge IS NOT NULL "
            "AND predicted_team_ml IS NOT NULL AND game_type != 'S' ORDER BY date")]
        conn.close()
        return rows
    except Exception as e:
        print(f"[warn] SQLite load failed ({e}); falling back to {_JSON}")
        data = json.load(open(_JSON))
        rows = [e for day in data.values() for e in day
                if e.get("correct") is not None and e.get("model_edge") is not None
                and e.get("predicted_team_ml") is not None and e.get("game_type") != "S"]
        return sorted(rows, key=lambda e: e["date"])


def pl_for(ml, won, stake):
    if won:
        return stake * (ml / 100 if ml >= 0 else 100 / abs(ml))
    return -stake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sample", type=int, default=30,
                    help="skip thresholds with fewer bets than this (default 30)")
    ap.add_argument("--stake", type=float, default=10.0)
    args = ap.parse_args()

    resolved = load_resolved()
    print(f"Resolved bets with edge + odds: {len(resolved)}")
    if not resolved:
        print("\nNo usable data yet. Odds+result pairs only start accumulating once "
              "GITHUB_TOKEN-backed log persistence is live (see OPERATIONS.md).")
        return 1
    if len(resolved) < DECISION_GRADE_N:
        print(f"[!] Below the ~{DECISION_GRADE_N}-bet bar for a decision-grade result — "
              "treat everything below as directional only.\n")

    results = []
    for t in range(1, 16):                       # thresholds 0.01 .. 0.15
        T = round(t * 0.01, 2)
        bets = [e for e in resolved if e["model_edge"] >= T]
        if len(bets) < args.min_sample:
            continue
        pl_list = [pl_for(b["predicted_team_ml"], b["correct"], args.stake) for b in bets]
        wins   = sum(1 for b in bets if b["correct"])
        net_pl = sum(pl_list)
        results.append({
            "threshold": T,
            "bets":      len(bets),
            "win_pct":   wins / len(bets),
            "roi":       net_pl / (len(bets) * args.stake),
            "sharpe":    (mean(pl_list) / stdev(pl_list)) if len(pl_list) > 1 and stdev(pl_list) > 0 else None,
            "net_pl":    net_pl,
        })

    if not results:
        print(f"No threshold reached min sample of {args.min_sample} bets "
              f"(largest pool: {len(resolved)} at threshold 0.01). Wait for more data.")
        return 1

    print(f"{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Sharpe':>8} {'Net P/L':>10}")
    for r in results:
        print(f"{r['threshold']:>10.2f} {r['bets']:>6} {r['win_pct']:>7.1%} "
              f"{r['roi']:>8.1%} {r['sharpe'] or 0:>8.2f} ${r['net_pl']:>9.2f}")

    best = max(results, key=lambda r: r["sharpe"] or 0)   # risk-adjusted, not raw ROI
    print(f"\nBest by Sharpe: threshold {best['threshold']:.2f} "
          f"(Sharpe={best['sharpe']:.2f}, ROI={best['roi']:.1%}, N={best['bets']})")
    print("Current live threshold: 0.05 — do NOT change it from this script; "
          "review the table and decide explicitly.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot([r["threshold"] for r in results], [r["roi"] for r in results], "b-o", label="ROI")
        ax1.axvline(0.05, color="gray", linestyle="--", label="Current (5%)")
        ax1.axvline(best["threshold"], color="green", linestyle="--",
                    label=f"Best Sharpe ({best['threshold']:.0%})")
        ax1.set_xlabel("Edge Threshold"); ax1.set_ylabel("ROI", color="b")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.bar([r["threshold"] for r in results], [r["bets"] for r in results],
                width=0.006, alpha=0.3, color="orange")
        ax2.set_ylabel("Number of Bets", color="orange")
        plt.title("Edge Threshold Calibration — ROI vs Sample Size")
        fig.tight_layout()
        out = os.path.join(_ROOT, "scripts", "threshold_calibration.png")
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
    except Exception as e:
        print(f"[plot skipped: {e}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
