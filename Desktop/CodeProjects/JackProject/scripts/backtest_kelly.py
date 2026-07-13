"""
backtest_kelly.py
-----------------
Compares stake-sizing strategies on historical resolved value bets:
    flat $10  vs  quarter-Kelly (the production default on /api/betting)
              vs  half-Kelly   (the original skeleton's suggestion)

Usage:
    python3 scripts/backtest_kelly.py [--bankroll 100] [--cap-pct 0.05]

Kelly:  f* = (b*p - q) / b   (b = net profit per $1, p = model win prob)
Stakes are flat off a fixed bankroll (non-compounding), capped at
cap-pct * bankroll — same math as _kelly_stake() in Main/app.py; keep in sync.

Data source: SQLite betting_log (deviation from the pseudocode's
betting_log.json — the DB is authoritative). Value bets only (bet_rating='good').
This script only reports; it changes nothing.
"""
import argparse
import os
import sqlite3
import sys
from statistics import mean, stdev

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB   = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

DECISION_GRADE_N = 200


def load_value_bets():
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT date, predicted_team_ml, correct, predicted_winner, home_win_prob "
        "FROM betting_log WHERE bet_rating = 'good' AND correct IS NOT NULL "
        "AND predicted_team_ml IS NOT NULL AND game_type != 'S' ORDER BY date")]
    conn.close()
    return rows


def kelly_stake(win_prob, ml, bankroll, fraction, max_stake_pct):
    # mirror of Main/app.py _kelly_stake — keep in sync
    if win_prob is None or ml is None:
        return None
    b = (ml / 100) if ml >= 0 else (100 / abs(ml))
    f = (win_prob * b - (1 - win_prob)) / b
    if f <= 0:
        return 0.0
    return round(bankroll * min(f * fraction, max_stake_pct), 2)


def simulate(bets, stake_fn):
    """Run one strategy; returns (net, stakes, max_drawdown, pl_list)."""
    running, peak, max_dd = 0.0, 0.0, 0.0
    stakes, pl_list = [], []
    for e in bets:
        stake = stake_fn(e)
        if not stake:            # None (missing prob) or 0.0 (Kelly passes)
            continue
        ml = e["predicted_team_ml"]
        pl = stake * (ml / 100 if ml >= 0 else 100 / abs(ml)) if e["correct"] else -stake
        running += pl
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
        stakes.append(stake)
        pl_list.append(pl)
    return running, stakes, max_dd, pl_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--cap-pct", type=float, default=0.05,
                    help="max stake as fraction of bankroll (default 0.05)")
    args = ap.parse_args()

    bets = load_value_bets()
    print(f"Resolved value bets: {len(bets)}")
    if not bets:
        print("\nNo usable data yet. Odds+result pairs only start accumulating once "
              "GITHUB_TOKEN-backed log persistence is live (see OPERATIONS.md).")
        return 1
    if len(bets) < DECISION_GRADE_N:
        print(f"[!] Below the ~{DECISION_GRADE_N}-bet bar for a decision-grade result — "
              "directional only.\n")

    def win_p(e):
        hwp = e.get("home_win_prob")
        if hwp is None:
            return None
        return hwp if e["predicted_winner"] == "Home" else 1 - hwp

    strategies = [
        ("Flat $10",       lambda e: 10.0),
        ("Quarter-Kelly",  lambda e: kelly_stake(win_p(e), e["predicted_team_ml"],
                                                 args.bankroll, 0.25, args.cap_pct)),
        ("Half-Kelly",     lambda e: kelly_stake(win_p(e), e["predicted_team_ml"],
                                                 args.bankroll, 0.50, args.cap_pct)),
    ]

    print(f"{'Strategy':<15} {'Bets':>5} {'Staked':>9} {'Net P/L':>9} {'ROI':>7} "
          f"{'Sharpe':>7} {'MaxDD':>8} {'AvgStake':>9}")
    curves = {}
    for name, fn in strategies:
        net, stakes, max_dd, pl_list = simulate(bets, fn)
        if not stakes:
            print(f"{name:<15} {'—':>5}   (no bets placed — Kelly passed on everything)")
            continue
        total = sum(stakes)
        sharpe = (mean(pl_list) / stdev(pl_list)) if len(pl_list) > 1 and stdev(pl_list) > 0 else 0.0
        print(f"{name:<15} {len(stakes):>5} ${total:>8.2f} ${net:>8.2f} {net/total:>7.1%} "
              f"{sharpe:>7.2f} ${max_dd:>7.2f} ${mean(stakes):>8.2f}")
        curves[name] = (net, stakes, pl_list)

    print("\nRead MaxDD (worst peak-to-trough) against Net P/L: Kelly earns its keep "
          "only if it improves risk-adjusted return (Sharpe), not just raw P/L.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, fn in strategies:
            if name not in curves:
                continue
            _, _, pl_list = curves[name]
            run, series = 0.0, []
            for pl in pl_list:
                run += pl
                series.append(run)
            axes[0].plot(series, label=name)
        axes[0].axhline(0, color="gray", linestyle="--")
        axes[0].set_title("Cumulative P/L by strategy (bet #)")
        axes[0].legend()
        if "Quarter-Kelly" in curves:
            axes[1].hist(curves["Quarter-Kelly"][1], bins=15, color="green", alpha=0.7)
            axes[1].axvline(10.0, color="blue", linestyle="--", label="Flat $10")
            axes[1].set_title("Quarter-Kelly stake distribution")
            axes[1].set_xlabel("Stake ($)")
            axes[1].legend()
        plt.tight_layout()
        out = os.path.join(_ROOT, "scripts", "kelly_comparison.png")
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
    except Exception as e:
        print(f"[plot skipped: {e}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
