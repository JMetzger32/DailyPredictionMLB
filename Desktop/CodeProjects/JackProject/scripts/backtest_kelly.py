"""
backtest_kelly.py
-----------------
Compares flat $10/game betting vs half-Kelly sizing (capped at $30)
against historical resolved value bets.

BEFORE RUNNING:
    1. Download latest logs: python3 scripts/download_logs.py
    2. Wait until ~July 10 — need ~200 resolved value bets for meaningful results
    3. python3 scripts/backtest_kelly.py

Kelly formula:  f* = (b*p - q) / b
  b = decimal odds - 1
  p = model win probability
  q = 1 - p
Half-Kelly: f*/2  (standard conservative practice — protects against noisy probs)
Cap: $30/bet maximum regardless of Kelly output
"""

# TODO (July 10): implement the following logic

# --- PSEUDOCODE ---
#
# STEP 1: Load resolved value bets from betting_log.json
#   data     = json.load(open("Databases_and_logs/betting_log.json"))
#   all_entries = [e for day in data.values() for e in day]
#   resolved = [e for e in all_entries
#               if e.get("correct") is not None
#               and e.get("predicted_team_ml") is not None
#               and e.get("away_ml") is not None
#               and e.get("bet_rating") == "good"]  # value bets only
#   print(f"Resolved value bets: {len(resolved)}")
#
# STEP 2: Simulate both strategies chronologically
#   BANKROLL   = 100.0
#   FLAT_STAKE = 10.0
#   MAX_KELLY  = 30.0
#
#   flat_cumulative   = []  # running P/L for flat betting
#   kelly_cumulative  = []  # running P/L for Kelly sizing
#   flat_running  = 0.0
#   kelly_running = 0.0
#   kelly_stakes  = []  # for distribution analysis
#
#   for entry in sorted(resolved, key=lambda e: e["date"]):
#       ml    = entry["predicted_team_ml"]
#       won   = entry["correct"]
#       p     = entry.get("home_win_prob") if entry["predicted_winner"] == "Home"
#               else entry.get("away_win_prob")
#
#       # Flat $10
#       flat_pl = FLAT_STAKE * (ml/100 if ml>=0 else 100/abs(ml)) if won else -FLAT_STAKE
#       flat_running += flat_pl
#       flat_cumulative.append((entry["date"], flat_running))
#
#       # Half-Kelly capped at $30
#       dec_odds = ml/100 + 1 if ml >= 0 else 100/abs(ml) + 1
#       b        = dec_odds - 1
#       q        = 1 - p
#       f_star   = (b * p - q) / b if b > 0 else 0
#       stake    = min(max(f_star / 2, 0) * BANKROLL, MAX_KELLY)
#       kelly_pl = stake * (ml/100 if ml>=0 else 100/abs(ml)) if won else -stake
#       kelly_running += kelly_pl
#       kelly_cumulative.append((entry["date"], kelly_running))
#       kelly_stakes.append(stake)
#
# STEP 3: Print summary
#   print(f"Flat $10:     net P/L = ${flat_running:.2f}  "
#         f"ROI = {flat_running/(len(resolved)*FLAT_STAKE):.1%}")
#   print(f"Half-Kelly:   net P/L = ${kelly_running:.2f}  "
#         f"avg stake = ${mean(kelly_stakes):.2f}  max stake used = ${max(kelly_stakes):.2f}")
#
# STEP 4: Plot side-by-side cumulative P/L
#   import matplotlib.pyplot as plt
#   fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#   axes[0].plot([x[0] for x in flat_cumulative],  [x[1] for x in flat_cumulative],
#                "b-", label="Flat $10")
#   axes[0].plot([x[0] for x in kelly_cumulative], [x[1] for x in kelly_cumulative],
#                "g-", label="Half-Kelly (max $30)")
#   axes[0].axhline(0, color="gray", linestyle="--")
#   axes[0].set_title("Cumulative P/L — Flat vs Kelly")
#   axes[0].legend()
#   axes[1].hist(kelly_stakes, bins=15, color="green", alpha=0.7)
#   axes[1].axvline(FLAT_STAKE, color="blue", linestyle="--", label="Flat $10")
#   axes[1].set_title("Kelly Stake Distribution")
#   axes[1].set_xlabel("Stake ($)")
#   axes[1].legend()
#   plt.tight_layout()
#   plt.savefig("scripts/kelly_comparison.png", dpi=150)
#   print("Saved: scripts/kelly_comparison.png")
#
# STEP 5: If Kelly shows better risk-adjusted return, implement in betting.html:
#   Add to openBetModal() JS function:
#     const kellyStake = Math.min(Math.max(f_star/2, 0) * 100, 30).toFixed(2);
#     document.getElementById("modal-kelly").textContent = `$${kellyStake} (on $100 bankroll)`;
# --- END PSEUDOCODE ---

print("This script is a pseudocode skeleton — implement fully in July after data accumulates.")
print("Run 'python3 scripts/download_logs.py' first to get the latest data.")
