"""
backtest_threshold.py
---------------------
Backtests different edge thresholds against historical betting_log data
to find the ROI-maximizing threshold for "value bet" classification.

BEFORE RUNNING:
    1. Download latest logs: python3 scripts/download_logs.py
    2. Wait until ~July 10 — need ~200 resolved value bets for meaningful results
    3. python3 scripts/backtest_threshold.py

Current threshold (hard-coded in Main/app.py line ~1123): 0.05 (5%)
This script will tell you if a different threshold would have performed better.
"""

# TODO (July 10): implement the following logic

# --- PSEUDOCODE ---
#
# STEP 1: Load betting_log.json
#   data = json.load(open("Databases_and_logs/betting_log.json"))
#   all_entries = [entry for day in data.values() for entry in day]
#   resolved = [e for e in all_entries
#               if e.get("correct") is not None
#               and e.get("model_edge") is not None
#               and e.get("predicted_team_ml") is not None]
#   print(f"Resolved value bets available: {len(resolved)}")
#
# STEP 2: Grid search over thresholds
#   STAKE = 10.0
#   MIN_SAMPLE = 30  # ignore thresholds with fewer bets
#   results = []
#
#   for T in [round(t * 0.01, 2) for t in range(1, 16)]:  # 0.01 to 0.15
#       bets = [e for e in resolved if (e.get("model_edge") or 0) >= T]
#       if len(bets) < MIN_SAMPLE:
#           continue
#
#       pl_list = []
#       for b in bets:
#           ml = b["predicted_team_ml"]
#           if b["correct"]:
#               pl = STAKE * (ml / 100 if ml >= 0 else 100 / abs(ml))
#           else:
#               pl = -STAKE
#           pl_list.append(pl)
#
#       wins    = sum(1 for b in bets if b["correct"])
#       net_pl  = sum(pl_list)
#       roi     = net_pl / (len(bets) * STAKE)
#       sharpe  = mean(pl_list) / stdev(pl_list) if len(pl_list) > 1 else None
#
#       results.append({
#           "threshold": T,
#           "bets":      len(bets),
#           "win_pct":   wins / len(bets),
#           "roi":       roi,
#           "sharpe":    sharpe,
#           "net_pl":    net_pl,
#       })
#
# STEP 3: Print results table
#   print(f"{'Threshold':>10} {'Bets':>6} {'Win%':>7} {'ROI':>8} {'Sharpe':>8} {'Net P/L':>10}")
#   for r in results:
#       print(f"{r['threshold']:>10.2f} {r['bets']:>6} {r['win_pct']:>7.1%} "
#             f"{r['roi']:>8.1%} {r['sharpe'] or 0:>8.2f} ${r['net_pl']:>9.2f}")
#
# STEP 4: Find recommended threshold
#   # Highest Sharpe ratio (risk-adjusted) where bets >= MIN_SAMPLE
#   best = max(results, key=lambda r: r["sharpe"] or 0)
#   print(f"\nRecommended threshold: {best['threshold']} "
#         f"(Sharpe={best['sharpe']:.2f}, ROI={best['roi']:.1%}, N={best['bets']})")
#
# STEP 5: Plot ROI curve + sample size
#   import matplotlib.pyplot as plt
#   fig, ax1 = plt.subplots(figsize=(10, 5))
#   ax1.plot([r["threshold"] for r in results], [r["roi"] for r in results],
#            "b-o", label="ROI")
#   ax1.axvline(0.05, color="gray", linestyle="--", label="Current threshold (5%)")
#   ax1.axvline(best["threshold"], color="green", linestyle="--",
#               label=f"Recommended ({best['threshold']:.0%})")
#   ax1.set_xlabel("Edge Threshold")
#   ax1.set_ylabel("ROI", color="b")
#   ax2 = ax1.twinx()
#   ax2.bar([r["threshold"] for r in results], [r["bets"] for r in results],
#           alpha=0.3, color="orange", label="Sample size")
#   ax2.set_ylabel("Number of Bets", color="orange")
#   plt.title("Edge Threshold Calibration — ROI vs Sample Size")
#   fig.tight_layout()
#   plt.savefig("scripts/threshold_calibration.png", dpi=150)
#   print("Saved: scripts/threshold_calibration.png")
#
# STEP 6: If new threshold looks better, update Main/app.py line ~1123:
#   if edge > NEW_THRESHOLD:
#       bet_rating = "good"
#   elif edge < -NEW_THRESHOLD:
#       bet_rating = "bad"
# --- END PSEUDOCODE ---

print("This script is a pseudocode skeleton — implement fully in July after data accumulates.")
print("Run 'python3 scripts/download_logs.py' first to get the latest data.")
