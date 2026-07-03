#!/usr/bin/env python3
"""
Phase 2a: calibration table on CLEAN predictions.

Filters predictions_log.json to entries that are:
  - resolved            (correct is not None)
  - regular season      (game_type == "R")
  - genuinely pre-game  (post_game_created is not True)

Then applies the canonical /api/calibration binning (winner-probability into 10 even bins)
and prints: bin -> predicted mid -> actual win rate -> sample count.
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")


def main():
    log = json.load(open(LOG))
    entries = [
        e for day in log.values() for e in day
        if e.get("correct") is not None
        and e.get("game_type") == "R"
        and e.get("post_game_created") is not True
        and e.get("home_win_prob") is not None
    ]

    bins = [{"lo": i * 10, "hi": i * 10 + 10, "mid": (i * 10 + 5) / 100,
             "wins": 0, "total": 0} for i in range(10)]
    for e in entries:
        prob = e["home_win_prob"]
        if e.get("predicted_winner") == "Away":
            prob = 1 - prob                      # winner-probability, matches /api/calibration
        idx = min(int(prob * 10), 9)
        bins[idx]["total"] += 1
        if e["correct"]:
            bins[idx]["wins"] += 1

    print(f"Clean sample: {len(entries)} predictions "
          f"(resolved, R, not post_game_created)\n")
    print(f"{'bin':>10} | {'pred_mid':>8} | {'actual':>7} | {'count':>6}")
    print("-" * 42)
    total = correct = 0
    for b in bins:
        if b["total"] == 0:
            continue
        actual = b["wins"] / b["total"]
        total += b["total"]
        correct += b["wins"]
        print(f"{b['lo']:>3}-{b['hi']:<3}% | {b['mid']:>8.2f} | "
              f"{actual:>7.3f} | {b['total']:>6}")
    print("-" * 42)
    if total:
        print(f"{'overall':>10} | {'':>8} | {correct/total:>7.3f} | {total:>6}")


if __name__ == "__main__":
    main()
