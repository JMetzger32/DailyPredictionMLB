#!/usr/bin/env python3
"""
C1: quantify live-resolve contamination of the earlier live-2026 calibration diagnostic.

Before the immutability fix, a page view resolving a finished game scored the STORED entry
using the FRESH view-time prediction (correct/brier/log_loss computed from the recomputed
pick/prob, not the logged one). Detectable signatures on the 1,212 clean entries
(resolved, game_type R, not post_game_created):

  S1  pick/prob inconsistency: predicted_winner == "Home" but home_win_prob < 0.5
      (or "Away" but > 0.5) — the stored pick and stored probability disagree, meaning at
      least one of them was rewritten relative to the other.
  S2  stored brier_score inconsistent with (home_win_prob - actual)^2 recomputed from the
      SAME entry (tolerance 2e-3 to absorb the stored 4-dp rounding) — the scoring was
      computed from a probability that is not the stored one.

Reports counts, then recomputes the 10-bin winner-probability reliability table + ECE
excluding flagged entries, next to the full-set numbers (original diagnostic: ECE 0.0795).
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")


def reliability(entries):
    bins = [{"lo": i * 10, "hi": i * 10 + 10, "mid": (i * 10 + 5) / 100, "w": 0, "n": 0}
            for i in range(10)]
    for e in entries:
        p = e["home_win_prob"]
        if e.get("predicted_winner") == "Away":
            p = 1 - p
        idx = min(int(p * 10), 9)
        bins[idx]["n"] += 1
        if e["correct"]:
            bins[idx]["w"] += 1
    total = sum(b["n"] for b in bins)
    rows, ece = [], 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        act = b["w"] / b["n"]
        ece += abs(act - b["mid"]) * b["n"] / total
        rows.append((b["lo"], b["hi"], b["mid"], act, b["n"]))
    return rows, ece, total


def fmt(rows, ece, total, title):
    out = [title, f"{'bin':>9} | {'pred_mid':>8} | {'actual':>7} | {'count':>6}", "-" * 40]
    for lo, hi, mid, act, n in rows:
        out.append(f"{lo:>3}-{hi:<3}% | {mid:>8.2f} | {act:>7.3f} | {n:>6}")
    out.append(f"n={total}   ECE = {ece:.4f}")
    return "\n".join(out)


def main():
    log = json.load(open(LOG))
    clean = [e for day in log.values() for e in day
             if e.get("correct") is not None and e.get("game_type") == "R"
             and e.get("post_game_created") is not True
             and e.get("home_win_prob") is not None]

    s1 = []  # pick/prob inconsistent
    s2 = []  # brier inconsistent with stored prob
    for e in clean:
        p, pick = e["home_win_prob"], e.get("predicted_winner")
        if (pick == "Home" and p < 0.5) or (pick == "Away" and p > 0.5):
            s1.append(e)
        bs = e.get("brier_score")
        aw = e.get("actual_winner")
        if bs is not None and aw in ("Home", "Away"):
            actual = 1 if aw == "Home" else 0
            if abs(bs - (p - actual) ** 2) > 2e-3:
                s2.append(e)

    flagged_pks = {id(e) for e in s1} | {id(e) for e in s2}
    flagged = [e for e in clean if id(e) in flagged_pks]
    unflagged = [e for e in clean if id(e) not in flagged_pks]

    print(f"Clean sample: {len(clean)} entries")
    print(f"  S1 pick/prob inconsistent : {len(s1)} ({100*len(s1)/len(clean):.1f}%)")
    print(f"  S2 brier != f(stored prob): {len(s2)} ({100*len(s2)/len(clean):.1f}%)")
    print(f"  Flagged (union)           : {len(flagged)} ({100*len(flagged)/len(clean):.1f}%)")
    print()
    r_all = reliability(clean)
    r_cln = reliability(unflagged)
    print(fmt(*r_all, "A) FULL clean set (matches the original ECE 0.0795 diagnostic)"))
    print()
    print(fmt(*r_cln, "B) EXCLUDING contaminated entries"))
    print()
    d = r_cln[1] - r_all[1]
    print(f"VERDICT: ECE {r_all[1]:.4f} (all) -> {r_cln[1]:.4f} (decontaminated), delta {d:+.4f}")
    return r_all, r_cln, len(clean), len(flagged)


if __name__ == "__main__":
    main()
