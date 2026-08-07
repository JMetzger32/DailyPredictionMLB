"""
segment_value_bets.py
---------------------
Phase 2 of the edge-calibration investigation: does a BIGGER claimed edge actually
mean a BETTER bet? Segments the resolved value-bet history three ways:

  2a  by |model_edge| bucket (5-8%, 8-12%, 12%+)  -> win%, avg CLV, net P/L
  2b  directional skew (home/away, favorite/underdog) among flagged value bets
  2c  game-level detail for the most recent N resolved value bets

Usage:
    python3 scripts/segment_value_bets.py [--stake 10] [--recent 15] [--no-pitchers]

Data source: Databases_and_logs/betting_log.json (the GitHub-backed log, which
survives Render restarts — the local SQLite betting_log is typically stale; see
CLAUDE.md). Value bet = bet_rating in {good, extreme}, matching what the betting
page counts, plus the resolved/regular-season filters used by
scripts/calibration_live_check.py.

Report-only: this script changes nothing.
"""
import argparse
import json
import os
import sys
from collections import Counter
from statistics import mean

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG  = os.path.join(_ROOT, "Databases_and_logs", "betting_log.json")

VALUE_RATINGS = ("good", "extreme")
BUCKETS = [("5-8%", 0.05, 0.08), ("8-12%", 0.08, 0.12), ("12%+", 0.12, 1.01)]


def american_to_raw(ml):
    """American moneyline -> raw (pre-vig) implied probability."""
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def devig(away_ml, home_ml):
    """Both implied probabilities, normalized to sum to 1 — mirrors
    updates/schedule_fetcher.py:598-618. Verified in Phase 1 to reproduce the
    stored model_edge exactly on all 222 rows."""
    a, h = american_to_raw(away_ml), american_to_raw(home_ml)
    t = a + h
    return round(a / t, 4), round(h / t, 4)


def pl_for(ml, won, stake):
    """Flat-stake profit/loss on an American-odds moneyline."""
    if not won:
        return -stake
    return stake * (ml / 100 if ml >= 0 else 100 / abs(ml))


def load_resolved():
    with open(_LOG) as f:
        log = json.load(f)
    rows = []
    for day in sorted(log):
        for e in log[day]:
            if e.get("correct") is None or e.get("bet_rating") is None:
                continue
            if e.get("game_type") != "R" or e.get("post_game_created"):
                continue
            rows.append(e)
    return rows


def summarize(rows, stake):
    """(n, wins, losses, win_pct, avg_clv, net_pl, roi) for a set of bets."""
    if not rows:
        return None
    wins = sum(1 for e in rows if e["correct"])
    clvs = [e["clv"] for e in rows if e.get("clv") is not None]
    net = sum(pl_for(e["predicted_team_ml"], e["correct"], stake) for e in rows)
    return {
        "n": len(rows),
        "w": wins,
        "l": len(rows) - wins,
        "win_pct": 100 * wins / len(rows),
        "avg_clv": mean(clvs) if clvs else None,
        "clv_n": len(clvs),
        "net": net,
        "roi": 100 * net / (len(rows) * stake),
    }


def row(label, s):
    if s is None:
        return f"| {label} | 0 | — | — | — | — | — |"
    clv = f"{100*s['avg_clv']:+.2f}%" if s["avg_clv"] is not None else "—"
    return (f"| {label} | {s['n']} | {s['w']}-{s['l']} | {s['win_pct']:.1f}% | "
            f"{clv} | ${s['net']:+.2f} | {s['roi']:+.1f}% |")


HEADER = "| segment | n | W-L | win% | avg CLV | net P/L | ROI |\n|---|---|---|---|---|---|---|"


def fetch_pitchers(game_pks):
    """Probable pitchers from the MLB StatsAPI. betting_log/predictions_log stop
    persisting pitcher names after 2026-04-04, so 2c has to re-fetch them."""
    import urllib.request
    out = {}
    for pk in game_pks:
        try:
            url = (f"https://statsapi.mlb.com/api/v1/schedule?gamePk={pk}"
                   f"&hydrate=probablePitcher")
            with urllib.request.urlopen(url, timeout=15) as r:
                g = json.load(r)["dates"][0]["games"][0]
            out[pk] = (
                g["teams"]["away"].get("probablePitcher", {}).get("fullName", "?"),
                g["teams"]["home"].get("probablePitcher", {}).get("fullName", "?"),
            )
        except Exception:
            out[pk] = ("?", "?")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--recent", type=int, default=15)
    ap.add_argument("--no-pitchers", action="store_true",
                    help="skip the MLB StatsAPI lookup in 2c")
    args = ap.parse_args()

    resolved = load_resolved()
    value = [e for e in resolved if e["bet_rating"] in VALUE_RATINGS]
    days = sorted({e["date"] for e in resolved})
    print(f"Resolved rated bets: {len(resolved)}  ({days[0]} -> {days[-1]})")
    print(f"Value bets (good+extreme): {len(value)}   flat stake ${args.stake:.2f}\n")

    # ---- 2a: by edge bucket -------------------------------------------------
    print("## 2a. By |model_edge| bucket")
    print(HEADER)
    for label, lo, hi in BUCKETS:
        sub = [e for e in value if lo <= abs(e["model_edge"]) < hi]
        print(row(label, summarize(sub, args.stake)))
    print(row("ALL value bets", summarize(value, args.stake)))
    # the comparison that motivated the investigation
    tossup = [e for e in resolved if e["bet_rating"] == "unsure"]
    print(row("(Toss-Ups, for contrast)", summarize(tossup, args.stake)))

    # ---- 2a-bis: persisted label vs today's rules ---------------------------
    # bet_rating is computed once when odds attach and never revisited, and the
    # "extreme" category was only introduced 2026-07-23. Rows tagged before that
    # keep bet_rating='good' even when edge > 0.12, so the page's "Value Bets"
    # bucket still contains high-edge bets that today's rules would exclude.
    print("\n## 2a-bis. Persisted label vs today's thresholds applied consistently")
    print(HEADER)
    print(row("as-labeled 'good' (page's Value Bets)",
              summarize([e for e in resolved if e["bet_rating"] == "good"], args.stake)))
    print(row("as-labeled 'extreme'",
              summarize([e for e in resolved if e["bet_rating"] == "extreme"], args.stake)))
    print(row("by edge 0.05 < e <= 0.12 (today's 'good')",
              summarize([e for e in resolved if 0.05 < e["model_edge"] <= 0.12], args.stake)))
    print(row("by edge e > 0.12 (today's 'extreme')",
              summarize([e for e in resolved if e["model_edge"] > 0.12], args.stake)))
    stale = [e for e in resolved if e["model_edge"] >= 0.12 and e["bet_rating"] == "good"]
    if stale:
        print(f"\n{len(stale)} rows carry bet_rating='good' despite edge >= 0.12 "
              f"(all dated {min(e['date'] for e in stale)} to {max(e['date'] for e in stale)}, "
              f"before 'extreme' existed).")

    # ---- 2b: directional skew ----------------------------------------------
    print("\n## 2b. Directional skew among value bets")
    print(HEADER)
    for label, pred in (("pick = Home", "Home"), ("pick = Away", "Away")):
        print(row(label, summarize([e for e in value if e["predicted_winner"] == pred],
                                   args.stake)))
    fav = [e for e in value if e["predicted_team_ml"] < 0]
    dog = [e for e in value if e["predicted_team_ml"] >= 0]
    print(row("pick = favorite (ML<0)", summarize(fav, args.stake)))
    print(row("pick = underdog (ML>0)", summarize(dog, args.stake)))

    # base rate for comparison: how often is a value bet even available on each side
    all_home = sum(1 for e in resolved if e["predicted_winner"] == "Home")
    print(f"\nFlag rate: {sum(1 for e in value if e['predicted_winner']=='Home')}/{all_home} "
          f"of Home picks flagged value, "
          f"{sum(1 for e in value if e['predicted_winner']=='Away')}/{len(resolved)-all_home} "
          f"of Away picks.")
    print(f"Favorite/underdog split of ALL resolved rated bets: "
          f"{sum(1 for e in resolved if e['predicted_team_ml']<0)} fav / "
          f"{sum(1 for e in resolved if e['predicted_team_ml']>=0)} dog.")

    # ---- 2c: recent game log ------------------------------------------------
    recent = value[-args.recent:]
    print(f"\n## 2c. Most recent {len(recent)} resolved value bets")
    pitchers = {} if args.no_pitchers else fetch_pitchers([e["game_pk"] for e in recent])
    print("| date | matchup | pick | ML | edge | SP (pick side) | result | CLV |")
    print("|---|---|---|---|---|---|---|---|")
    for e in recent:
        pick_team = e["home_team"] if e["predicted_winner"] == "Home" else e["away_team"]
        ap_, hp_ = pitchers.get(e["game_pk"], ("?", "?"))
        sp = hp_ if e["predicted_winner"] == "Home" else ap_
        clv = f"{100*e['clv']:+.1f}%" if e.get("clv") is not None else "—"
        print(f"| {e['date']} | {e['away_team']}@{e['home_team']} | {pick_team} "
              f"| {e['predicted_team_ml']:+d} | {100*e['model_edge']:+.1f}pp | {sp} "
              f"| {'W' if e['correct'] else 'L'} | {clv} |")

    losses = [e for e in recent if not e["correct"]]
    print(f"\nRecent {len(recent)}: {sum(1 for e in recent if e['correct'])}W-{len(losses)}L")
    if losses:
        print("Loss clustering — team (as picked): "
              f"{Counter((e['home_team'] if e['predicted_winner']=='Home' else e['away_team']) for e in losses).most_common(5)}")
        print(f"Loss clustering — edge bucket: "
              f"{Counter(next(lb for lb, lo, hi in BUCKETS if lo <= abs(e['model_edge']) < hi) for e in losses).most_common()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
