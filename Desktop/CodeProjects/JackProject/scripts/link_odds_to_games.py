"""
link_odds_to_games.py
---------------------
Join odds_snapshots rows to the games they price. Costs zero credits and is fully
re-runnable -- deliberately separate from the fetcher so the join can be rebuilt as
the DB grows past its 2026-07-07 cutoff, or when the matching logic improves,
without ever touching prices that cost real money.

Two targets:
  games            2021-2025, keyed (date, visiting_team, home_team). The raw games
                   table stores OAK for 2021-2024 Athletics rows while odds always
                   resolve to ATH, so the games side is normalized on the join --
                   648 rows locally. MLBModel.load_data does this too, but the raw
                   table does not, and joining raw would lose every A's game.
  predictions_log  2026, keyed (date, game_pk). NEVER game_pk alone: 5 game_pks
                   recur across two dates (postponement + makeup), so a pk-only join
                   double-attaches odds to the postponed shell entry.

Doubleheaders are matched by commence_time order ONLY when the number of odds events
equals the number of games and the start times differ. Anything else is left
unmatched and counted. The common bad case is 1 event / 2 games -- doubleheaders
announced after the morning snapshot -- where attaching game 1's price to both games
would inject noise exactly where the starting-pitcher feature is also wrong.

Usage:
    .venv/bin/python scripts/link_odds_to_games.py [--report-only]
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
RESULTS = os.path.join(_ROOT, "scripts", "results")


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_odds(conn):
    """Primary (same-day) odds events only, grouped by (date, away, home)."""
    rows = conn.execute(
        "SELECT game_date_et, event_id, away_team, home_team, commence_time, source "
        "FROM odds_snapshots WHERE horizon_days = 0 "
        "AND away_team IS NOT NULL AND home_team IS NOT NULL").fetchall()
    g = defaultdict(list)
    for d, eid, a, h, ct, src in rows:
        g[(d, a, h)].append({"event_id": eid, "commence_time": ct or "", "source": src})
    for v in g.values():
        v.sort(key=lambda r: r["commence_time"])
    return g


def load_games(conn):
    """DB games 2021-2025, ISO date, OAK normalized to ATH to match the odds side."""
    rows = conn.execute(
        "SELECT game_id, date, visiting_team, home_team, doubleheader, season "
        "FROM games WHERE season BETWEEN 2021 AND 2025").fetchall()
    g = defaultdict(list)
    for gid, d, away, home, dh, season in rows:
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        away = "ATH" if away == "OAK" else away
        home = "ATH" if home == "OAK" else home
        g[(iso, away, home)].append({"game_id": gid, "doubleheader": dh or 0,
                                     "season": season})
    for v in g.values():
        v.sort(key=lambda r: (r["doubleheader"], r["game_id"]))
    return g


def load_pred_entries():
    """2026 predictions_log entries grouped the same way."""
    log = json.load(open(PRED_LOG))
    g = defaultdict(list)
    for date_str, entries in log.items():
        for e in entries:
            if e.get("game_type") == "S":
                continue
            g[(date_str, e.get("away_team"), e.get("home_team"))].append(
                {"game_pk": e.get("game_pk"), "date": date_str})
    for v in g.values():
        v.sort(key=lambda r: (r["game_pk"] or 0))
    return g


def match(odds_groups, game_groups, target, id_field):
    """Pair odds events to games. Returns (links, stats)."""
    links, stats = [], defaultdict(int)
    for key, games in game_groups.items():
        events = odds_groups.get(key)
        if not events:
            stats["no_odds"] += len(games)
            continue
        if len(events) == 1 and len(games) == 1:
            links.append((key[0], events[0]["event_id"], target,
                          games[0].get(id_field), "unique_date_teams", "exact"))
            stats["exact"] += 1
        elif len(events) == len(games) and \
                len({e["commence_time"] for e in events}) == len(events):
            for ev, gm in zip(events, games):
                links.append((key[0], ev["event_id"], target, gm.get(id_field),
                              "dh_by_commence_order", "exact"))
            stats["dh_exact"] += len(games)
        elif len(events) == len(games):
            stats["dh_ambiguous"] += len(games)      # identical start times
        else:
            stats["dh_count_mismatch"] += len(games)  # usually 1 event / 2 games
    return links, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    odds = load_odds(conn)
    print(f"odds event groups (horizon_days=0): {len(odds)}")

    all_links, report = [], []
    for target, loader, id_field in (("games", load_games, "game_id"),
                                     ("predictions_log", None, "game_pk")):
        groups = loader(conn) if loader else load_pred_entries()
        links, stats = match(odds, groups, target, id_field)
        all_links += links
        n_games = sum(len(v) for v in groups.values())
        linked = len(links)
        print(f"\n[{target}] {n_games} games in {len(groups)} groups -> {linked} linked "
              f"({linked/n_games*100:.1f}%)" if n_games else f"\n[{target}] no games")
        for k in sorted(stats):
            print(f"    {k:<20} {stats[k]}")
        report.append((target, n_games, linked, dict(stats)))

    if not args.report_only and all_links:
        conn.executemany(
            "INSERT OR REPLACE INTO odds_game_link (game_date_et, event_id, target, "
            f"game_id, game_pk, match_method, confidence, linked_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(d, e, t, (g if t == "games" else None),
              (g if t == "predictions_log" else None), m, c, _now())
             for d, e, t, g, m, c in all_links])
        conn.commit()
        print(f"\nwrote {len(all_links)} rows to odds_game_link")

    # per-team coverage -- the check that caught the Athletics bug; keep it permanent
    print("\nper-team odds coverage (primary events):")
    tc = conn.execute(
        "SELECT team, COUNT(*) FROM ("
        "  SELECT away_team AS team FROM odds_snapshots WHERE horizon_days=0"
        "  UNION ALL SELECT home_team FROM odds_snapshots WHERE horizon_days=0"
        ") WHERE team IS NOT NULL GROUP BY team ORDER BY 2").fetchall()
    if tc:
        counts = [c for _, c in tc]
        median = sorted(counts)[len(counts) // 2]
        low = [(t, c) for t, c in tc if c < 0.8 * median]
        print(f"    teams={len(tc)} median={median} "
              f"min={tc[0][0]}:{tc[0][1]} max={tc[-1][0]}:{tc[-1][1]}")
        print(f"    BELOW 80% OF MEDIAN: {low if low else 'none'}")
        if len(tc) < 30:
            missing = 30 - len(tc)
            print(f"    !! {missing} team(s) have NO odds rows at all")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
