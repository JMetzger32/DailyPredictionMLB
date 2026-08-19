"""
seed_odds_from_logs.py
----------------------
Seed odds_snapshots from odds ALREADY on disk. Costs zero API credits.

Two sources, both tagged so horizon-sensitive analysis can tell them apart:

  live_log         the away_ml/home_ml captured live on predictions_log entries.
                   These are MORNING prices (the 8 AM ET daily job / the 10:15 ET
                   refresh), which is the horizon the historical backfill targets.

  closing_archive  closing_odds_log.json, written by the 6:45 PM ET job. These are
                   CLOSING prices -- a different, later market than the morning
                   snapshot -- so they are usable as a fallback but must never be
                   pooled with morning prices in a horizon-sensitive comparison.

Usage:
    .venv/bin/python scripts/seed_odds_from_logs.py [--fill-log] [--dry-run]

--fill-log additionally writes recovered odds back onto predictions_log.json
entries that have closing_* fields but no away_ml (a small free win). Off by
default because it mutates a tracked data file.
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "updates"))
DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
CLOSING = os.path.join(_ROOT, "Databases_and_logs", "closing_odds_log.json")

from init_odds_tables import init_odds_tables  # noqa: E402


def _now():
    return datetime.now(timezone.utc).isoformat()


def _row(game_date_et, event_id, away_raw, home_raw, away, home, away_ml, home_ml,
         away_impl, home_impl, source, books=None, commence=None):
    return (game_date_et, event_id, game_date_et, "", None, commence, 0,
            away_raw, home_raw, away, home, away_ml, home_ml, away_impl, home_impl,
            None, len(books or []), json.dumps(books) if books else None, None,
            0, source, _now())


INSERT = """INSERT OR IGNORE INTO odds_snapshots
 (game_date_et, event_id, requested_date, requested_ts, snapshot_ts, commence_time,
  horizon_days, away_team_raw, home_team_raw, away_team, home_team, away_ml, home_ml,
  away_implied, home_implied, overround, n_books, books_json, arbitrage_pct,
  started_before_snapshot, source, fetched_at)
 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def implied(away_ml, home_ml):
    """De-vig, mirroring schedule_fetcher's math exactly."""
    if away_ml is None or home_ml is None:
        return None, None
    raw = lambda ml: abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)
    a, h = raw(away_ml), raw(home_ml)
    t = a + h
    return round(a / t, 4), round(h / t, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-log", action="store_true",
                    help="also write recovered odds back onto predictions_log entries")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    init_odds_tables(verbose=False)
    log = json.load(open(PRED_LOG))
    closing = json.load(open(CLOSING)) if os.path.exists(CLOSING) else {}

    rows, seen = [], set()

    # 1) live-captured odds already on predictions_log entries
    n_live = 0
    for date_str, entries in log.items():
        for e in entries:
            if e.get("away_ml") is None:
                continue
            pk = e.get("game_pk")
            eid = f"pk{pk}" if pk else f"live:{date_str}:{e.get('away_team')}:{e.get('home_team')}"
            key = (date_str, eid)
            if key in seen:
                continue
            seen.add(key)
            ai, hi = e.get("away_implied"), e.get("home_implied")
            if ai is None or hi is None:
                ai, hi = implied(e.get("away_ml"), e.get("home_ml"))
            rows.append(_row(date_str, eid, e.get("away_team_name") or "",
                             e.get("home_team_name") or "", e.get("away_team"),
                             e.get("home_team"), e.get("away_ml"), e.get("home_ml"),
                             ai, hi, "live_log", e.get("odds_books")))
            n_live += 1

    # 2) the closing-odds archive
    n_close = 0
    for date_str, games in closing.items():
        for key_str, od in games.items():
            try:
                away, home = key_str.split("|")
            except ValueError:
                continue
            eid = f"close:{away}:{home}"
            k = (date_str, eid)
            if k in seen:
                continue
            seen.add(k)
            ai, hi = od.get("away_implied"), od.get("home_implied")
            if ai is None or hi is None:
                ai, hi = implied(od.get("away_ml"), od.get("home_ml"))
            rows.append(_row(date_str, eid, away, home, away, home,
                             od.get("away_ml"), od.get("home_ml"), ai, hi,
                             "closing_archive"))
            n_close += 1

    print(f"live_log rows:        {n_live}")
    print(f"closing_archive rows: {n_close}")
    print(f"total to insert:      {len(rows)}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    conn = sqlite3.connect(DB)
    conn.executemany(INSERT, rows)
    conn.commit()
    got = conn.execute(
        "SELECT source, COUNT(*) FROM odds_snapshots GROUP BY source").fetchall()
    conn.close()
    print("odds_snapshots by source:", dict(got))

    # 3) optional: fill predictions_log entries that have closing_* but no away_ml
    if args.fill_log:
        filled = 0
        for date_str, entries in log.items():
            for e in entries:
                if e.get("away_ml") is not None:
                    continue
                if e.get("closing_away_ml") is None:
                    continue
                e["away_ml"] = e["closing_away_ml"]
                e["home_ml"] = e["closing_home_ml"]
                ai = e.get("closing_away_implied")
                hi = e.get("closing_home_implied")
                if ai is None or hi is None:
                    ai, hi = implied(e["away_ml"], e["home_ml"])
                e["away_implied"], e["home_implied"] = ai, hi
                # provenance: a 6:45 PM ET closing price, NOT a morning price
                e["odds_source"] = "closing_archive"
                filled += 1
        if filled:
            json.dump(log, open(PRED_LOG, "w"), indent=2)
        print(f"predictions_log entries filled from closing archive: {filled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
