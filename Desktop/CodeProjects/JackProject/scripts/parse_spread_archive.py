"""
parse_spread_archive.py
-----------------------
Parse the gzipped run-line (spreads) archive into odds_snapshots_spreads.

Costs ZERO credits and is fully re-runnable -- deliberately separate from the
fetcher (updates/backfill_historical_spreads.py) so the parse can be rebuilt as the
parsing logic improves, without ever touching prices that cost real money and, after
the paid plan lapses, cannot be re-bought at any price.

Mirrors the h2h side's division of labour: backfill_historical_odds.py fetches and
archives, this parses. Only events whose ET game date equals the requested date are
'primary' (horizon_days = 0); the endpoint returns a rolling window of upcoming
events, so out-of-window rows are stored tagged but must never be used as a primary
price -- a line 24-48h out is a different market.

Usage:
    .venv/bin/python scripts/parse_spread_archive.py [--report-only] [--limit N]
"""
import argparse
import glob
import gzip
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "updates"))

DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
ARCHIVE = os.path.join(_ROOT, "Databases_and_logs", "odds_archive_spreads")

import schedule_fetcher as sf                                   # noqa: E402
from init_odds_spreads_table import init_odds_spreads_table     # noqa: E402


def _now():
    return datetime.now(timezone.utc).isoformat()


def store_events(conn, date_str, requested_ts, snap_ts, rows, source="historical_api"):
    """Persist parsed spread events. Returns (n_stored, n_primary)."""
    out, n_primary = [], 0
    for r in rows:
        gd = r["game_date_et"]
        if not gd:
            continue
        horizon = (datetime.fromisoformat(gd) - datetime.fromisoformat(date_str)).days
        if horizon == 0:
            n_primary += 1
        started = 0
        if snap_ts and r["commence_time"]:
            try:
                started = int(
                    datetime.fromisoformat(r["commence_time"].replace("Z", "+00:00"))
                    <= datetime.fromisoformat(snap_ts.replace("Z", "+00:00")))
            except Exception:
                started = 0
        out.append((gd, r["event_id"], date_str, requested_ts, snap_ts,
                    r["commence_time"], horizon, r["away_team_raw"], r["home_team_raw"],
                    r["away_team"], r["home_team"], r["away_point"], r["home_point"],
                    r["away_spread_ml"], r["home_spread_ml"], r["away_implied"],
                    r["home_implied"], r["overround"], r["n_books"],
                    json.dumps(r["books"]), started, source, _now()))
    if out:
        conn.executemany(
            "INSERT OR REPLACE INTO odds_snapshots_spreads (game_date_et, event_id, "
            "requested_date, requested_ts, snapshot_ts, commence_time, horizon_days, "
            "away_team_raw, home_team_raw, away_team, home_team, away_point, home_point, "
            "away_spread_ml, home_spread_ml, away_implied, home_implied, overround, "
            "n_books, books_json, started_before_snapshot, source, fetched_at) "
            "VALUES (" + ",".join("?" * 23) + ")", out)
        conn.commit()
    return len(out), n_primary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="parse and report, write nothing")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.report_only:
        init_odds_spreads_table(verbose=False)
    conn = sqlite3.connect(DB)

    files = sorted(glob.glob(os.path.join(ARCHIVE, "*", "*.json.gz")))
    if args.limit:
        files = files[:args.limit]
    print(f"archived dates: {len(files)}")

    sf.UNMAPPED_ODDS_TEAMS.clear()
    tot_events = tot_rows = tot_primary = 0
    unreadable, no_rows = [], []
    overrounds, points = [], Counter()

    for i, path in enumerate(files, 1):
        date_str = os.path.basename(path).replace(".json.gz", "")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as e:
            unreadable.append((date_str, str(e)))
            continue
        data = payload.get("data") or []
        snap_ts = payload.get("timestamp")
        rows = sf.parse_spread_events(data)
        tot_events += len(data)
        tot_rows += len(rows)
        if data and not rows:
            no_rows.append(date_str)
        for r in rows:
            overrounds.append(r["overround"])
            points[(r["away_point"], r["home_point"])] += 1
        if not args.report_only:
            _, n_prim = store_events(conn, date_str,
                                     f"{date_str}T14:00:00Z", snap_ts, rows)
            tot_primary += n_prim
        if i % 200 == 0:
            print(f"  [{i}/{len(files)}] ...", flush=True)

    print(f"\nevents in archive : {tot_events}")
    print(f"rows parsed       : {tot_rows} "
          f"({tot_rows/tot_events*100:.1f}%)" if tot_events else "")
    if not args.report_only:
        print(f"primary (horizon0): {tot_primary}")
    if overrounds:
        print(f"overround         : mean={sum(overrounds)/len(overrounds):.4f} "
              f"min={min(overrounds):.4f} max={max(overrounds):.4f}")
    print(f"point pairs seen  : {dict(points)}")
    print(f"unmapped teams    : {dict(sf.UNMAPPED_ODDS_TEAMS) or 'none'}")
    if unreadable:
        print(f"UNREADABLE archives: {unreadable[:10]}")
    if no_rows:
        print(f"dates with events but no parsable rows: {len(no_rows)} {no_rows[:10]}")

    if not args.report_only:
        n = conn.execute("SELECT COUNT(*) FROM odds_snapshots_spreads").fetchone()[0]
        prim = conn.execute("SELECT COUNT(*) FROM odds_snapshots_spreads "
                            "WHERE horizon_days=0").fetchone()[0]
        print(f"\nodds_snapshots_spreads rows: {n} ({prim} primary)")
        # per-team coverage -- the same check that caught the Athletics bug on h2h
        tc = conn.execute(
            "SELECT team, COUNT(*) FROM ("
            "  SELECT away_team AS team FROM odds_snapshots_spreads WHERE horizon_days=0"
            "  UNION ALL SELECT home_team FROM odds_snapshots_spreads WHERE horizon_days=0"
            ") WHERE team IS NOT NULL GROUP BY team ORDER BY 2").fetchall()
        if tc:
            counts = [c for _, c in tc]
            median = sorted(counts)[len(counts) // 2]
            low = [(t, c) for t, c in tc if c < 0.8 * median]
            print(f"per-team coverage: teams={len(tc)} median={median} "
                  f"min={tc[0][0]}:{tc[0][1]} max={tc[-1][0]}:{tc[-1][1]}")
            print(f"  BELOW 80% OF MEDIAN: {low if low else 'none'}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
