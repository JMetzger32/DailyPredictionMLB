"""
backfill_historical_odds.py
---------------------------
Fetch historical MLB odds snapshots from The Odds API.

THIS IS THE ONLY SCRIPT IN THE PROJECT THAT SPENDS A PERISHABLE RESOURCE.
Historical calls cost 10 credits each (10 x markets x regions) and are available
only on paid plans. Once the account downgrades, these prices cannot be re-bought
at any price, so every guard here exists to make the one shot count.

Design:
  * One snapshot per date, at a fixed UTC time, so every price is measured at the
    same horizon. The endpoint returns a rolling window of UPCOMING events, so the
    response spans several game dates; only events whose ET game date equals the
    requested date are 'primary' (horizon_days=0). Out-of-window events are stored
    tagged but must never be used as a primary price -- a line 24-48h before first
    pitch is a different market (pre-lineup, higher vig).
  * The raw envelope is archived gzipped to disk BEFORE parsing, so re-parsing after
    any bug fix costs zero credits forever (--archive-only).
  * Resumability is archive-driven: a date is done iff its .json.gz exists and
    parses. No checkpoint file to desync.
  * Three independent budget layers, of which the per-call one is authoritative
    because it reads the server's own remaining count rather than a local estimate.

Usage:
    --probe DATE [--snapshot-time HH:MM,HH:MM]   validate cost/parse, no DB write
    --canary                                     one date per season, GO/NO-GO gate
    --dry-run                                    resolve dates + cost, no HTTP
    --archive-only                               re-parse from disk, ZERO credits
    --status                                     progress + credits
"""
import argparse
import gzip
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone

_UPDATES = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UPDATES)
sys.path.insert(0, _UPDATES)

DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
ARCHIVE = os.path.join(_ROOT, "Databases_and_logs", "odds_archive")

from odds_key import load_odds_api_key, key_fingerprint          # noqa: E402
from init_odds_tables import init_odds_tables                    # noqa: E402
import schedule_fetcher as sf                                    # noqa: E402

HISTORICAL_COST = 10          # credits per successful call; asserted at runtime
DEFAULT_SNAPSHOT = "14:00"    # UTC = 10 AM ET, before nearly every first pitch
THROTTLE = 0.25
CANARY_DATES = ["2021-06-15",   # "Cleveland Indians" era, 10-min snapshots
                "2022-08-15",   # pre 5-min-interval boundary
                "2023-06-15",
                "2024-06-15",   # "Oakland Athletics"
                "2025-06-15",   # "Athletics" / "Sacramento Athletics"
                "2026-05-15"]   # real backfill target shape


def _now():
    return datetime.now(timezone.utc).isoformat()


def would_breach_floor(remaining, floor, cost=HISTORICAL_COST):
    """True if spending one more call would drop below the reserve.

    The reserve exists so live/future games always have credits. Deliberately
    evaluated against the SERVER's reported `remaining` before every call, never a
    locally-tracked estimate -- a local counter drifts the moment any other client
    (the Render app shares this key) spends anything. Unknown remaining is treated
    as unsafe.
    """
    if remaining is None:
        return True
    return remaining - cost < floor


def archive_path(date_str):
    return os.path.join(ARCHIVE, date_str[:4], f"{date_str}.json.gz")


def archive_write(date_str, payload):
    p = archive_path(date_str)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return p


def archive_read(date_str):
    p = archive_path(date_str)
    if not os.path.exists(p):
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def target_dates():
    """Every date needing odds: DB games 2021-2025 + predictions_log 2026 gaps."""
    conn = sqlite3.connect(DB)
    db_dates = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for (d,) in conn.execute(
        "SELECT DISTINCT date FROM games WHERE season BETWEEN 2021 AND 2025 "
        "ORDER BY date")]
    conn.close()
    log = json.load(open(PRED_LOG))
    log_dates = sorted(d for d, es in log.items()
                       if any(e.get("away_ml") is None and e.get("game_type") != "S"
                              for e in es))
    return sorted(set(db_dates) | set(log_dates))


def store_events(conn, date_str, requested_ts, meta, rows, source="historical_api"):
    """Persist parsed events. horizon_days marks how far the game is from the
    requested date; only 0 is a primary same-horizon price."""
    snap_ts = meta.get("timestamp")
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
                started = int(datetime.fromisoformat(r["commence_time"].replace("Z", "+00:00"))
                              <= datetime.fromisoformat(snap_ts.replace("Z", "+00:00")))
            except Exception:
                started = 0
        out.append((gd, r["event_id"], date_str, requested_ts, snap_ts,
                    r["commence_time"], horizon, r["away_team_raw"], r["home_team_raw"],
                    r["away_team"], r["home_team"], r["away_ml"], r["home_ml"],
                    r["away_implied"], r["home_implied"], r["overround"], r["n_books"],
                    json.dumps(r["books"]),
                    (r["arbitrage"] or {}).get("profit_pct") if r["arbitrage"] else None,
                    started, source, _now()))
    if out:
        conn.executemany(
            "INSERT OR REPLACE INTO odds_snapshots (game_date_et, event_id, "
            "requested_date, requested_ts, snapshot_ts, commence_time, horizon_days, "
            "away_team_raw, home_team_raw, away_team, home_team, away_ml, home_ml, "
            "away_implied, home_implied, overround, n_books, books_json, "
            "arbitrage_pct, started_before_snapshot, source, fetched_at) "
            "VALUES (" + ",".join("?" * 22) + ")", out)
        conn.commit()
    return len(out), n_primary


def fetch_one(key, date_str, snapshot_time, conn, attempt=1):
    """One paid call. Returns (ok, meta, rows, credits_before, credits_after)."""
    ts = f"{date_str}T{snapshot_time}:00Z"
    before = sf.get_last_odds_quota().get("remaining")
    meta, payload, rows = sf.get_historical_mlb_odds(key, ts)
    after = sf.get_last_odds_quota().get("remaining")
    charged = (before - after) if (before is not None and after is not None) else None
    if payload is not None:
        archive_write(date_str, payload)
    conn.execute(
        "INSERT OR REPLACE INTO odds_fetch_log (requested_date, requested_ts, attempt,"
        " http_status, snapshot_ts, n_events, n_primary, n_unmapped,"
        " credits_remaining_before, credits_remaining_after, credits_charged, error,"
        " fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (date_str, ts, attempt, meta.get("status"), meta.get("timestamp"), len(rows),
         None, len(sf.UNMAPPED_ODDS_TEAMS), before, after, charged,
         meta.get("error"), _now()))
    conn.commit()
    return meta.get("http_ok"), meta, rows, before, after


def cmd_probe(key, args, conn):
    """Validate cost model, envelope, parse and snapshot time before the big spend."""
    date_str = args.probe
    times = [t.strip() for t in (args.snapshot_time or DEFAULT_SNAPSHOT).split(",")]
    log = json.load(open(PRED_LOG))
    stored = {(e["away_team"], e["home_team"]): e for e in log.get(date_str, [])
              if e.get("away_ml") is not None}
    print(f"PROBE {date_str}  snapshot times: {times}")
    print(f"  stored live-odds games on that date: {len(stored)}")

    results = {}
    for t in times:
        sf.UNMAPPED_ODDS_TEAMS.clear()
        ok, meta, rows, before, after = fetch_one(key, date_str, t, conn)
        charged = (before - after) if (before is not None and after is not None) else None
        print(f"\n  [{t}Z] http={meta.get('status')} events={len(rows)} "
              f"credits {before} -> {after} (charged {charged})")
        if not ok:
            print(f"    ERROR {meta.get('error')}")
            return 1
        # ASSERTION 1: cost model. A 1,051-call loop at an unknown cost is exactly
        # how the reserve gets blown, so this is fatal rather than a warning.
        if charged is not None and charged != HISTORICAL_COST:
            print(f"    FATAL: expected {HISTORICAL_COST} credits/call, got {charged}. "
                  f"ABORTING — re-plan the budget before any bulk run.")
            return 2
        print(f"    snapshot_ts={meta.get('timestamp')} prev={meta.get('previous_timestamp')}")
        primary = [r for r in rows if r["game_date_et"] == date_str]
        print(f"    primary (ET date == {date_str}): {len(primary)} / {len(rows)}")
        print(f"    unmapped team names: {dict(sf.UNMAPPED_ODDS_TEAMS) or 'none'}")
        started = [r for r in primary if r["commence_time"] and meta.get("timestamp")
                   and r["commence_time"] <= meta["timestamp"]]
        print(f"    already started at snapshot: {len(started)}")
        # ASSERTION 2: prices should be CLOSE to, but not identical to, stored live
        # ones. Identical would mean a caching artifact; far off means wrong parse.
        diffs = []
        for r in primary:
            s = stored.get((r["away_team"], r["home_team"]))
            if not s:
                continue
            s_ai = s.get("away_implied")
            if s_ai is None and s.get("away_ml") is not None:
                # most stored rows keep only the moneylines; de-vig them the same way
                raw = lambda ml: abs(ml)/(abs(ml)+100) if ml < 0 else 100/(ml+100)
                a, h = raw(s["away_ml"]), raw(s["home_ml"])
                s_ai = a / (a + h)
            if s_ai is not None:
                diffs.append(abs(r["away_implied"] - s_ai))
        if diffs:
            mean_d = sum(diffs) / len(diffs)
            print(f"    vs stored live odds: n={len(diffs)} mean|d implied|={mean_d:.4f}")
            results[t] = mean_d
        else:
            print("    (no overlapping stored implied probs to compare)")
        p = archive_path(date_str)
        print(f"    archived {os.path.getsize(p)} bytes -> {os.path.relpath(p, _ROOT)}")
        print(f"    extrapolated archive for 1051 dates: "
              f"{os.path.getsize(p)*1051/1e6:.1f} MB")
    if len(results) > 1:
        best = min(results, key=results.get)
        print(f"\n  SNAPSHOT TIME A/B: " +
              "  ".join(f"{t}={d:.4f}" for t, d in results.items()) +
              f"  -> closest to live: {best}Z")
    return 0


def cmd_canary(key, args, conn):
    print(f"CANARY sweep: {len(CANARY_DATES)} dates, one per season "
          f"({len(CANARY_DATES)*HISTORICAL_COST} credits)")
    conn_games = sqlite3.connect(DB)
    all_raw_names, failures = Counter(), []
    for d in CANARY_DATES:
        sf.UNMAPPED_ODDS_TEAMS.clear()
        ok, meta, rows, before, after = fetch_one(key, d, args.snapshot_time or DEFAULT_SNAPSHOT, conn)
        if not ok:
            print(f"  {d}: FAILED http={meta.get('status')} {meta.get('error')}")
            failures.append(d)
            continue
        primary = [r for r in rows if r["game_date_et"] == d]
        for r in rows:
            all_raw_names[r["away_team_raw"]] += 1
            all_raw_names[r["home_team_raw"]] += 1
        ymd = d.replace("-", "")
        n_db = conn_games.execute(
            "SELECT COUNT(*) FROM games WHERE date=?", (ymd,)).fetchone()[0]
        if not n_db:
            log = json.load(open(PRED_LOG))
            n_db = len([e for e in log.get(d, []) if e.get("game_type") != "S"])
        pct = (len(primary) / n_db * 100) if n_db else 0
        ovr = sum(r["overround"] for r in primary) / len(primary) if primary else 0
        nbk = sum(r["n_books"] for r in primary) / len(primary) if primary else 0
        flag = "OK " if (pct >= 90 and not sf.UNMAPPED_ODDS_TEAMS) else "!! "
        print(f"  {flag}{d}: primary={len(primary)} games_in_db={n_db} ({pct:.0f}%) "
              f"overround={ovr:.4f} books={nbk:.1f} "
              f"unmapped={dict(sf.UNMAPPED_ODDS_TEAMS) or '-'}")
        store_events(conn, d, f"{d}T{args.snapshot_time or DEFAULT_SNAPSHOT}:00Z", meta, rows)
        time.sleep(THROTTLE)
    conn_games.close()
    print(f"\n  distinct raw team names seen: {len(all_raw_names)}")
    unresolved = [n for n in all_raw_names if not sf.resolve_odds_team(n)]
    print(f"  UNRESOLVED: {unresolved or 'none'}")
    print(f"  failures: {failures or 'none'}")
    print("\n  GO" if not unresolved and not failures else "\n  NO-GO — fix before bulk run")
    return 0


def cmd_run(key, args, conn):
    """The bulk backfill. THE irreversible step -- guards are not optional."""
    snapshot = args.snapshot_time or DEFAULT_SNAPSHOT
    dates = [d for d in target_dates() if not os.path.exists(archive_path(d))]
    if args.max_calls:
        dates = dates[:args.max_calls]
    if not dates:
        print("Nothing to do -- every target date is already archived.")
        return 0

    # LAYER 1: pre-flight ceiling, using a FREE quota call.
    q = sf.get_odds_quota(key)
    remaining = q.get("remaining")
    need = len(dates) * HISTORICAL_COST
    print(f"dates to fetch : {len(dates)}  ({dates[0]} .. {dates[-1]})")
    print(f"cost estimate  : {need} credits @ {HISTORICAL_COST}/call")
    print(f"credits now    : {remaining}")
    print(f"budget floor   : {args.budget_floor}")
    print(f"after run      : {remaining - need if remaining is not None else '?'}")
    if remaining is None:
        print("ABORT: could not read remaining credits.")
        return 1
    if remaining - need < args.budget_floor:
        print(f"ABORT: {need} credits would breach the {args.budget_floor} reserve. "
              f"Reduce scope with --max-calls, or lower --budget-floor deliberately.")
        return 1

    t0, done, failed, credits_start = time.time(), 0, [], remaining
    for i, d in enumerate(dates, 1):
        # LAYER 2: per-call stop, using the SERVER's number, never a local estimate.
        rem = sf.get_last_odds_quota().get("remaining")
        if rem is not None and would_breach_floor(rem, args.budget_floor):
            print(f"\nSTOP: remaining={rem} would breach floor {args.budget_floor}. "
                  f"Resume later with the same command -- {len(dates)-done} dates left.")
            break
        sf.UNMAPPED_ODDS_TEAMS.clear()
        ok, meta, rows, before, after = fetch_one(key, d, snapshot, conn)
        if not ok:
            failed.append((d, meta.get("status")))
            print(f"  [{i}/{len(dates)}] {d} FAILED http={meta.get('status')}", flush=True)
            time.sleep(THROTTLE)
            continue
        n_all, n_prim = store_events(conn, d, f"{d}T{snapshot}:00Z", meta, rows)
        done += 1
        # LAYER 3: cost-divergence abort. If the real cost is not what we modelled,
        # continuing for hundreds more calls is exactly how the reserve disappears.
        if before is not None and after is not None:
            charged = before - after
            if charged != HISTORICAL_COST:
                print(f"\nABORT: {d} charged {charged} credits, expected "
                      f"{HISTORICAL_COST}. Cost model is wrong -- stopping.")
                break
        if i % 25 == 0 or i == len(dates):
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"  [{i}/{len(dates)}] {d}  events={n_all} primary={n_prim} "
                  f"credits_left={after}  {el/60:.1f}m elapsed, "
                  f"~{(len(dates)-i)/rate/60:.0f}m left", flush=True)
        time.sleep(THROTTLE)

    rem_now = sf.get_last_odds_quota().get("remaining")
    spent = (credits_start - rem_now) if rem_now is not None else None
    print(f"\ndone={done} failed={len(failed)} in {(time.time()-t0)/60:.1f}m")
    print(f"credits: {credits_start} -> {rem_now} (spent {spent}, "
          f"expected {done*HISTORICAL_COST})")
    if failed:
        print(f"failed dates: {failed[:20]}")
    return 0


def cmd_archive_only(conn):
    """Re-parse every archived snapshot into the DB. Costs ZERO credits."""
    dates = [d for d in target_dates() if os.path.exists(archive_path(d))]
    print(f"re-parsing {len(dates)} archived dates (0 credits)")
    sf.UNMAPPED_ODDS_TEAMS.clear()
    tot = prim = 0
    for i, d in enumerate(dates, 1):
        payload = archive_read(d)
        if payload is None:
            print(f"  {d}: unreadable archive")
            continue
        rows = sf.parse_odds_events(payload.get("data") or [])
        meta = {"timestamp": payload.get("timestamp")}
        a, p_ = store_events(conn, d, f"{d}T{DEFAULT_SNAPSHOT}:00Z", meta, rows)
        tot += a; prim += p_
        if i % 200 == 0:
            print(f"  [{i}/{len(dates)}] ...", flush=True)
    print(f"stored {tot} events ({prim} primary)")
    print(f"unmapped names: {dict(sf.UNMAPPED_ODDS_TEAMS) or 'none'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--snapshot-time", default=None)
    ap.add_argument("--budget-floor", type=int, default=5000)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--archive-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    init_odds_tables(verbose=False)
    conn = sqlite3.connect(DB)
    dates = target_dates()

    if args.status or args.dry_run:
        have = [d for d in dates if os.path.exists(archive_path(d))]
        print(f"target dates: {len(dates)}  archived: {len(have)}  "
              f"remaining: {len(dates)-len(have)}")
        print(f"cost to finish: {(len(dates)-len(have))*HISTORICAL_COST} credits")
        if dates:
            print(f"range: {dates[0]} .. {dates[-1]}")
        if not args.dry_run:
            key = load_odds_api_key()
            q = sf.get_odds_quota(key)
            print(f"quota: remaining={q.get('remaining')} used={q.get('used')} "
                  f"(key {key_fingerprint(key)})")
        return 0

    if args.archive_only:
        return cmd_archive_only(conn)

    key = load_odds_api_key(verbose=True)
    if not key:
        print("No API key. Set ODDS_API_KEY or write ~/.odds_api_key")
        return 1
    print(f"key fingerprint: {key_fingerprint(key)}")

    if args.probe:
        return cmd_probe(key, args, conn)
    if args.canary:
        return cmd_canary(key, args, conn)

    if args.archive_only:
        return cmd_archive_only(conn)
    return cmd_run(key, args, conn)


if __name__ == "__main__":
    sys.exit(main())
