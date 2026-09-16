"""
backfill_historical_spreads.py
-------------------------------
Archive-only historical SPREADS (run-line) snapshots from The Odds API.

The moneyline model has been validated exhaustively against the market (see
CLAUDE.md) and shown to have ~no edge. The run-line model (`home_covers` in
MLBModel.py) has NEVER been checked against real market prices -- schedule_fetcher
has only ever requested markets="h2h", live or historical. This script closes that
gap for the historical side, following the same irreplaceable-resource discipline
as backfill_historical_odds.py: historical calls are paid-plan-only and become
permanently unbuyable after a downgrade, so every raw envelope is archived to disk
FIRST, before any parsing/schema work. Parsing is free and can happen anytime after
the credits are gone; this script's only job is to not lose the one shot at the
data itself.

Deliberately does NOT touch the DB or the existing h2h archive/tables -- pure
gzip-to-disk, keyed the same way as the h2h archive (one file per date) but under
a separate directory, so there is zero risk to data that already cost money.

Usage:
    --status                      progress + credits, zero cost
    --dry-run                     resolve dates + cost, no HTTP
    --probe DATE                  one date, validate cost + payload shape
    --run [--budget-floor N] [--max-calls N]   the bulk archive
"""
import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone

_UPDATES = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UPDATES)
sys.path.insert(0, _UPDATES)

ARCHIVE = os.path.join(_ROOT, "Databases_and_logs", "odds_archive_spreads")

from odds_key import load_odds_api_key, key_fingerprint          # noqa: E402
import schedule_fetcher as sf                                    # noqa: E402
from backfill_historical_odds import target_dates, DEFAULT_SNAPSHOT  # noqa: E402

MARKETS = "spreads"
COST_PER_CALL = 10   # 10 x 1 market x 1 region; asserted at runtime
THROTTLE = 0.25


def _now():
    return datetime.now(timezone.utc).isoformat()


def archive_path(date_str):
    return os.path.join(ARCHIVE, date_str[:4], f"{date_str}.json.gz")


def archive_write(date_str, payload):
    p = archive_path(date_str)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return p


def cmd_status(dates, dry_run):
    have = [d for d in dates if os.path.exists(archive_path(d))]
    print(f"target dates: {len(dates)}  archived: {len(have)}  "
          f"remaining: {len(dates) - len(have)}")
    print(f"cost to finish: {(len(dates) - len(have)) * COST_PER_CALL} credits")
    if dates:
        print(f"range: {dates[0]} .. {dates[-1]}")
    if not dry_run:
        key = load_odds_api_key()
        q = sf.get_odds_quota(key)
        print(f"quota: remaining={q.get('remaining')} used={q.get('used')} "
              f"(key {key_fingerprint(key)})")
    return 0


def cmd_probe(key, date_str, snapshot_time):
    ts = f"{date_str}T{snapshot_time}:00Z"
    before = sf.get_odds_quota(key).get("remaining")
    meta, payload, _rows = sf.get_historical_mlb_odds(key, ts, markets=MARKETS)
    after = sf.get_odds_quota(key).get("remaining")
    charged = (before - after) if (before is not None and after is not None) else None
    print(f"PROBE {date_str}  http={meta.get('status')} charged={charged} "
          f"(expected {COST_PER_CALL})")
    if not meta.get("http_ok"):
        print(f"  ERROR {meta.get('error')}")
        return 1
    if charged != COST_PER_CALL:
        print(f"  FATAL: cost model wrong. ABORT before any bulk run.")
        return 2
    data = payload.get("data") or []
    n_with_spreads = sum(
        1 for e in data
        if any(m.get("key") == "spreads" for bm in e.get("bookmakers", [])
               for m in bm.get("markets", [])))
    print(f"  events={len(data)}  events_with_a_spread_price={n_with_spreads}")
    p = archive_write(date_str, payload)
    print(f"  archived -> {os.path.relpath(p, _ROOT)} ({os.path.getsize(p)} bytes)")
    return 0


def cmd_run(key, dates, budget_floor, max_calls, snapshot_time):
    todo = [d for d in dates if not os.path.exists(archive_path(d))]
    if max_calls:
        todo = todo[:max_calls]
    if not todo:
        print("Nothing to do -- every target date is already archived.")
        return 0

    q = sf.get_odds_quota(key)
    remaining = q.get("remaining")
    need = len(todo) * COST_PER_CALL
    print(f"dates to fetch : {len(todo)}  ({todo[0]} .. {todo[-1]})")
    print(f"cost estimate  : {need} credits @ {COST_PER_CALL}/call")
    print(f"credits now    : {remaining}")
    print(f"budget floor   : {budget_floor}")
    if remaining is None:
        print("ABORT: could not read remaining credits.")
        return 1
    if remaining - need < budget_floor:
        print(f"ABORT: {need} credits would breach the {budget_floor} reserve. "
              f"Reduce scope with --max-calls.")
        return 1

    t0, done, failed = time.time(), 0, []
    for i, d in enumerate(todo, 1):
        rem = sf.get_last_odds_quota().get("remaining")
        if rem is not None and rem - COST_PER_CALL < budget_floor:
            print(f"\nSTOP: remaining={rem} would breach floor {budget_floor}. "
                  f"Resume later with the same command -- {len(todo) - done} left.")
            break
        ts = f"{d}T{snapshot_time}:00Z"
        before = sf.get_last_odds_quota().get("remaining")
        meta, payload, _rows = sf.get_historical_mlb_odds(key, ts, markets=MARKETS)
        after = sf.get_last_odds_quota().get("remaining")
        if not meta.get("http_ok"):
            failed.append((d, meta.get("status")))
            print(f"  [{i}/{len(todo)}] {d} FAILED http={meta.get('status')}", flush=True)
            time.sleep(THROTTLE)
            continue
        if before is not None and after is not None and (before - after) != COST_PER_CALL:
            print(f"\nABORT: {d} charged {before-after} credits, expected "
                  f"{COST_PER_CALL}. Cost model is wrong -- stopping.")
            break
        archive_write(d, payload)
        done += 1
        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"  [{i}/{len(todo)}] {d}  credits_left={after}  "
                  f"{el/60:.1f}m elapsed, ~{(len(todo)-i)/rate/60:.0f}m left", flush=True)
        time.sleep(THROTTLE)

    rem_now = sf.get_last_odds_quota().get("remaining")
    print(f"\ndone={done} failed={len(failed)} in {(time.time()-t0)/60:.1f}m")
    print(f"credits: {remaining} -> {rem_now}")
    if failed:
        print(f"failed dates: {failed[:20]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--snapshot-time", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--budget-floor", type=int, default=5000)
    ap.add_argument("--max-calls", type=int, default=None)
    args = ap.parse_args()

    dates = target_dates()

    if args.status or args.dry_run:
        return cmd_status(dates, args.dry_run)

    key = load_odds_api_key(verbose=True)
    if not key:
        print("No API key. Set ODDS_API_KEY or write ~/.odds_api_key")
        return 1

    if args.probe:
        return cmd_probe(key, args.probe, args.snapshot_time)

    if args.run:
        return cmd_run(key, dates, args.budget_floor, args.max_calls, args.snapshot_time)

    print("Nothing to do -- pass --status, --dry-run, --probe DATE, or --run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
