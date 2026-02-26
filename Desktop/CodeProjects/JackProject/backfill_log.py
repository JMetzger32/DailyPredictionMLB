"""
backfill_log.py — One-time script to backfill predictions_log.json
for past dates where the logging system wasn't yet running.

Usage:
    python3 backfill_log.py                           # backfill last 5 days
    python3 backfill_log.py --days 10                 # backfill last N days
    python3 backfill_log.py --start 2026-02-20 --end 2026-02-24
"""

import argparse
import json
import os
import pickle
from datetime import date, timedelta

from MLBModel import _default_sp_stats, predict_game
from schedule_fetcher import find_pitcher_by_name, get_game_results, get_todays_schedule

ARTIFACTS_PATH  = os.environ.get("ARTIFACTS_PATH", "mlb_model_artifacts.pkl")
PREDICTIONS_LOG = os.environ.get("PREDICTIONS_LOG", "predictions_log.json")


def _load_artifacts():
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


def _load_log():
    try:
        with open(PREDICTIONS_LOG) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_log(log):
    with open(PREDICTIONS_LOG, "w") as f:
        json.dump(log, f, indent=2)


def backfill_date(target_date, artifacts, log):
    date_str = target_date.isoformat()

    if date_str in log:
        print(f"  {date_str}: already in log, skipping")
        return 0

    team_baselines = artifacts.get("team_baselines", {})
    sp_baselines   = artifacts.get("sp_baselines", {})
    lr_model       = artifacts.get("lr_model")
    scaler         = artifacts.get("scaler")

    if lr_model is None:
        print("  ERROR: model not loaded")
        return 0

    games_raw = get_todays_schedule(target_date)
    if not games_raw:
        print(f"  {date_str}: no games found")
        return 0

    results = get_game_results(target_date)

    entries = []
    for game in games_raw:
        home = game.get("home_team")
        away = game.get("away_team")
        if not home or not away:
            continue

        home_ts = dict(team_baselines.get(home, {}))
        away_ts = dict(team_baselines.get(away, {}))
        if not home_ts or not away_ts:
            continue

        home_sp_id = find_pitcher_by_name(game.get("home_pitcher_name"), sp_baselines)
        away_sp_id = find_pitcher_by_name(game.get("away_pitcher_name"), sp_baselines)
        home_sp = dict(sp_baselines[home_sp_id]) if home_sp_id and home_sp_id in sp_baselines else _default_sp_stats()
        away_sp = dict(sp_baselines[away_sp_id]) if away_sp_id and away_sp_id in sp_baselines else _default_sp_stats()

        try:
            result = predict_game(home_ts, away_ts, home_sp, away_sp, lr_model, scaler=scaler)
        except Exception as e:
            print(f"    Skipping {away} @ {home}: {e}")
            continue

        entry = {
            "game_pk":          game.get("game_pk"),
            "date":             date_str,
            "game_type":        game.get("game_type", "R"),
            "away_team":        away,
            "away_team_name":   game.get("away_team_name"),
            "home_team":        home,
            "home_team_name":   game.get("home_team_name"),
            "predicted_winner": result["predicted_winner"],
            "away_win_prob":    round(result["away_win_prob"], 4),
            "home_win_prob":    round(result["home_win_prob"], 4),
            "actual_winner":    None,
            "away_score":       None,
            "home_score":       None,
            "correct":          None,
        }

        # Attach actual results if the game is already final
        r = results.get(game.get("game_pk"))
        if r and r["final"] and r["away_score"] is not None and r["home_score"] is not None:
            entry["away_score"]    = r["away_score"]
            entry["home_score"]    = r["home_score"]
            if r["home_score"] == r["away_score"]:
                entry["actual_winner"] = "Tie"
                entry["correct"]       = None  # ties excluded from accuracy
            else:
                actual                 = "Home" if r["home_score"] > r["away_score"] else "Away"
                entry["actual_winner"] = actual
                entry["correct"]       = (entry["predicted_winner"] == actual)

        entries.append(entry)

    log[date_str] = entries
    resolved = sum(1 for e in entries if e["correct"] is not None)
    correct   = sum(1 for e in entries if e["correct"])
    print(f"  {date_str}: {len(entries)} games logged, {resolved} resolved ({correct}/{resolved} correct)")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Backfill predictions log for past dates")
    parser.add_argument("--days",  type=int, default=5,    help="Number of past days to backfill (default 5)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, default=None, help="End date YYYY-MM-DD (inclusive)")
    args = parser.parse_args()

    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
    else:
        end   = date.today() - timedelta(days=1)
        start = end - timedelta(days=args.days - 1)

    print(f"Backfilling {start} → {end}")
    print("Loading artifacts...")
    artifacts = _load_artifacts()
    log = _load_log()

    total = 0
    d = start
    while d <= end:
        total += backfill_date(d, artifacts, log)
        d += timedelta(days=1)

    _save_log(log)
    print(f"\nDone. {total} total games logged. Saved to {PREDICTIONS_LOG}")


if __name__ == "__main__":
    main()
