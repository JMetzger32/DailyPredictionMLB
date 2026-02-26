"""
app.py — DailyPredictionMLB Flask web server.

Routes:
  GET  /                  → serves the main predictions page
  GET  /accuracy          → serves the season accuracy leaderboard page
  GET  /api/predictions   → returns today's game predictions as JSON
  GET  /api/accuracy      → returns per-team prediction accuracy as JSON
  POST /api/refresh       → manually triggers update_daily baseline refresh

APScheduler runs update_daily.main() every day at 8 AM to refresh baselines,
update yesterday's results, and log today's predictions.

Run locally:
  pip3 install flask apscheduler requests
  python3 app.py
  Open: http://localhost:5000
"""

import json
import os
import pickle
import traceback
from datetime import date, datetime, timedelta

from flask import Flask, jsonify, render_template, request

from MLBModel import predict_game, _default_sp_stats
from schedule_fetcher import get_todays_schedule, get_game_results, find_pitcher_by_name, RETRO_TO_FULL_NAME

app = Flask(__name__)

ARTIFACTS_PATH    = os.environ.get("ARTIFACTS_PATH", "mlb_model_artifacts.pkl")
PREDICTIONS_LOG   = os.environ.get("PREDICTIONS_LOG", "predictions_log.json")

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
_artifacts = {}


def load_artifacts():
    global _artifacts
    with open(ARTIFACTS_PATH, "rb") as f:
        _artifacts = pickle.load(f)
    print(f"[app] Artifacts loaded — "
          f"{len(_artifacts.get('team_baselines', {}))} teams, "
          f"{len(_artifacts.get('sp_baselines', {}))} pitchers")


load_artifacts()


# ---------------------------------------------------------------------------
# Predictions log helpers
# ---------------------------------------------------------------------------
def _load_log():
    try:
        with open(PREDICTIONS_LOG) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_log(log):
    with open(PREDICTIONS_LOG, "w") as f:
        json.dump(log, f, indent=2)


def _build_prediction_entry(game, result):
    """Build a log entry dict from a schedule game + predict_game result."""
    return {
        "game_pk":          game.get("game_pk"),
        "date":             game.get("game_time_utc", "")[:10],
        "game_type":        game.get("game_type", "R"),
        "away_team":        game.get("away_team"),
        "away_team_name":   game.get("away_team_name"),
        "home_team":        game.get("home_team"),
        "home_team_name":   game.get("home_team_name"),
        "predicted_winner": result["predicted_winner"],
        "away_win_prob":    round(result["away_win_prob"], 4),
        "home_win_prob":    round(result["home_win_prob"], 4),
        "actual_winner":    None,
        "away_score":       None,
        "home_score":       None,
        "correct":          None,
    }


def update_yesterday_results():
    """Fetch yesterday's final scores and mark predictions correct/incorrect."""
    from schedule_fetcher import get_game_results
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    log = _load_log()
    if yesterday not in log:
        return
    try:
        results = get_game_results(date.fromisoformat(yesterday))
    except Exception as e:
        print(f"[app] update_yesterday_results failed: {e}")
        return

    changed = False
    for entry in log[yesterday]:
        if entry.get("correct") is not None:
            continue  # already resolved
        r = results.get(entry["game_pk"])
        if r and r["final"] and r["away_score"] is not None and r["home_score"] is not None:
            entry["away_score"] = r["away_score"]
            entry["home_score"] = r["home_score"]
            if r["home_score"] == r["away_score"]:
                entry["actual_winner"] = "Tie"
                entry["correct"] = None  # ties excluded from accuracy
            else:
                actual = "Home" if r["home_score"] > r["away_score"] else "Away"
                entry["actual_winner"] = actual
                entry["correct"] = (entry["predicted_winner"] == actual)
            changed = True

    if changed:
        _save_log(log)
        print(f"[app] Updated results for {yesterday}")


def log_todays_predictions():
    """Run today's predictions and save them to the log (skipped if already logged)."""
    today = date.today().isoformat()
    log = _load_log()
    if today in log:
        return  # already logged today

    team_baselines = _artifacts.get("team_baselines", {})
    sp_baselines   = _artifacts.get("sp_baselines", {})
    lr_model       = _artifacts.get("lr_model")
    scaler         = _artifacts.get("scaler")
    if lr_model is None:
        return

    games_raw = get_todays_schedule(date.today())
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
            entries.append(_build_prediction_entry(game, result))
        except Exception:
            continue

    log[today] = entries
    _save_log(log)
    print(f"[app] Logged {len(entries)} predictions for {today}")


# ---------------------------------------------------------------------------
# APScheduler — 8 AM daily refresh
# ---------------------------------------------------------------------------
def run_daily_update():
    """Wrapper called by APScheduler at 8 AM."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running scheduled daily update...")
    try:
        import update_daily
        update_daily.main()
        load_artifacts()
        update_yesterday_results()
        log_todays_predictions()
        print("[app] Daily update complete.")
    except Exception as e:
        print(f"[app] Daily update failed: {e}")
        traceback.print_exc()


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_daily_update, "cron", hour=8, minute=0)
    scheduler.start()
    print("[app] APScheduler started — daily update at 8:00 AM")
except ImportError:
    print("[app] apscheduler not installed — daily auto-refresh disabled")
    scheduler = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/accuracy")
def accuracy_page():
    return render_template("accuracy.html")


@app.route("/api/predictions")
def predictions():
    """
    Returns game predictions as JSON.
    Optional query param: ?date=YYYY-MM-DD  (defaults to today)
    For past dates, attaches actual results from predictions_log.json if available.
    """
    date_param = request.args.get("date")
    if date_param:
        try:
            target_date = date.fromisoformat(date_param)
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        target_date = date.today()

    team_baselines = _artifacts.get("team_baselines", {})
    sp_baselines   = _artifacts.get("sp_baselines", {})
    lr_model       = _artifacts.get("lr_model")
    scaler         = _artifacts.get("scaler")

    if lr_model is None:
        return jsonify({"error": "Model not loaded. Run MLBModel.py first."}), 500

    # Load any stored results for this date
    log = _load_log()
    date_str = target_date.isoformat()
    log_by_pk = {entry["game_pk"]: entry for entry in log.get(date_str, [])}

    # Fetch live scores so completed games show results immediately
    try:
        live_results = get_game_results(target_date)
    except Exception:
        live_results = {}

    games_raw = get_todays_schedule(target_date)
    predictions_out = []
    log_changed = False

    for game in games_raw:
        home = game.get("home_team")
        away = game.get("away_team")

        if not home or not away:
            continue

        home_ts = dict(team_baselines.get(home, {}))
        away_ts = dict(team_baselines.get(away, {}))

        if not home_ts or not away_ts:
            predictions_out.append({
                **game,
                "skipped": True,
                "skip_reason": f"No baseline for {home if not home_ts else away}",
            })
            continue

        home_sp_id = find_pitcher_by_name(game.get("home_pitcher_name"), sp_baselines)
        away_sp_id = find_pitcher_by_name(game.get("away_pitcher_name"), sp_baselines)

        home_sp = dict(sp_baselines[home_sp_id]) if home_sp_id and home_sp_id in sp_baselines else _default_sp_stats()
        away_sp = dict(sp_baselines[away_sp_id]) if away_sp_id and away_sp_id in sp_baselines else _default_sp_stats()

        home_sp_name = home_sp.get("name", game.get("home_pitcher_name") or "TBD")
        away_sp_name = away_sp.get("name", game.get("away_pitcher_name") or "TBD")

        try:
            result = predict_game(home_ts, away_ts, home_sp, away_sp, lr_model, scaler=scaler)
        except Exception as e:
            predictions_out.append({**game, "skipped": True, "skip_reason": str(e)})
            continue

        # Determine actual results — use stored log first, then overlay live scores
        pk     = game.get("game_pk")
        stored = log_by_pk.get(pk, {})
        live_r = live_results.get(pk, {})

        actual_winner = stored.get("actual_winner")
        away_score    = stored.get("away_score")
        home_score    = stored.get("home_score")
        correct       = stored.get("correct")

        # If not yet resolved in log but game is now final, use live score
        if (actual_winner is None
                and live_r.get("final")
                and live_r.get("away_score") is not None
                and live_r.get("home_score") is not None):
            away_score = live_r["away_score"]
            home_score = live_r["home_score"]
            if away_score == home_score:
                actual_winner = "Tie"
                correct       = None
            else:
                actual        = "Home" if home_score > away_score else "Away"
                actual_winner = actual
                correct       = (result["predicted_winner"] == actual)
            # Persist back to log so accuracy tracker stays current
            if pk in log_by_pk:
                log_by_pk[pk].update({
                    "away_score":    away_score,
                    "home_score":    home_score,
                    "actual_winner": actual_winner,
                    "correct":       correct,
                })
            else:
                # Game wasn't logged yet (8 AM job missed) — create full entry now
                entry = _build_prediction_entry(game, result)
                entry.update({
                    "away_score":    away_score,
                    "home_score":    home_score,
                    "actual_winner": actual_winner,
                    "correct":       correct,
                })
                log_by_pk[pk] = entry
            log_changed = True

        predictions_out.append({
            **game,
            "skipped": False,
            "home_win_prob":    result["home_win_prob"],
            "away_win_prob":    result["away_win_prob"],
            "predicted_winner": result["predicted_winner"],
            "confidence":       result["confidence"],
            "home_sp_name":     home_sp_name,
            "away_sp_name":     away_sp_name,
            "home_sp_era":      round(home_sp.get("era",   4.0), 2),
            "home_sp_xfip":     round(home_sp.get("xfip",  4.0), 2),
            "home_sp_siera":    round(home_sp.get("siera", 4.0), 2),
            "away_sp_era":      round(away_sp.get("era",   4.0), 2),
            "away_sp_xfip":     round(away_sp.get("xfip",  4.0), 2),
            "away_sp_siera":    round(away_sp.get("siera", 4.0), 2),
            "home_obp":         round(home_ts.get("obp",            0.318), 3),
            "home_slg":         round(home_ts.get("slg",            0.400), 3),
            "home_hits_pg":     round(home_ts.get("hits_per_game",  8.5),   1),
            "home_runs_pg":     round(home_ts.get("recent_runs_per_game", 4.5), 1),
            "home_bp_era":      round(home_ts.get("bullpen_era",    4.20),  2),
            "away_obp":         round(away_ts.get("obp",            0.318), 3),
            "away_slg":         round(away_ts.get("slg",            0.400), 3),
            "away_hits_pg":     round(away_ts.get("hits_per_game",  8.5),   1),
            "away_runs_pg":     round(away_ts.get("recent_runs_per_game", 4.5), 1),
            "away_bp_era":      round(away_ts.get("bullpen_era",    4.20),  2),
            "actual_winner":    actual_winner,
            "away_score":       away_score,
            "home_score":       home_score,
            "correct":          correct,
        })

    # Persist any live results we just resolved back to the log
    if log_changed:
        log[date_str] = list(log_by_pk.values())
        _save_log(log)

    return jsonify({
        "date":         target_date.isoformat(),
        "games":        predictions_out,
        "last_updated": datetime.now().isoformat(),
    })


@app.route("/api/accuracy")
def accuracy():
    """
    Returns per-team season accuracy from predictions_log.json.
    Splits regular season (game_type=R) and spring training (game_type=S).
    """
    log = _load_log()

    def compute_stats(entries):
        team_stats = {}
        total_games = total_correct = 0
        for entry in entries:
            if entry.get("correct") is None:
                continue  # game not yet resolved
            total_games += 1
            if entry["correct"]:
                total_correct += 1
            for side in ("away", "home"):
                team_code = entry.get(f"{side}_team")
                team_name = entry.get(f"{side}_team_name") or RETRO_TO_FULL_NAME.get(team_code, team_code)
                if not team_code:
                    continue
                if team_code not in team_stats:
                    team_stats[team_code] = {"team": team_code, "name": team_name, "games": 0, "correct": 0}
                team_stats[team_code]["games"] += 1
                if entry["correct"]:
                    team_stats[team_code]["correct"] += 1

        teams_list = sorted(
            [
                {**s, "accuracy": round(s["correct"] / s["games"], 3) if s["games"] else 0}
                for s in team_stats.values()
            ],
            key=lambda x: (-x["accuracy"], -x["games"])
        )
        return {
            "overall": {
                "games":    total_games,
                "correct":  total_correct,
                "accuracy": round(total_correct / total_games, 3) if total_games else 0,
            },
            "teams": teams_list,
        }

    # Separate by game type
    regular = []
    spring  = []
    for day_entries in log.values():
        for entry in day_entries:
            if entry.get("game_type") == "S":
                spring.append(entry)
            else:
                regular.append(entry)

    return jsonify({
        "regular_season":  compute_stats(regular),
        "spring_training": compute_stats(spring),
        "last_updated":    datetime.now().isoformat(),
    })


@app.route("/api/refresh", methods=["POST"])
def refresh():
    """Manually trigger a baseline refresh (calls update_daily.main())."""
    try:
        import update_daily
        update_daily.main()
        load_artifacts()
        return jsonify({"status": "ok", "updated_at": datetime.now().isoformat()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/teams")
def teams():
    """Return list of available teams with their baselines."""
    team_baselines = _artifacts.get("team_baselines", {})
    return jsonify({
        code: {
            "name":    RETRO_TO_FULL_NAME.get(code, code),
            "win_pct": round(info.get("win_pct", 0.5), 3),
            "obp":     round(info.get("obp",     0.318), 3),
            "slg":     round(info.get("slg",     0.400), 3),
            "bp_era":  round(info.get("bullpen_era", 4.20), 2),
        }
        for code, info in sorted(team_baselines.items())
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[app] Starting DailyPredictionMLB on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
