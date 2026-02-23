"""
app.py — DailyPredictionMLB Flask web server.

Routes:
  GET  /                  → serves the main predictions page
  GET  /api/predictions   → returns today's game predictions as JSON
  POST /api/refresh       → manually triggers update_daily baseline refresh

APScheduler runs update_daily.main() every day at 8 AM to refresh baselines.

Run locally:
  pip3 install flask apscheduler requests
  python3 app.py
  Open: http://localhost:5000
"""

import os
import pickle
import traceback
from datetime import date, datetime

from flask import Flask, jsonify, render_template, request

from MLBModel import predict_game, _default_sp_stats
from schedule_fetcher import get_todays_schedule, find_pitcher_by_name

app = Flask(__name__)

ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", "mlb_model_artifacts.pkl")

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
# APScheduler — 8 AM daily baseline refresh
# ---------------------------------------------------------------------------
def run_daily_update():
    """Wrapper called by APScheduler at 8 AM."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running scheduled daily update...")
    try:
        import update_daily
        update_daily.main()
        load_artifacts()
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


@app.route("/api/predictions")
def predictions():
    """
    Returns today's game predictions as JSON.
    Optional query param: ?date=YYYY-MM-DD  (defaults to today)
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

    games_raw = get_todays_schedule(target_date)
    predictions_out = []

    for game in games_raw:
        home = game.get("home_team")
        away = game.get("away_team")

        # Skip if we can't map the team to Retrosheet codes
        if not home or not away:
            continue

        home_ts = dict(team_baselines.get(home, {}))
        away_ts = dict(team_baselines.get(away, {}))

        # If team not in baselines, skip (pre-season or expansion)
        if not home_ts or not away_ts:
            predictions_out.append({
                **game,
                "skipped": True,
                "skip_reason": f"No baseline for {home if not home_ts else away}",
            })
            continue

        # Match probable pitcher names to sp_baselines IDs
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

        predictions_out.append({
            **game,
            "skipped": False,
            # Prediction
            "home_win_prob":   result["home_win_prob"],
            "away_win_prob":   result["away_win_prob"],
            "predicted_winner": result["predicted_winner"],
            "confidence":      result["confidence"],
            # Pitcher display
            "home_sp_name":    home_sp_name,
            "away_sp_name":    away_sp_name,
            "home_sp_era":     round(home_sp.get("era",   4.0), 2),
            "home_sp_xfip":    round(home_sp.get("xfip",  4.0), 2),
            "home_sp_siera":   round(home_sp.get("siera", 4.0), 2),
            "away_sp_era":     round(away_sp.get("era",   4.0), 2),
            "away_sp_xfip":    round(away_sp.get("xfip",  4.0), 2),
            "away_sp_siera":   round(away_sp.get("siera", 4.0), 2),
            # Team stats
            "home_obp":        round(home_ts.get("obp",            0.318), 3),
            "home_slg":        round(home_ts.get("slg",            0.400), 3),
            "home_hits_pg":    round(home_ts.get("hits_per_game",  8.5),   1),
            "home_runs_pg":    round(home_ts.get("recent_runs_per_game", 4.5), 1),
            "home_bp_era":     round(home_ts.get("bullpen_era",    4.20),  2),
            "away_obp":        round(away_ts.get("obp",            0.318), 3),
            "away_slg":        round(away_ts.get("slg",            0.400), 3),
            "away_hits_pg":    round(away_ts.get("hits_per_game",  8.5),   1),
            "away_runs_pg":    round(away_ts.get("recent_runs_per_game", 4.5), 1),
            "away_bp_era":     round(away_ts.get("bullpen_era",    4.20),  2),
        })

    return jsonify({
        "date":  target_date.isoformat(),
        "games": predictions_out,
        "last_updated": datetime.now().isoformat(),
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
    from schedule_fetcher import RETRO_TO_FULL_NAME
    return jsonify({
        code: {
            "name": RETRO_TO_FULL_NAME.get(code, code),
            "win_pct":   round(info.get("win_pct", 0.5), 3),
            "obp":       round(info.get("obp",     0.318), 3),
            "slg":       round(info.get("slg",     0.400), 3),
            "bp_era":    round(info.get("bullpen_era", 4.20), 2),
        }
        for code, info in sorted(team_baselines.items())
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[app] Starting DailyPredictionMLB on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
