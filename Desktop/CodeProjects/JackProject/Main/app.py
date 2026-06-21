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
import sys
import pickle
import time
import traceback
import numpy as np
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

# __file__-based paths so this module works regardless of working directory
_HERE = os.path.dirname(os.path.abspath(__file__))   # Main/
_ROOT = os.path.dirname(_HERE)                        # JackProject/

# Ensure helpers are importable: Main/ for MLBModel, updates/ for schedule_fetcher etc.
sys.path.insert(0, os.path.join(_ROOT, "updates"))
sys.path.insert(0, _HERE)

_ET = ZoneInfo("America/New_York")

def _today_et():
    """Return the current prediction date in US Eastern time.
    Before 7 AM ET, returns yesterday — predictions aren't posted yet and the
    page shouldn't flip away from the previous day's games until morning lines load."""
    now = datetime.now(tz=_ET)
    if now.hour < 7:
        return (now - timedelta(days=1)).date()
    return now.date()

from flask import Flask, jsonify, render_template, request

from MLBModel import predict_game, _default_sp_stats
from schedule_fetcher import get_todays_schedule, get_game_results, get_schedule_and_results, get_mlb_odds, get_team_standings, find_pitcher_by_name, RETRO_TO_FULL_NAME

app = Flask(__name__,
            template_folder=os.path.join(_ROOT, "templates"),
            static_folder=os.path.join(_ROOT, "static"))

ARTIFACTS_PATH      = os.environ.get("ARTIFACTS_PATH",       os.path.join(_ROOT, "updates",            "mlb_model_artifacts.pkl"))
PREDICTIONS_LOG     = os.environ.get("PREDICTIONS_LOG",      os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json"))
BETTING_LOG_PATH    = os.environ.get("BETTING_LOG_PATH",     os.path.join(_ROOT, "Databases_and_logs", "betting_log.json"))
CLOSING_ODDS_LOG    = os.environ.get("CLOSING_ODDS_LOG",     os.path.join(_ROOT, "Databases_and_logs", "closing_odds_log.json"))
SUBSCRIBERS_PATH    = os.environ.get("SUBSCRIBERS_PATH",     os.path.join(_ROOT, "Databases_and_logs", "subscribers.json"))
PICKS_LOG_PATH      = os.environ.get("PICKS_LOG_PATH",       os.path.join(_ROOT, "Databases_and_logs", "picks_log.json"))
RESEND_API_KEY    = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL        = os.environ.get("FROM_EMAIL", "onboarding@resend.dev")
TRIGGER_SECRET    = os.environ.get("TRIGGER_SECRET", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO       = os.environ.get("GITHUB_REPO", "JMetzger32/DailyPredictionMLB")

def _github_path(filepath):
    """Return the path to use in GitHub API calls (repo-root-relative).
    Uses _ROOT as the repo base so absolute paths are normalised correctly."""
    try:
        rel = os.path.relpath(os.path.abspath(filepath), _ROOT).replace("\\", "/")
        return rel
    except Exception:
        return os.path.basename(filepath)


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

# Log artifact status for debugging accuracy discrepancies
_team_count = len(_artifacts.get('team_baselines', {}))
_sp_count = len(_artifacts.get('sp_baselines', {}))
_lr = _artifacts.get('lr_model')
print(f"[app] Artifacts loaded — {_team_count} teams, {_sp_count} pitchers, LR model: {_lr is not None}", flush=True)

# Refresh SP baselines from the live MLB Stats API immediately after loading artifacts.
# This ensures pitcher ERA/WHIP/FIP display always reflects the current 2026 season,
# even when a locally-retrained pkl is restored from GitHub on Render startup.
try:
    import update_daily as _ud
    _prior_sp = _artifacts.get("sp_baselines", {})
    print("[startup] Refreshing SP baselines from MLB Stats API...", flush=True)
    _fresh_sp = _ud.fetch_sp_baselines_from_mlb_api(_ud.SEASON, games_played=70, prior_sp=_prior_sp)
    if _fresh_sp and len(_fresh_sp) >= 5:
        _artifacts["sp_baselines"] = {**_prior_sp, **_fresh_sp}
        print(f"[startup] SP baselines refreshed: {len(_fresh_sp)} pitchers with live {_ud.SEASON} data", flush=True)
    else:
        print("[startup] SP refresh returned no data — using pkl baselines", flush=True)
except Exception as _sp_err:
    print(f"[startup] SP refresh failed: {_sp_err} — using pkl baselines", flush=True)

# ---------------------------------------------------------------------------
# Schedule cache  (avoids hitting the MLB API on every page load)
# ---------------------------------------------------------------------------
_schedule_cache: dict = {}   # {date_str: {"ts": float, "games": list, "results": dict}}
_CACHE_TTL      = 120        # seconds — today / future dates
_CACHE_TTL_PAST = 600        # seconds — historical dates (scores won't change)

ODDS_API_KEY    = os.environ.get("ODDS_API_KEY", "")
_odds_cache: dict = {}       # {date_str: {"ts": float, "odds": dict}}
_ODDS_CACHE_TTL = 1800       # 30 minutes — saves API credits


def _get_schedule_cached(target_date):
    """Return (games, live_results) from cache or MLB API."""
    date_str = target_date.isoformat()
    cached   = _schedule_cache.get(date_str)
    ttl      = _CACHE_TTL if target_date >= _today_et() else _CACHE_TTL_PAST
    if cached and (time.monotonic() - cached["ts"]) < ttl:
        return cached["games"], cached["results"]
    games, results = get_schedule_and_results(target_date)
    _schedule_cache[date_str] = {"ts": time.monotonic(), "games": games, "results": results}
    return games, results


def _get_odds_cached():
    """Return today's odds map from cache or The Odds API (1 credit per call)."""
    today  = _today_et().isoformat()
    cached = _odds_cache.get(today)
    if cached and (time.monotonic() - cached["ts"]) < _ODDS_CACHE_TTL:
        return cached["odds"]
    if not ODDS_API_KEY:
        print("[odds] ODDS_API_KEY not set — skipping odds fetch", flush=True)
        return {}
    odds = get_mlb_odds(ODDS_API_KEY)
    print(f"[odds] Fetched {len(odds)} games from The Odds API for {today}", flush=True)
    _odds_cache[today] = {"ts": time.monotonic(), "odds": odds}
    return odds


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


# ---------------------------------------------------------------------------
# Betting log — separate persistent store for entries that have odds data.
# Never touched by backfills; backed up to GitHub independently.
# ---------------------------------------------------------------------------
def _load_betting_log():
    try:
        with open(BETTING_LOG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_betting_log(blog):
    with open(BETTING_LOG_PATH, "w") as f:
        json.dump(blog, f, indent=2)


def _push_betting_log_to_github():
    _push_file_to_github(BETTING_LOG_PATH, f"Auto-backup betting log {_today_et().isoformat()}")


def _push_closing_odds_to_github():
    _push_file_to_github(CLOSING_ODDS_LOG, f"Auto-backup closing odds {_today_et().isoformat()}")


def _upsert_betting_entries(entries):
    """Write entries that have odds data into betting_log table.
    Only saves entries with bet_rating set (good/bad/unsure).
    Upserts by game_pk to avoid duplicates."""
    odds_entries = [e for e in entries if e.get("bet_rating") is not None]
    if not odds_entries:
        return

    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db"))
        cur = conn.cursor()

        for entry in odds_entries:
            pk = entry.get("game_pk")
            if not pk:
                continue

            # Upsert by game_pk
            cur.execute("""
                INSERT OR REPLACE INTO betting_log (
                    game_pk, date, game_type, away_team, home_team,
                    predicted_winner, away_win_prob, home_win_prob,
                    away_ml, home_ml, away_implied, home_implied,
                    bet_rating, model_edge, predicted_team_ml,
                    predicted_total, actual_winner, away_score, home_score,
                    correct, closing_away_ml, closing_home_ml, clv,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                pk,
                entry.get("date"),
                entry.get("game_type"),
                entry.get("away_team"),
                entry.get("home_team"),
                entry.get("predicted_winner"),
                entry.get("away_win_prob"),
                entry.get("home_win_prob"),
                entry.get("away_ml"),
                entry.get("home_ml"),
                entry.get("away_implied"),
                entry.get("home_implied"),
                entry.get("bet_rating"),
                entry.get("model_edge"),
                entry.get("predicted_team_ml"),
                entry.get("predicted_total"),
                entry.get("actual_winner"),
                entry.get("away_score"),
                entry.get("home_score"),
                entry.get("correct"),
                entry.get("closing_away_ml"),
                entry.get("closing_home_ml"),
                entry.get("clv"),
            ))

        conn.commit()
        conn.close()
        print(f"[betting] Upserted {len(odds_entries)} entries to betting_log table", flush=True)
    except Exception as e:
        print(f"[betting] Failed to upsert to betting_log table: {e}", flush=True)
        # Fallback to JSON for backward compatibility
        blog = _load_betting_log()
        changed = False
        for entry in odds_entries:
            date_str = entry.get("date")
            if not date_str:
                continue
            if date_str not in blog:
                blog[date_str] = []
            existing = {e["game_pk"]: e for e in blog[date_str] if e.get("game_pk")}
            pk = entry.get("game_pk")
            if pk and pk in existing:
                existing[pk].update(entry)
            else:
                blog[date_str].append(entry)
            changed = True
        if changed:
            _save_betting_log(blog)


def _resolve_betting_log_results(date_str, results):
    """Update correct/scores for betting_log entries on date_str using results dict."""
    if not results:
        return

    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db"))
        cur = conn.cursor()

        # Get unresolved entries from this date
        cur.execute("""
            SELECT game_pk, predicted_winner FROM betting_log
            WHERE date = ? AND correct IS NULL
        """, (date_str,))

        updated_count = 0
        for (game_pk, predicted_winner) in cur.fetchall():
            r = results.get(game_pk)
            if r and r["final"] and r["away_score"] is not None:
                away_score = r["away_score"]
                home_score = r["home_score"]

                if home_score == away_score:
                    actual_winner = "Tie"
                    correct = None
                else:
                    actual_winner = "Home" if home_score > away_score else "Away"
                    correct = 1 if (predicted_winner == actual_winner) else 0

                cur.execute("""
                    UPDATE betting_log
                    SET actual_winner = ?, away_score = ?, home_score = ?, correct = ?, updated_at = datetime('now')
                    WHERE game_pk = ?
                """, (actual_winner, away_score, home_score, correct, game_pk))

                updated_count += 1

        if updated_count > 0:
            conn.commit()
            print(f"[betting] Resolved {updated_count} results in betting_log for {date_str}", flush=True)

        conn.close()
    except Exception as e:
        print(f"[betting] Failed to resolve results in betting_log table: {e}", flush=True)


# ---------------------------------------------------------------------------
# Closing odds archive (for historical odds persistence)
# ---------------------------------------------------------------------------
def _load_closing_odds_archive():
    """Load closing odds archive from JSON."""
    try:
        with open(CLOSING_ODDS_LOG) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_closing_odds_archive(archive):
    """Save closing odds archive to JSON."""
    with open(CLOSING_ODDS_LOG, "w") as f:
        json.dump(archive, f, indent=2)


def _get_closing_odds_archive():
    """Return closing odds from archive (for past dates)."""
    try:
        archive = _load_closing_odds_archive()
        return archive
    except Exception as e:
        print(f"[odds] Failed to load closing odds archive: {e}", flush=True)
        return {}


def _store_closing_odds_to_archive(date_str, odds_map):
    """Store odds snapshot for a date in closing_odds_archive."""
    if not odds_map:
        return
    try:
        archive = _load_closing_odds_archive()
        if date_str not in archive:
            archive[date_str] = {}

        # Store keyed by (away_team, home_team) tuple
        for (away_team, home_team), odds in odds_map.items():
            key = f"{away_team}|{home_team}"  # Use pipe separator for JSON keys
            archive[date_str][key] = {
                "away_ml": odds.get("away_ml"),
                "home_ml": odds.get("home_ml"),
                "away_implied": odds.get("away_implied"),
                "home_implied": odds.get("home_implied"),
            }

        _save_closing_odds_archive(archive)
        print(f"[odds] Stored closing odds for {date_str}: {len(odds_map)} games", flush=True)
    except Exception as e:
        print(f"[odds] Failed to store closing odds archive: {e}", flush=True)


# ---------------------------------------------------------------------------
# Subscriber helpers
# ---------------------------------------------------------------------------
def _load_subscribers():
    try:
        with open(SUBSCRIBERS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_subscribers(subs):
    with open(SUBSCRIBERS_PATH, "w") as f:
        json.dump(subs, f, indent=2)


# ---------------------------------------------------------------------------
# Picks log helpers
# ---------------------------------------------------------------------------
def _load_picks():
    try:
        with open(PICKS_LOG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_picks(picks):
    with open(PICKS_LOG_PATH, "w") as f:
        json.dump(picks, f, indent=2)
    _push_file_to_github(PICKS_LOG_PATH, f"Auto-backup picks log {_today_et().isoformat()}")


def _resolve_picks_for_date(target_date):
    """Update correct/actual_winner for all users' picks on target_date from the log."""
    date_str = target_date.isoformat()
    log = _load_log()
    log_entries = {e["game_pk"]: e for e in log.get(date_str, [])}
    if not log_entries:
        return
    picks = _load_picks()
    changed = False
    for user_picks in picks.values():
        for entry in user_picks.get(date_str, []):
            if entry.get("correct") is not None:
                continue
            log_e = log_entries.get(entry["game_pk"])
            if log_e and log_e.get("actual_winner") is not None:
                entry["actual_winner"] = log_e["actual_winner"]
                if log_e["actual_winner"] == "Tie":
                    entry["correct"] = None
                else:
                    entry["correct"] = (entry["pick"] == log_e["actual_winner"])
                changed = True
    if changed:
        _save_picks(picks)


def _calibration_bucket(prob):
    """Return the probability bucket string for a given win probability (50-100%)."""
    p = round(prob * 100)
    if p < 55:   return "50-55%"
    if p < 60:   return "55-60%"
    if p < 65:   return "60-65%"
    if p < 70:   return "65-70%"
    if p < 80:   return "70-80%"
    return "80%+"


def _compute_error_metrics(home_win_prob, actual_winner):
    """Compute Brier score and log loss once a game result is known."""
    import math
    actual = 1 if actual_winner == "Home" else 0
    p = max(min(home_win_prob, 0.9999), 0.0001)  # clip to avoid log(0)
    brier = round((p - actual) ** 2, 4)
    ll    = round(-(actual * math.log(p) + (1 - actual) * math.log(1 - p)), 4)
    return brier, ll


def _build_prediction_entry(game, result, odds_data=None):
    """Build a log entry dict from a schedule game + predict_game result."""
    odds_data = odds_data or {}
    # Use ET date so late-night games (9 PM ET = next UTC day) log correctly
    _utc = game.get("game_time_utc", "")
    try:
        _game_date = datetime.fromisoformat(_utc.replace("Z", "+00:00")).astimezone(_ET).date().isoformat()
    except Exception:
        _game_date = _utc[:10]
    return {
        "game_pk":           game.get("game_pk"),
        "date":              _game_date,
        "game_type":         game.get("game_type", "R"),
        "away_team":         game.get("away_team"),
        "away_team_name":    game.get("away_team_name"),
        "home_team":         game.get("home_team"),
        "home_team_name":    game.get("home_team_name"),
        "predicted_winner":  result["predicted_winner"],
        "away_win_prob":     round(result["away_win_prob"], 4),
        "home_win_prob":     round(result["home_win_prob"], 4),
        "confidence":        round(max(result["home_win_prob"], result["away_win_prob"]) - 0.5, 4),
        "calibration_bucket": _calibration_bucket(max(result["home_win_prob"], result["away_win_prob"])),
        "away_ml":           odds_data.get("away_ml"),
        "home_ml":           odds_data.get("home_ml"),
        "bet_rating":        odds_data.get("bet_rating"),
        "predicted_team_ml": odds_data.get("predicted_team_ml"),
        "model_edge":        odds_data.get("model_edge"),
        "actual_winner":     None,
        "away_score":        None,
        "home_score":        None,
        "correct":           None,
        "brier_score":       None,
        "log_loss":          None,
        "clv":               None,
        "predicted_total":   result.get("predicted_total"),
        "home_est_score":    result.get("home_est_score"),
        "away_est_score":    result.get("away_est_score"),
        "actual_total":      None,
        "ou_correct":        None,
        "x_scaled_features": result.get("x_scaled_features"),
    }


def update_yesterday_results():
    """Fetch yesterday's final scores and mark predictions correct/incorrect."""
    from schedule_fetcher import get_game_results
    yesterday = (_today_et() - timedelta(days=1)).isoformat()
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
        moneyline_done = entry.get("correct") is not None
        ou_done = entry.get("ou_correct") is not None
        if moneyline_done and ou_done:
            continue  # fully resolved
        r = results.get(entry["game_pk"])
        if r and r["final"] and r["away_score"] is not None and r["home_score"] is not None:
            if not moneyline_done:
                entry["away_score"] = r["away_score"]
                entry["home_score"] = r["home_score"]
                if r["home_score"] == r["away_score"]:
                    entry["actual_winner"] = "Tie"
                    entry["correct"] = None  # ties excluded from accuracy
                else:
                    actual = "Home" if r["home_score"] > r["away_score"] else "Away"
                    entry["actual_winner"] = actual
                    entry["correct"] = (entry["predicted_winner"] == actual)
                    # Error bounds: Brier score + log loss
                    if entry.get("home_win_prob") is not None:
                        bs, ll = _compute_error_metrics(entry["home_win_prob"], actual)
                        entry["brier_score"] = bs
                        entry["log_loss"]    = ll
                    # Run-line: did predicted team cover ±1.5?
                    home_covers = (r["home_score"] - r["away_score"]) > 1.5
                    hcp = entry.get("home_cover_prob")
                    if hcp is not None:
                        entry["correct_rl"] = ((hcp > 0.5) == home_covers)
            if not ou_done and entry.get("predicted_total") is not None:
                actual_total = r["home_score"] + r["away_score"]
                entry["actual_total"] = actual_total
                entry["ou_correct"] = abs(actual_total - entry["predicted_total"]) <= 2.0
            changed = True

    if changed:
        _save_log(log)
        _resolve_betting_log_results(yesterday, results)
        print(f"[app] Updated results for {yesterday}")


def _log_predictions_for_date(target_date, log=None):
    """
    Run predictions for target_date and add to log.
    For past dates, also immediately resolves results.
    Returns the log (possibly modified). Skips if date already in log.
    """
    date_str = target_date.isoformat()
    if log is None:
        log = _load_log()
    if date_str in log:
        return log  # already logged

    team_baselines = _artifacts.get("team_baselines", {})
    sp_baselines   = _artifacts.get("sp_baselines", {})
    lr_model       = _artifacts.get("lr_model")
    scaler         = _artifacts.get("scaler")
    gb_model       = _artifacts.get("gb_model")
    xgb_model      = _artifacts.get("xgb_model")
    if lr_model is None:
        return log

    games_raw = get_todays_schedule(target_date)
    # Fetch live odds for today/future; use closing odds archive for past dates
    if target_date >= _today_et():
        odds_map = _get_odds_cached()
    else:
        # For past dates, try to get from closing odds archive
        date_str = target_date.isoformat()
        closing_archive = _get_closing_odds_archive()
        date_odds = closing_archive.get(date_str, {})
        # Reconstruct odds dict from archive (keyed by team tuple)
        odds_map = {}
        for key_str, odds in date_odds.items():
            try:
                away_team, home_team = key_str.split("|")
                odds_map[(away_team, home_team)] = {
                    "away_ml": odds.get("away_ml"),
                    "home_ml": odds.get("home_ml"),
                    "away_implied": odds.get("away_implied"),
                    "home_implied": odds.get("home_implied"),
                }
            except Exception:
                pass
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
            result = predict_game(home_ts, away_ts, home_sp, away_sp, lr_model, scaler=scaler,
                                  gb_model=gb_model, xgb_model=xgb_model)
            odds_data = _compute_odds_fields(away, home, result, odds_map)
            entries.append(_build_prediction_entry(game, result, odds_data=odds_data))
        except Exception:
            continue

    if not entries:
        return log

    # For past dates, immediately attach final results
    if target_date < _today_et():
        try:
            from schedule_fetcher import get_game_results
            results = get_game_results(target_date)
            for entry in entries:
                r = results.get(entry["game_pk"])
                if r and r["final"] and r["away_score"] is not None:
                    entry["away_score"] = r["away_score"]
                    entry["home_score"] = r["home_score"]
                    if r["home_score"] == r["away_score"]:
                        entry["actual_winner"] = "Tie"
                        entry["correct"] = None
                    else:
                        actual = "Home" if r["home_score"] > r["away_score"] else "Away"
                        entry["actual_winner"] = actual
                        entry["correct"] = (entry["predicted_winner"] == actual)
                        if entry.get("home_win_prob") is not None:
                            bs, ll = _compute_error_metrics(entry["home_win_prob"], actual)
                            entry["brier_score"] = bs
                            entry["log_loss"]    = ll
        except Exception as e:
            print(f"[app] Could not resolve results for {date_str}: {e}")

    log[date_str] = entries
    # Persist any entries that have odds data into the separate betting log
    _upsert_betting_entries(entries)
    print(f"[app] Backfilled {len(entries)} predictions for {date_str}")
    return log


def _resolve_unresolved_for_date(log, target_date):
    """Resolve any unresolved entries for a specific past date. Returns True if changed."""
    from schedule_fetcher import get_game_results
    date_str = target_date.isoformat()
    entries = log.get(date_str, [])
    unresolved = [e for e in entries if e.get("correct") is None and e.get("actual_winner") is None]
    if not unresolved:
        return False

    # Build set of game_pks already resolved on OTHER dates (UTC-date duplicates).
    # If a game is already resolved elsewhere, remove the stale unresolved copy.
    resolved_pks = {
        e["game_pk"]
        for d, day_entries in log.items() if d != date_str
        for e in day_entries
        if e.get("game_pk") and e.get("actual_winner") is not None
    }
    stale = [e for e in unresolved if e.get("game_pk") in resolved_pks]
    if stale:
        stale_pks = {e["game_pk"] for e in stale}
        log[date_str] = [e for e in entries if e.get("game_pk") not in stale_pks]
        if not log[date_str]:
            del log[date_str]
        return True  # log changed (stale entries removed)

    try:
        results = get_game_results(target_date)
    except Exception as e:
        print(f"[app] get_game_results failed for {date_str}: {e}")
        return False
    changed = False
    for entry in unresolved:
        r = results.get(entry["game_pk"])
        if r and r["final"] and r["away_score"] is not None:
            entry["away_score"] = r["away_score"]
            entry["home_score"] = r["home_score"]
            if r["home_score"] == r["away_score"]:
                entry["actual_winner"] = "Tie"
                entry["correct"] = None
            else:
                actual = "Home" if r["home_score"] > r["away_score"] else "Away"
                entry["actual_winner"] = actual
                entry["correct"] = (entry["predicted_winner"] == actual)
                if entry.get("home_win_prob") is not None:
                    bs, ll = _compute_error_metrics(entry["home_win_prob"], actual)
                    entry["brier_score"] = bs
                    entry["log_loss"]    = ll
                home_covers = (r["home_score"] - r["away_score"]) > 1.5
                hcp = entry.get("home_cover_prob")
                if hcp is not None:
                    entry["correct_rl"] = ((hcp > 0.5) == home_covers)
            changed = True
    return changed


def _auto_heal_log(days=7):
    """
    For the last `days` days (excluding today):
    - Backfill any dates missing from the log entirely
    - Resolve any unresolved entries
    Saves log once if anything changed.
    """
    log = _load_log()
    changed = False
    for i in range(1, days + 1):
        target = _today_et() - timedelta(days=i)
        date_str = target.isoformat()
        if date_str not in log:
            log = _log_predictions_for_date(target, log)
            if date_str in log:
                changed = True
        else:
            if _resolve_unresolved_for_date(log, target):
                changed = True
    if changed:
        _save_log(log)
        _push_log_to_github()
        print(f"[app] _auto_heal_log: log updated and pushed to GitHub", flush=True)
    # Resolve picks for any recently-healed dates
    for i in range(1, days + 1):
        _resolve_picks_for_date(_today_et() - timedelta(days=i))


def log_todays_predictions():
    """Run today's predictions and save them to the log (skipped if already logged)."""
    log = _log_predictions_for_date(_today_et())
    _save_log(log)
    print(f"[app] Logged predictions for {_today_et().isoformat()}")


# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------
def _build_email_html(subscriber_email, top_picks, yesterday_entries, yesterday_str, today):
    """Build the HTML body for the daily email."""
    picks_data = _load_picks()
    user_picks_yesterday = picks_data.get(subscriber_email, {}).get(yesterday_str, [])

    # Today's top picks section
    if top_picks:
        picks_rows = "".join(
            f"<tr><td>{p['away_team_name']} @ {p['home_team_name']}</td>"
            f"<td><strong>{p['pick_team']}</strong></td>"
            f"<td>{round(p['confidence_pct'])}%</td>"
            f"<td>{p.get('game_time_et', '')}</td></tr>"
            for p in top_picks
        )
        today_section = f"""
        <h3 style="color:#1e2329;">⚾ Today's Top Picks — {today.strftime('%B %-d, %Y')}</h3>
        <table width="100%" cellpadding="6" style="border-collapse:collapse;font-size:14px;">
          <thead><tr style="background:#1e2329;color:#fff;">
            <th align="left">Matchup</th><th>Model Pick</th><th>Confidence</th><th>Time (ET)</th>
          </tr></thead>
          <tbody>{picks_rows}</tbody>
        </table>"""
    else:
        today_section = f"<p>No high-confidence games scheduled for {today.strftime('%B %-d')}.</p>"

    # Yesterday's recap section
    if yesterday_entries:
        correct = sum(1 for e in yesterday_entries if e.get("correct"))
        total   = len(yesterday_entries)
        acc_pct = round(100 * correct / total) if total else 0
        recap_rows = "".join(
            f"<tr><td>{e.get('away_team_name','?')} @ {e.get('home_team_name','?')}</td>"
            f"<td>{e.get('away_score','?')}–{e.get('home_score','?')}</td>"
            f"<td>{'✅' if e.get('correct') else ('🔘' if e.get('correct') is None else '❌')}</td></tr>"
            for e in yesterday_entries
        )
        yesterday_section = f"""
        <h3 style="color:#1e2329;margin-top:24px;">📊 Yesterday's Results — {yesterday_str}</h3>
        <p>Model: <strong>{correct}/{total} ({acc_pct}%)</strong></p>
        <table width="100%" cellpadding="6" style="border-collapse:collapse;font-size:13px;">
          <thead><tr style="background:#1e2329;color:#fff;">
            <th align="left">Game</th><th>Score</th><th>Result</th>
          </tr></thead>
          <tbody>{recap_rows}</tbody>
        </table>"""
    else:
        yesterday_section = ""

    # User picks section
    if user_picks_yesterday:
        u_correct = sum(1 for p in user_picks_yesterday if p.get("correct"))
        u_total   = len(user_picks_yesterday)
        u_pct     = round(100 * u_correct / u_total) if u_total else 0
        user_rows = "".join(
            f"<tr><td>{p.get('away_team_name','?')} @ {p.get('home_team_name','?')}</td>"
            f"<td>{p['pick']}</td>"
            f"<td>{'✅' if p.get('correct') else ('🔘' if p.get('correct') is None else '❌')}</td></tr>"
            for p in user_picks_yesterday
        )
        user_section = f"""
        <h3 style="color:#1e2329;margin-top:24px;">🎯 Your Picks Yesterday</h3>
        <p>You went <strong>{u_correct}/{u_total} ({u_pct}%)</strong></p>
        <table width="100%" cellpadding="6" style="border-collapse:collapse;font-size:13px;">
          <thead><tr style="background:#1e2329;color:#fff;">
            <th align="left">Game</th><th>Your Pick</th><th>Result</th>
          </tr></thead>
          <tbody>{user_rows}</tbody>
        </table>"""
    else:
        user_section = ""

    return f"""
    <html><body style="font-family:system-ui,sans-serif;max-width:600px;margin:0 auto;padding:16px;color:#333;">
      <h2 style="background:#1e2329;color:#fff;padding:12px 16px;border-radius:8px;margin:0 0 16px;">
        ⚾ DailyPredictionMLB
      </h2>
      {today_section}
      {yesterday_section}
      {user_section}
      <hr style="margin-top:24px;border:none;border-top:1px solid #e0e0e0;">
      <p style="font-size:12px;color:#999;margin-top:8px;">
        <a href="https://dailypredictionmlb.onrender.com">View full predictions</a> ·
        <a href="https://dailypredictionmlb.onrender.com/unsubscribe?email={subscriber_email}">Unsubscribe</a>
      </p>
    </body></html>"""


def send_daily_email():
    """Send the daily picks email to all subscribers."""
    if not RESEND_API_KEY:
        print("[email] RESEND_API_KEY not set — skipping email.")
        return
    subs = _load_subscribers()
    if not subs:
        return

    today     = _today_et()
    yesterday = (today - timedelta(days=1)).isoformat()
    log       = _load_log()
    yesterday_entries = [e for e in log.get(yesterday, []) if e.get("actual_winner") is not None]

    # Build top picks: run predictions, keep confidence >= 0.55, top 5
    team_baselines = _artifacts.get("team_baselines", {})
    sp_baselines   = _artifacts.get("sp_baselines", {})
    lr_model       = _artifacts.get("lr_model")
    scaler         = _artifacts.get("scaler")
    gb_model       = _artifacts.get("gb_model")
    xgb_model      = _artifacts.get("xgb_model")
    top_picks = []
    if lr_model:
        try:
            games_raw, _ = _get_schedule_cached(today)
            for game in games_raw:
                home, away = game.get("home_team"), game.get("away_team")
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
                result = predict_game(home_ts, away_ts, home_sp, away_sp, lr_model, scaler=scaler,
                                      gb_model=gb_model, xgb_model=xgb_model)
                if result["confidence"] >= 0.10:  # confidence=0.10 means 55% win prob
                    pick_team = game.get("home_team_name") if result["predicted_winner"] == "Home" else game.get("away_team_name")
                    top_picks.append({**game, "pick_team": pick_team,
                                      "confidence_pct": max(result["home_win_prob"], result["away_win_prob"]) * 100})
            top_picks = sorted(top_picks, key=lambda x: -x["confidence_pct"])[:5]
        except Exception as e:
            print(f"[email] Failed to build top picks: {e}")

    try:
        import resend
        resend.api_key = RESEND_API_KEY
        sent = 0
        for sub in subs:
            body = _build_email_html(sub["email"], top_picks, yesterday_entries, yesterday, today)
            unsubscribe_url = f"https://dailypredictionmlb.onrender.com/unsubscribe?email={sub['email']}"
            # Plain-text fallback — helps spam filters recognize legitimate newsletter
            plain = (
                f"DailyPredictionMLB — {today.strftime('%B %-d')} Picks\n\n"
                f"Top picks and yesterday's recap at:\n"
                f"https://dailypredictionmlb.onrender.com\n\n"
                f"Unsubscribe: {unsubscribe_url}"
            )
            try:
                resend.Emails.send({
                    "from":    FROM_EMAIL,
                    "to":      sub["email"],
                    "subject": f"DailyPredictionMLB — {today.strftime('%B %-d')} Picks",
                    "html":    body,
                    "text":    plain,
                    "headers": {
                        "List-Unsubscribe":      f"<{unsubscribe_url}>",
                        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                    },
                })
                sent += 1
            except Exception as e:
                print(f"[email] Failed to send to {sub['email']}: {e}")
        print(f"[email] Sent daily email to {sent}/{len(subs)} subscribers.")
    except ImportError:
        print("[email] resend package not installed.")


# ---------------------------------------------------------------------------
# GitHub log backup — commits predictions_log.json to git after each update
# ---------------------------------------------------------------------------
def _push_file_to_github(filepath, commit_message):
    """Push any local file to GitHub so it survives Render redeploys."""
    if not GITHUB_TOKEN:
        print("[github] GITHUB_TOKEN not set — skipping backup")
        return
    import base64
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{_github_path(filepath)}"
        r = requests.get(api_url, headers=headers, timeout=10)
        sha = r.json().get("sha") if r.status_code == 200 else None
        with open(filepath, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()
        payload = {"message": commit_message, "content": content_b64}
        if sha:
            payload["sha"] = sha
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            print(f"[github] {filepath} backed up to GitHub ({_github_path(filepath)})", flush=True)
        else:
            print(f"[github] backup failed: {resp.status_code} {resp.text[:200]}", flush=True)
    except Exception as e:
        print(f"[github] backup error: {e}")


def _push_log_to_github():
    _push_file_to_github(PREDICTIONS_LOG, f"Auto-backup predictions log {_today_et().isoformat()}")


def _restore_file_from_github(filepath):
    """On startup, pull the latest backed-up file from GitHub if the remote copy is
    larger than the local one. This recovers data that was pushed before a redeploy."""
    if not GITHUB_TOKEN:
        print(f"[github] GITHUB_TOKEN not set — cannot restore {filepath}")
        return
    import base64
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/contents/{_github_path(filepath)}",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            print(f"[github] restore {filepath}: GitHub returned {r.status_code} (path: {_github_path(filepath)})", flush=True)
            return
        remote_bytes = base64.b64decode(r.json()["content"])
        local_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        if len(remote_bytes) > local_size:
            with open(filepath, "wb") as f:
                f.write(remote_bytes)
            print(f"[github] Restored {filepath} from GitHub ({len(remote_bytes)} bytes)", flush=True)
        else:
            print(f"[github] {filepath} is up-to-date locally ({local_size} bytes)", flush=True)
    except Exception as e:
        print(f"[github] restore error for {filepath}: {e}")


def _refresh_today_odds():
    """Ensure today's predictions exist in the log with odds.
    Seeds today if missing, then patches any entries still lacking odds.
    Safe to call repeatedly — exits early if everything is already set."""
    if not ODDS_API_KEY:
        return
    today_str = _today_et().isoformat()
    log = _load_log()

    # Seed today if not in log yet (e.g. server was alive at midnight, missed startup)
    if today_str not in log:
        print(f"[app] _refresh_today_odds: today not in log — seeding now...", flush=True)
        log = _log_predictions_for_date(_today_et(), log)
        _save_log(log)
        _push_log_to_github()

    entries = [e for e in log.get(today_str, []) if e.get("game_type") != "S"]
    if not entries:
        return

    needs_odds = [e for e in entries if e.get("away_ml") is None]
    if not needs_odds:
        return

    _odds_cache.pop(today_str, None)  # force fresh fetch, bypass cache
    odds_map = _get_odds_cached()
    if not odds_map:
        print("[app] _refresh_today_odds: no odds returned from API", flush=True)
        return

    changed = 0
    for entry in log.get(today_str, []):
        if entry.get("game_type") == "S" or entry.get("away_ml") is not None:
            continue
        fields = _compute_odds_fields(entry["away_team"], entry["home_team"], entry, odds_map)
        if fields.get("away_ml") is not None:
            entry.update(fields)
            changed += 1

    if changed:
        _save_log(log)
        _upsert_betting_entries(log.get(today_str, []))
        _push_log_to_github()
        _push_betting_log_to_github()
        print(f"[app] _refresh_today_odds: patched odds into {changed} entries for {today_str}", flush=True)
    else:
        print(f"[app] _refresh_today_odds: {len(needs_odds)} entries need odds but none matched API — check team codes", flush=True)


# Restore persisted data files from GitHub on startup (recovers data after Render redeploy)
_restore_file_from_github(PICKS_LOG_PATH)
_restore_file_from_github(PREDICTIONS_LOG)
_restore_file_from_github(BETTING_LOG_PATH)
_restore_file_from_github(CLOSING_ODDS_LOG)
_restore_file_from_github(ARTIFACTS_PATH)

# Ensure artifacts are freshly loaded after GitHub restore (fixes accuracy discrepancy between Render/local)
try:
    load_artifacts()
    _team_count = len(_artifacts.get('team_baselines', {}))
    _sp_count = len(_artifacts.get('sp_baselines', {}))
    print(f"[startup] Artifacts reloaded after GitHub restore: {_team_count} teams, {_sp_count} pitchers", flush=True)
except Exception as _e:
    print(f"[startup] Failed to reload artifacts: {_e}", flush=True)

# Initialize betting_log table on startup (no-op if already exists)
try:
    import init_betting_log as _init_betting
    _init_betting.init_betting_log_table()
except Exception as _e:
    print(f"[startup] betting_log table init failed: {_e}", flush=True)

# Bootstrap 2026 game data into DB on startup if missing (DB is gitignored, so Render starts empty)
try:
    import sqlite3 as _sqlite3, fetch_2026_games as _f26
    _conn = _sqlite3.connect(_f26.DB_PATH)
    _cur  = _conn.cursor()
    _cur.execute("SELECT COUNT(*) FROM games WHERE season=2026")
    _count_2026 = _cur.fetchone()[0]
    _conn.close()
    if _count_2026 == 0:
        print("[startup] No 2026 games in DB — bootstrapping full season...", flush=True)
        _f26.fetch_and_insert(_f26.RS_START, _today_et() - timedelta(days=1), verbose=False)
        print("[startup] 2026 game bootstrap complete.", flush=True)
    else:
        # Catch up any missing days since last run
        _f26.fetch_and_insert(_f26.RS_START, _today_et() - timedelta(days=1), verbose=False)
except Exception as _e:
    print(f"[startup] 2026 game fetch failed: {_e}", flush=True)

# Seed today's predictions on startup (Render free tier may sleep through the 8 AM scheduler).
# This ensures x_scaled_features and odds data are always logged for today's games.
try:
    _startup_today = _today_et().isoformat()
    _startup_log   = _load_log()
    _today_entries = _startup_log.get(_startup_today, [])
    _rs_entries = [e for e in _today_entries if e.get("game_type") != "S"]
    _missing_features = _rs_entries and all(e.get("x_scaled_features") is None for e in _rs_entries)
    if _startup_today not in _startup_log or _missing_features:
        reason = "missing features" if _missing_features else "not in log"
        print(f"[startup] Today {reason} — re-seeding predictions now...", flush=True)
        _startup_log.pop(_startup_today, None)  # clear stale entry so it gets re-logged
        _startup_log = _log_predictions_for_date(_today_et(), _startup_log)
        _save_log(_startup_log)
        # Don't push here — _refresh_today_odds() will push after patching odds in
        print(f"[startup] Seeded {len(_startup_log.get(_startup_today, []))} predictions for {_startup_today}", flush=True)
    # Always patch odds — no-op if already set. Runs before betting_log rebuild.
    _refresh_today_odds()
    # Merge betting_log: start from the GitHub-restored version (has all historical
    # data), then supplement with anything newly available in predictions_log.
    # IMPORTANT: do NOT rebuild from scratch — that would erase historical entries
    # that only live in the betting_log backup (predictions_log only has odds for
    # the current day; past days' odds are not retroactively available).
    try:
        _blog = _load_betting_log()           # what was restored from GitHub
        _restored_total = sum(len(v) for v in _blog.values())
        _full_log = _load_log()
        for _d, _entries in _full_log.items():
            _odds_entries = [
                e for e in _entries
                if e.get("bet_rating") is not None and e.get("away_ml") is not None
            ]
            if _odds_entries:
                _blog[_d] = _odds_entries    # update this day
        _save_betting_log(_blog)
        _blog_total = sum(len(v) for v in _blog.values())
        print(f"[startup] betting_log merged: {_blog_total} entries across {len(_blog)} dates", flush=True)
        # Only push if we have at least as many entries as what was restored.
        # Prevents a failed restore (empty backup) from overwriting a larger
        # GitHub backup with a smaller merged result on the next deploy.
        if _blog_total >= _restored_total:
            _push_betting_log_to_github()
        else:
            print(f"[startup] Skipping push — merged ({_blog_total}) < restored ({_restored_total}); keeping larger GitHub backup", flush=True)
    except Exception as _be:
        print(f"[startup] betting_log merge failed: {_be}", flush=True)
except Exception as _e:
    print(f"[startup] Today-seeding failed: {_e}", flush=True)


# ---------------------------------------------------------------------------
# APScheduler — 8 AM daily refresh
# ---------------------------------------------------------------------------
def run_daily_update():
    """Wrapper called by APScheduler at 8 AM."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running scheduled daily update...")
    try:
        import fetch_2026_games
        fetch_2026_games.fetch_and_insert(
            fetch_2026_games.RS_START, _today_et() - timedelta(days=1), verbose=False
        )
        import update_daily
        update_daily.main()
        load_artifacts()
        _push_file_to_github(ARTIFACTS_PATH, f"Auto-backup model artifacts {_today_et().isoformat()}")
        update_yesterday_results()
        _auto_heal_log(days=7)
        log_todays_predictions()
        _refresh_today_odds()
        send_daily_email()
        _push_log_to_github()
        print("[app] Daily update complete.")
    except Exception as e:
        print(f"[app] Daily update failed: {e}")
        traceback.print_exc()


def resolve_todays_completed_games():
    """Persist results of any games that finished today. Called every 30 min."""
    try:
        today = _today_et()
        log = _load_log()
        if _resolve_unresolved_for_date(log, today):
            _save_log(log)
            _push_log_to_github()
            # Mirror results into betting_log so it stays current throughout the day
            from schedule_fetcher import get_game_results
            results = get_game_results(today)
            _resolve_betting_log_results(today.isoformat(), results)
            print(f"[app] Interval job: resolved completed games for {today.isoformat()}")
        _resolve_picks_for_date(today)
    except Exception as e:
        print(f"[app] resolve_todays_completed_games failed: {e}")


def _store_closing_odds():
    """Fetch odds near first pitch and store as closing line for CLV tracking."""
    today = _today_et().isoformat()
    try:
        # Bypass cache: clear the odds cache entry so we get a fresh fetch
        _odds_cache.pop(today, None)
        odds = get_mlb_odds(ODDS_API_KEY)

        # Store odds snapshot in archive for historical lookup
        _store_closing_odds_to_archive(today, odds)
        _push_closing_odds_to_github()

        log = _load_log()
        changed = False
        for entry in log.get(today, []):
            if entry.get("closing_away_ml") is not None:
                continue
            key = (entry.get("away_team"), entry.get("home_team"))
            game_odds = odds.get(key, {})
            if not game_odds:
                continue
            closing_away_impl = game_odds.get("away_implied")
            closing_home_impl = game_odds.get("home_implied")
            entry["closing_away_ml"]      = game_odds.get("away_ml")
            entry["closing_home_ml"]      = game_odds.get("home_ml")
            entry["closing_away_implied"] = closing_away_impl
            entry["closing_home_implied"] = closing_home_impl
            # CLV = model's implied prob - closing line's implied prob for predicted team
            predicted = entry.get("predicted_winner")
            model_prob = entry.get("home_win_prob") if predicted == "Home" else entry.get("away_win_prob")
            closing_impl = closing_home_impl if predicted == "Home" else closing_away_impl
            if model_prob is not None and closing_impl is not None:
                entry["clv"] = round(model_prob - closing_impl, 4)
            changed = True
        if changed:
            _save_log(log)
            _push_log_to_github()
            _upsert_betting_entries(log.get(today, []))
            print(f"[app] Closing odds stored for {today}", flush=True)
    except Exception as e:
        print(f"[app] _store_closing_odds failed: {e}", flush=True)


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_daily_update, "cron", hour=8, minute=0, timezone="America/New_York")
    scheduler.add_job(resolve_todays_completed_games, "interval", minutes=30)
    scheduler.add_job(_store_closing_odds, "cron", hour=18, minute=45, timezone="America/New_York")
    scheduler.add_job(_refresh_today_odds, "interval", hours=3)
    scheduler.start()
    print("[app] APScheduler started — daily update at 8:00 AM, results every 30 min, closing odds at 6:45 PM, odds refresh every 3h")
except ImportError:
    print("[app] apscheduler not installed — daily auto-refresh disabled")
    scheduler = None


# ---------------------------------------------------------------------------
# Feature contribution helpers
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "home_park_factor":              "Park Factor",
    "diff_pyth_win_pct":             "Pythagorean Win %",
    "diff_season_win_pct":           "Season Win %",
    "diff_roll30_obp":               "Team OBP (30g)",
    "diff_roll30_iso":               "Isolated Power / ISO (30g)",
    "diff_roll10_runs_scored":       "Runs/G (recent 10g)",
    "diff_roll10_homeruns":          "HR/G (recent 10g)",
    "diff_roll30_opp_whip":          "Opp WHIP Allowed (30g)",
    "diff_roll30_opp_hr_per9":       "Opp HR/9 Allowed (30g)",
    "diff_roll30_opp_strikeouts":    "Opp K Rate (30g)",
    "diff_roll10_win_pct":           "Win % (recent 10g)",
    "diff_bullpen_era":              "Bullpen ERA",
    "diff_roll7_bullpen_fatigue":    "Bullpen Fatigue (7d exp-decay)",
    "diff_rest_days":                "Rest Days",
    "home_sp_is_lhp":                "Home SP is LHP",
    "away_sp_is_lhp":                "Away SP is LHP",
    "diff_sp_era":                   "Starter ERA",
    "diff_sp_whip":                  "Starter WHIP",
    "diff_sp_xfip":                  "Starter xFIP",
    "diff_sp_ip_gs":                 "Starter IP/Start",
    "diff_sp_k_bb":                  "Starter K/BB",
    # Extra display-only labels (not in FEATURE_COLS — shown in modal for context)
    "diff_sp_siera":                 "Starter SIERA",
    "diff_sp_so9":                   "Starter K/9",
    "diff_sp_bb9":                   "Starter BB/9",
    "diff_sp_hr9":                   "Starter HR/9",
    "diff_roll30_hits":              "Hits/G (30g)",
    "diff_roll30_homeruns":          "HR/G (30g)",
}


def _compute_feature_contributions(home_ts, away_ts, home_sp, away_sp):
    """
    Compute per-feature contributions to the LR prediction.
    Returns list of {feature, label, raw_diff, contribution, favors}
    sorted by abs(contribution) descending.
    """
    import pandas as pd
    from MLBModel import FEATURE_COLS
    lr     = _artifacts.get("lr_model")
    scaler = _artifacts.get("scaler")
    if lr is None or scaler is None:
        return []

    features = {
        "home_park_factor":           home_ts.get("park_factor", 1.0),
        "diff_pyth_win_pct":          home_ts.get("pyth_win_pct", 0.5)          - away_ts.get("pyth_win_pct", 0.5),
        "diff_season_win_pct":        home_ts.get("win_pct", 0.5)               - away_ts.get("win_pct", 0.5),
        "diff_roll30_runs_scored":    home_ts.get("runs_per_game", 4.5)         - away_ts.get("runs_per_game", 4.5),
        "diff_roll30_runs_allowed":   home_ts.get("runs_allowed_per_game", 4.5) - away_ts.get("runs_allowed_per_game", 4.5),
        "diff_roll10_runs_scored":    home_ts.get("recent_runs_per_game", 4.5)  - away_ts.get("recent_runs_per_game", 4.5),
        "diff_roll30_obp":               home_ts.get("obp", 0.318)                  - away_ts.get("obp", 0.318),
        "diff_roll30_iso":               home_ts.get("iso", 0.150)                  - away_ts.get("iso", 0.150),
        "diff_roll10_win_pct":           home_ts.get("recent_win_pct", 0.5)         - away_ts.get("recent_win_pct", 0.5),
        "diff_roll10_homeruns":          home_ts.get("recent_hr_per_game", 1.1)     - away_ts.get("recent_hr_per_game", 1.1),
        "diff_roll30_opp_whip":          home_ts.get("opp_whip", 1.30)              - away_ts.get("opp_whip", 1.30),
        "diff_roll30_opp_hr_per9":       home_ts.get("opp_hr_per9", 1.10)           - away_ts.get("opp_hr_per9", 1.10),
        "diff_roll30_opp_strikeouts":    home_ts.get("opp_k_per_game", 8.5)         - away_ts.get("opp_k_per_game", 8.5),
        "diff_bullpen_era":              home_ts.get("bullpen_era", 4.20)            - away_ts.get("bullpen_era", 4.20),
        "diff_roll7_bullpen_fatigue":    home_ts.get("roll7_bullpen_fatigue", 8.0)  - away_ts.get("roll7_bullpen_fatigue", 8.0),
        # Display-only (not in FEATURE_COLS, shown in modal for context)
        "diff_roll30_hits":              home_ts.get("hits_per_game", 8.5)          - away_ts.get("hits_per_game", 8.5),
        "diff_roll30_homeruns":          home_ts.get("hr_per_game", 1.1)            - away_ts.get("hr_per_game", 1.1),
        "diff_rest_days":             home_ts.get("rest_days", 1)               - away_ts.get("rest_days", 1),
        "home_sp_is_lhp":             1 if home_sp.get("pitch_hand", "R") == "L" else 0,
        "away_sp_is_lhp":             1 if away_sp.get("pitch_hand", "R") == "L" else 0,
        "diff_sp_era":                home_sp.get("era", 4.0)    - away_sp.get("era", 4.0),
        "diff_sp_whip":               home_sp.get("whip", 1.3)  - away_sp.get("whip", 1.3),
        "diff_sp_ip_gs":              home_sp.get("ip_gs", 5.8) - away_sp.get("ip_gs", 5.8),
        "diff_sp_k_bb":               home_sp.get("k_bb", 2.5)  - away_sp.get("k_bb", 2.5),
        "diff_sp_xfip":               home_sp.get("xfip", 4.0)  - away_sp.get("xfip", 4.0),
        "diff_sp_siera":              home_sp.get("siera", 4.0) - away_sp.get("siera", 4.0),
        "diff_sp_so9":                home_sp.get("so9", 8.0)   - away_sp.get("so9", 8.0),
        "diff_sp_bb9":                home_sp.get("bb9", 3.0)   - away_sp.get("bb9", 3.0),
        "diff_sp_hr9":                home_sp.get("hr9", 1.2)   - away_sp.get("hr9", 1.2),
    }

    try:
        X_raw    = pd.DataFrame([features])[FEATURE_COLS]
        X_scaled = scaler.transform(X_raw)
        coefs    = lr.coef_[0]
        contribs = coefs * X_scaled[0]

        result = []
        for i, col in enumerate(FEATURE_COLS):
            c = float(contribs[i])
            if abs(c) < 0.001:
                continue
            result.append({
                "feature":      col,
                "label":        FEATURE_LABELS.get(col, col),
                "raw_diff":     round(float(X_raw.iloc[0, i]), 4),
                "contribution": round(c, 4),
                "favors":       "home" if c > 0 else "away",
            })
        return sorted(result, key=lambda x: -abs(x["contribution"]))[:8]
    except Exception as e:
        print(f"[app] feature contributions failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------
def _compute_odds_fields(away_retro, home_retro, pred_result, odds_map):
    """
    Look up odds for (away_retro, home_retro) and compute bet rating.
    Returns a dict with away_ml, home_ml, away_implied, home_implied, bet_rating.
    All values are None when no odds are available.
    """
    game_odds  = odds_map.get((away_retro, home_retro), {})
    away_ml    = game_odds.get("away_ml")
    home_ml    = game_odds.get("home_ml")
    away_impl  = game_odds.get("away_implied")
    home_impl  = game_odds.get("home_implied")

    bet_rating       = None
    model_edge       = None
    predicted_team_ml = None
    if away_impl is not None and home_impl is not None:
        predicted  = pred_result["predicted_winner"]
        model_prob  = pred_result["home_win_prob"] if predicted == "Home" else pred_result["away_win_prob"]
        market_prob = home_impl                    if predicted == "Home" else away_impl
        edge = model_prob - market_prob
        model_edge = round(edge, 4)
        predicted_team_ml = home_ml if predicted == "Home" else away_ml
        if edge > 0.05:
            bet_rating = "good"
        elif edge < -0.05:
            bet_rating = "bad"
        else:
            bet_rating = "unsure"

    return {
        "away_ml":           away_ml,
        "home_ml":           home_ml,
        "away_implied":      away_impl,
        "home_implied":      home_impl,
        "bet_rating":        bet_rating,
        "predicted_team_ml": predicted_team_ml,
        "model_edge":        model_edge,
        "odds_books":        game_odds.get("books", []),
        "arbitrage":         game_odds.get("arbitrage"),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"})


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictions")
def index():
    return render_template("index.html")


@app.route("/accuracy")
def accuracy_page():
    return render_template("accuracy.html")


@app.route("/betting")
def betting_page():
    return render_template("betting.html")


@app.route("/picks")
def picks_page():
    return render_template("picks.html")


# ---------------------------------------------------------------------------
# Subscriber routes
# ---------------------------------------------------------------------------
@app.route("/api/subscribe", methods=["POST"])
def subscribe():
    email = (request.json or {}).get("email", "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "Invalid email"}), 400
    subs = _load_subscribers()
    if any(s["email"] == email for s in subs):
        return jsonify({"error": "Already subscribed"}), 409
    subs.append({"email": email, "subscribed_at": datetime.now().isoformat()})
    _save_subscribers(subs)
    return jsonify({"status": "ok"})


@app.route("/unsubscribe")
def unsubscribe():
    email = request.args.get("email", "").strip().lower()
    if not email:
        return "Invalid unsubscribe link.", 400
    subs = _load_subscribers()
    subs = [s for s in subs if s["email"] != email]
    _save_subscribers(subs)
    return f"<p style='font-family:sans-serif;padding:2rem;'>✅ <strong>{email}</strong> has been unsubscribed from DailyPredictionMLB.</p>"


# ---------------------------------------------------------------------------
# Picks routes
# ---------------------------------------------------------------------------
@app.route("/api/picks/submit", methods=["POST"])
def picks_submit():
    data  = request.json or {}
    email = data.get("email", "").strip().lower()
    picks_list = data.get("picks", [])   # [{game_pk, pick, home_team, away_team, ...}]
    if not email or "@" not in email:
        return jsonify({"error": "Invalid email"}), 400
    if not picks_list:
        return jsonify({"error": "No picks provided"}), 400

    date_str = _today_et().isoformat()
    picks = _load_picks()
    if email not in picks:
        picks[email] = {}

    # Merge — don't overwrite picks already made for a game
    existing = {p["game_pk"]: p for p in picks[email].get(date_str, [])}
    for p in picks_list:
        pk = p.get("game_pk")
        if pk and pk not in existing:
            existing[pk] = {
                "game_pk":        pk,
                "pick":           p.get("pick"),
                "home_team":      p.get("home_team"),
                "away_team":      p.get("away_team"),
                "home_team_name": p.get("home_team_name"),
                "away_team_name": p.get("away_team_name"),
                "model_pick":     p.get("model_pick"),
                "correct":        None,
                "actual_winner":  None,
                "submitted_at":   datetime.now().isoformat(),
            }
    picks[email][date_str] = list(existing.values())
    _save_picks(picks)
    return jsonify({"status": "ok", "saved": len(existing)})


@app.route("/api/picks/mine")
def picks_mine():
    email    = request.args.get("email", "").strip().lower()
    date_str = request.args.get("date", _today_et().isoformat())
    if not email:
        return jsonify({"error": "email required"}), 400
    picks = _load_picks()
    return jsonify({"picks": picks.get(email, {}).get(date_str, [])})


@app.route("/api/picks/leaderboard")
def picks_leaderboard():
    picks    = _load_picks()
    model_log = _load_log()

    # Model stats
    model_correct = model_total = 0
    for day_entries in model_log.values():
        for e in day_entries:
            if e.get("correct") is not None:
                model_total += 1
                if e["correct"]:
                    model_correct += 1

    rows = []
    for email, days in picks.items():
        correct = total = 0
        for day_picks in days.values():
            for p in day_picks:
                if p.get("correct") is not None:
                    total += 1
                    if p["correct"]:
                        correct += 1
        if total == 0:
            continue
        # Mask email: j***@gmail.com
        parts = email.split("@")
        masked = parts[0][0] + "***@" + parts[1] if len(parts) == 2 else email
        rows.append({
            "email":    masked,
            "correct":  correct,
            "total":    total,
            "accuracy": round(correct / total, 3),
        })

    rows.sort(key=lambda x: (-x["accuracy"], -x["total"]))
    return jsonify({
        "leaderboard": rows,
        "model": {
            "correct":  model_correct,
            "total":    model_total,
            "accuracy": round(model_correct / model_total, 3) if model_total else 0,
        },
    })


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
        target_date = _today_et()

    team_baselines = _artifacts.get("team_baselines", {})
    sp_baselines   = _artifacts.get("sp_baselines", {})
    lr_model       = _artifacts.get("lr_model")
    scaler         = _artifacts.get("scaler")
    gb_model       = _artifacts.get("gb_model")
    xgb_model      = _artifacts.get("xgb_model")
    # Run line models (optional — None if not yet trained)
    _rl_lr  = _artifacts.get("lr_runline")
    _rl_gb  = _artifacts.get("gb_runline")
    _rl_sc  = _artifacts.get("scaler_runline")
    runline_models = (_rl_lr, _rl_gb, _rl_sc) if (_rl_lr and _rl_gb) else None

    if lr_model is None:
        return jsonify({"error": "Model not loaded. Run MLBModel.py first."}), 500

    # Load any stored results for this date
    log = _load_log()
    date_str = target_date.isoformat()
    log_by_pk = {entry["game_pk"]: entry for entry in log.get(date_str, [])}

    # One cached API call for both schedule and live scores
    games_raw, live_results = _get_schedule_cached(target_date)

    # Fetch live odds for today/future; use closing odds archive for past dates
    if target_date >= _today_et():
        odds_map = _get_odds_cached()
    else:
        # For past dates, try to get from closing odds archive
        closing_archive = _get_closing_odds_archive()
        date_odds = closing_archive.get(date_str, {})
        # Reconstruct odds dict from archive (keyed by team tuple)
        odds_map = {}
        for key_str, odds in date_odds.items():
            try:
                away_team, home_team = key_str.split("|")
                odds_map[(away_team, home_team)] = {
                    "away_ml": odds.get("away_ml"),
                    "home_ml": odds.get("home_ml"),
                    "away_implied": odds.get("away_implied"),
                    "home_implied": odds.get("home_implied"),
                }
            except Exception:
                pass

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

        home_sp_name = game.get("home_pitcher_name") or home_sp.get("name", "TBD")
        away_sp_name = game.get("away_pitcher_name") or away_sp.get("name", "TBD")

        try:
            result = predict_game(home_ts, away_ts, home_sp, away_sp, lr_model,
                                  scaler=scaler, runline_models=runline_models,
                                  gb_model=gb_model, xgb_model=xgb_model)
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
            _brier, _ll = (None, None)
            if actual_winner not in (None, "Tie") and result.get("home_win_prob") is not None:
                _brier, _ll = _compute_error_metrics(result["home_win_prob"], actual_winner)
            if pk in log_by_pk:
                log_by_pk[pk].update({
                    "away_score":    away_score,
                    "home_score":    home_score,
                    "actual_winner": actual_winner,
                    "correct":       correct,
                    "brier_score":   _brier,
                    "log_loss":      _ll,
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
            "home_sp_era":        round(home_sp.get("era",   4.20), 2),
            "home_sp_xfip":       round(home_sp.get("xfip",  4.20), 2),
            "home_sp_siera":      round(home_sp.get("siera", 4.20), 2),
            "home_sp_wins":       home_sp.get("wins",   0),
            "home_sp_losses":     home_sp.get("losses", 0),
            "home_sp_era_raw":    home_sp.get("era_raw"),
            "home_sp_whip_raw":   home_sp.get("whip_raw"),
            "home_sp_fip_raw":    home_sp.get("fip_raw"),
            "home_sp_gs":           home_sp.get("gs"),
            "home_sp_is_blended":    home_sp.get("is_blended", False),
            "home_sp_is_league_avg": home_sp.get("is_league_avg", False),
            "home_sp_is_prior_year": home_sp.get("is_prior_year", False),
            "home_sp_hand":          home_sp.get("pitch_hand", "R"),
            "away_sp_era":        round(away_sp.get("era",   4.20), 2),
            "away_sp_xfip":       round(away_sp.get("xfip",  4.20), 2),
            "away_sp_siera":      round(away_sp.get("siera", 4.20), 2),
            "away_sp_wins":       away_sp.get("wins",   0),
            "away_sp_losses":     away_sp.get("losses", 0),
            "away_sp_era_raw":    away_sp.get("era_raw"),
            "away_sp_whip_raw":   away_sp.get("whip_raw"),
            "away_sp_fip_raw":    away_sp.get("fip_raw"),
            "away_sp_gs":           away_sp.get("gs"),
            "away_sp_is_blended":    away_sp.get("is_blended", False),
            "away_sp_is_league_avg": away_sp.get("is_league_avg", False),
            "away_sp_is_prior_year": away_sp.get("is_prior_year", False),
            "away_sp_hand":          away_sp.get("pitch_hand", "R"),
            "home_obp":         round(home_ts.get("obp",                0.318), 3),
            "home_iso":         round(home_ts.get("iso",                0.150), 3),
            "home_opp_whip":    round(home_ts.get("opp_whip",           1.30),  2),
            "home_runs_pg":     round(home_ts.get("recent_runs_per_game", 4.5), 1),
            "home_bp_era":      round(home_ts.get("bullpen_era",        4.20),  2),
            "away_obp":         round(away_ts.get("obp",                0.318), 3),
            "away_iso":         round(away_ts.get("iso",                0.150), 3),
            "away_opp_whip":    round(away_ts.get("opp_whip",           1.30),  2),
            "away_runs_pg":     round(away_ts.get("recent_runs_per_game", 4.5), 1),
            "away_bp_era":      round(away_ts.get("bullpen_era",        4.20),  2),
            "home_sp_k_bb":     round(home_sp.get("k_bb",               2.5),   2),
            "away_sp_k_bb":     round(away_sp.get("k_bb",               2.5),   2),
            "home_cover_prob":  result.get("home_cover_prob"),
            "away_cover_prob":  result.get("away_cover_prob"),
            "predicted_total":  result.get("predicted_total"),
            "est_components":   result.get("est_components"),
            "actual_winner":         actual_winner,
            "away_score":            away_score,
            "home_score":            home_score,
            "correct":               correct,
            "feature_contributions": _compute_feature_contributions(home_ts, away_ts, home_sp, away_sp),
            **_compute_odds_fields(away, home, result, odds_map),
        })

    # Write odds fields back into log entries for today (they're computed live above
    # but never stored — this ensures betting_log stays populated going forward).
    if target_date >= _today_et() and odds_map:
        for entry in log.get(date_str, []):
            if entry.get("away_ml") is not None:
                continue
            pk = entry.get("game_pk")
            fields = _compute_odds_fields(entry["away_team"], entry["home_team"], entry, odds_map)
            if fields.get("away_ml") is not None:
                entry.update(fields)
                if pk:
                    log_by_pk[pk] = entry
                log_changed = True

    # Persist any live results or newly added odds back to the log
    if log_changed:
        log[date_str] = list(log_by_pk.values())
        _save_log(log)
        _push_log_to_github()
        _upsert_betting_entries(log.get(date_str, []))

    return jsonify({
        "date":         target_date.isoformat(),
        "games":        predictions_out,
        "last_updated": datetime.now().isoformat(),
    })


_accuracy_cache: dict = {"ts": 0.0, "payload": None}
_ACCURACY_CACHE_TTL = 60  # 1 minute — short TTL to catch stale baselines


@app.route("/api/accuracy")
def accuracy():
    """
    Returns per-team season accuracy from predictions_log.json.
    Splits regular season (game_type=R) and spring training (game_type=S).
    Auto-heals the log before computing: resolves unresolved entries and
    backfills missing days for the last 7 days.
    Result is cached for 5 minutes so rapid reloads always show the same number.
    """
    if time.time() - _accuracy_cache["ts"] < _ACCURACY_CACHE_TTL and _accuracy_cache["payload"]:
        return _accuracy_cache["payload"]

    from datetime import date as _date
    _season_start = _date(2026, 3, 27)
    _days_since_start = (_today_et() - _season_start).days

    import threading as _threading
    _existing = _load_log()
    _missing = sum(
        1 for i in range(1, _days_since_start + 1)
        if (_today_et() - timedelta(days=i)).isoformat() not in _existing
    )
    if _missing > 3:
        # Large backfill: run in background so the page doesn't time out
        _heal_target = _days_since_start
        _threading.Thread(
            target=_auto_heal_log, kwargs={"days": _heal_target}, daemon=True
        ).start()
        log = _existing
    else:
        # Small gap (≤3 days): heal synchronously so the accuracy % is stable
        # across reloads and doesn't flip between values
        _auto_heal_log(days=7)
        log = _load_log()

    def compute_stats(entries):
        team_stats = {}
        total_games = total_correct = 0
        for entry in entries:
            actual    = entry.get("actual_winner")
            predicted = entry.get("predicted_winner")
            correct   = entry.get("correct")

            for side in ("away", "home"):
                team_code = entry.get(f"{side}_team")
                team_name = entry.get(f"{side}_team_name") or RETRO_TO_FULL_NAME.get(team_code, team_code)
                if not team_code:
                    continue
                if team_code not in team_stats:
                    team_stats[team_code] = {
                        "team": team_code, "name": team_name,
                        "games": 0, "correct": 0,
                        "actual_wins": 0, "actual_losses": 0, "actual_ties": 0,
                        "win_pred_total": 0, "win_pred_correct": 0,
                        "loss_pred_total": 0, "loss_pred_correct": 0,
                    }
                s = team_stats[team_code]

                # Actual team W-L-T (counts all resolved games including ties)
                if actual == "Tie":
                    s["actual_ties"] += 1
                elif actual is not None:
                    team_side = "Home" if side == "home" else "Away"
                    if actual == team_side:
                        s["actual_wins"] += 1
                    else:
                        s["actual_losses"] += 1

                # Win/loss prediction accuracy (skip unresolved and ties)
                if correct is None:
                    continue
                s["games"] += 1
                if correct:
                    s["correct"] += 1
                team_side = "Home" if side == "home" else "Away"
                if predicted == team_side:
                    s["win_pred_total"] += 1
                    if correct:
                        s["win_pred_correct"] += 1
                else:
                    s["loss_pred_total"] += 1
                    if correct:
                        s["loss_pred_correct"] += 1

            if correct is None:
                continue
            total_games += 1
            if correct:
                total_correct += 1

        def _pct(num, den):
            return round(num / den, 3) if den else None

        teams_list = sorted(
            [
                {
                    **s,
                    "accuracy":       _pct(s["correct"], s["games"]) or 0,
                    "win_pred_pct":   _pct(s["win_pred_correct"],  s["win_pred_total"]),
                    "loss_pred_pct":  _pct(s["loss_pred_correct"], s["loss_pred_total"]),
                    "actual_record":  (
                        f"{s['actual_wins']}-{s['actual_losses']}"
                        + (f"-{s['actual_ties']}" if s["actual_ties"] else "")
                    ),
                }
                for s in team_stats.values()
            ],
            key=lambda x: (-x["accuracy"], -x["games"])
        )
        return {
            "overall": {
                "games":    total_games,
                "correct":  total_correct,
                "accuracy": _pct(total_correct, total_games) or 0,
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

    rs_stats = compute_stats(regular)
    st_stats = compute_stats(spring)

    # Overlay live MLB standings as authoritative actual_record for RS teams
    try:
        live_standings = get_team_standings()
        for team in rs_stats["teams"]:
            s = live_standings.get(team["team"])
            if s:
                team["actual_wins"]   = s["wins"]
                team["actual_losses"] = s["losses"]
                team["actual_record"] = f"{s['wins']}-{s['losses']}"
    except Exception as e:
        print(f"[app] get_team_standings failed: {e}")

    # Week-by-week accuracy breakdown (regular season only)
    from collections import defaultdict as _dd
    from datetime import date as _date
    season_start = _date(2026, 3, 27)
    weekly_acc = _dd(lambda: {"correct": 0, "games": 0})
    for date_str, day_entries in log.items():
        try:
            d = _date.fromisoformat(date_str)
        except ValueError:
            continue
        week = ((d - season_start).days // 7) + 1
        if week < 1:
            continue
        for e in day_entries:
            if e.get("game_type") == "S" or e.get("correct") is None:
                continue
            weekly_acc[week]["games"] += 1
            if e["correct"]:
                weekly_acc[week]["correct"] += 1

    by_week = [
        {
            "week":     w,
            "label":    f"Wk {w}",
            "games":    v["games"],
            "correct":  v["correct"],
            "accuracy": round(v["correct"] / v["games"], 4) if v["games"] else None,
        }
        for w, v in sorted(weekly_acc.items()) if v["games"] >= 3
    ]

    # O/U prediction accuracy (regular season only)
    ou_total = ou_correct_count = 0
    ou_errors = []
    for day_entries in log.values():
        for e in day_entries:
            if e.get("game_type") == "S" or e.get("ou_correct") is None:
                continue
            ou_total += 1
            if e["ou_correct"]:
                ou_correct_count += 1
            if e.get("actual_total") is not None and e.get("predicted_total") is not None:
                ou_errors.append(abs(e["actual_total"] - e["predicted_total"]))

    ou_stats = {
        "games":    ou_total,
        "correct":  ou_correct_count,
        "accuracy": round(ou_correct_count / ou_total, 3) if ou_total else None,
        "mae":      round(sum(ou_errors) / len(ou_errors), 2) if ou_errors else None,
    }

    # Home team win rate — actual baseline (how often home team wins regardless of model)
    _home_wins    = sum(1 for e in regular if e.get("correct") is not None and e.get("actual_winner") == "Home")
    _total_resolved = rs_stats["overall"]["games"]
    home_win_rate = round(_home_wins / _total_resolved, 3) if _total_resolved else None

    response = jsonify({
        "regular_season":  rs_stats,
        "spring_training": st_stats,
        "by_week":         by_week,
        "ou_stats":        ou_stats,
        "home_win_rate":   home_win_rate,
        "last_updated":    datetime.now().isoformat(),
    })
    _accuracy_cache["payload"] = response
    _accuracy_cache["ts"]      = time.time()
    return response


def _load_betting_log_from_db():
    """Load all betting_log entries from SQLite as a list of dicts."""
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db"))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM betting_log ORDER BY date DESC")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"[betting] Failed to load from DB: {e}", flush=True)
        # Fallback to JSON
        log = _load_betting_log()
        return [e for day in log.values() for e in day]


@app.route("/api/betting")
def betting_stats():
    """Return betting accuracy stats broken down by bet_rating category."""
    all_entries_raw = _load_betting_log_from_db()

    # Only include RS entries with odds data that are resolved
    all_entries = [
        e for e in all_entries_raw
        if e.get("game_type") != "S"
        and e.get("bet_rating") is not None
        and e.get("correct") is not None
    ]

    categories = {"good": [], "unsure": [], "bad": []}
    for e in all_entries:
        rating = e.get("bet_rating")
        if rating in categories:
            categories[rating].append(e)

    def _pl_for_bet(b):
        """Return net P/L for a $10 bet on the predicted team."""
        ml = b.get("predicted_team_ml")
        if ml is None:
            return None
        if b["correct"]:
            return round(10 * (ml / 100) if ml >= 0 else 10 * (100 / abs(ml)), 2)
        return -10.0

    def cat_stats(bets):
        if not bets:
            return {"games": 0, "correct": 0, "accuracy": None,
                    "net_pl": 0, "total_wagered": 0, "roi": None}
        correct = sum(1 for b in bets if b["correct"])
        pl_values = [_pl_for_bet(b) for b in bets if _pl_for_bet(b) is not None]
        net_pl = round(sum(pl_values), 2)
        total_wagered = 10 * len(pl_values)
        roi = round(net_pl / total_wagered, 4) if total_wagered else None
        wins  = sum(1 for b in bets if b["correct"])
        losses = len(bets) - wins
        return {
            "games":         len(bets),
            "correct":       correct,
            "losses":        losses,
            "accuracy":      round(correct / len(bets), 3),
            "net_pl":        net_pl,
            "total_wagered": total_wagered,
            "roi":           roi,
        }

    # Cumulative P/L series for chart (value bets only, sorted by date)
    value_bets = sorted(categories["good"], key=lambda e: (e["date"], e.get("game_pk", 0)))
    cumulative_pl = []
    running = 0.0
    for b in value_bets:
        pl = _pl_for_bet(b)
        if pl is None:
            continue
        running = round(running + pl, 2)
        cumulative_pl.append({
            "date":    b["date"],
            "pl":      running,
            "matchup": f"{b.get('away_team_name','')} @ {b.get('home_team_name','')}",
            "correct": b["correct"],
            "ml":      b.get("predicted_team_ml"),
            "edge":    b.get("model_edge"),
        })

    # Recent value bets (last 20, most recent first)
    recent_value_bets = []
    for b in reversed(value_bets[-20:]):
        pl = _pl_for_bet(b)
        pick_is_home = b["predicted_winner"] == "Home"
        ml   = b.get("predicted_team_ml")
        hwp  = b.get("home_win_prob")
        win_p = hwp if pick_is_home else (1 - hwp) if hwp is not None else None
        if ml is not None and win_p is not None:
            profit = 10 * (ml / 100) if ml >= 0 else 10 * (100 / abs(ml))
            ev = round(win_p * profit - (1 - win_p) * 10, 2)
        else:
            ev = None
        recent_value_bets.append({
            "game_pk":        b.get("game_pk"),
            "date":           b["date"],
            "away_team":      b.get("away_team_name", b.get("away_team")),
            "home_team":      b.get("home_team_name", b.get("home_team")),
            "pick":           b.get("home_team_name") if pick_is_home else b.get("away_team_name"),
            "ml":             ml,
            "edge":           b.get("model_edge"),
            "ev":             ev,
            "correct":        b["correct"],
            "pl":             pl,
            "home_win_prob":  hwp,
            "away_win_prob":  b.get("away_win_prob"),
            "predicted_winner": b.get("predicted_winner"),
            "away_score":     b.get("away_score"),
            "home_score":     b.get("home_score"),
        })

    # Tracking start date (first entry with odds data, RS only)
    odds_entries = [e for day in log.values() for e in day
                    if e.get("bet_rating") is not None and e.get("game_type") != "S"]
    tracking_start = min((e["date"] for e in odds_entries), default=None)

    # Team-by-team stats for value bets
    team_map = {}  # team_name → {bets, correct, pl_values}
    for b in value_bets:
        pick_is_home = b["predicted_winner"] == "Home"
        team = b.get("home_team_name") if pick_is_home else b.get("away_team_name")
        if not team:
            continue
        pl = _pl_for_bet(b)
        if pl is None:
            continue
        if team not in team_map:
            team_map[team] = {"bets": 0, "correct": 0, "net_pl": 0.0}
        team_map[team]["bets"]    += 1
        team_map[team]["correct"] += int(b["correct"])
        team_map[team]["net_pl"]  = round(team_map[team]["net_pl"] + pl, 2)

    team_stats = sorted([
        {
            "team":     team,
            "bets":     v["bets"],
            "correct":  v["correct"],
            "losses":   v["bets"] - v["correct"],
            "accuracy": round(v["correct"] / v["bets"], 3),
            "net_pl":   v["net_pl"],
        }
        for team, v in team_map.items()
    ], key=lambda x: x["net_pl"], reverse=True)

    # Closing Line Value stats (entries with clv logged)
    clv_entries = [e for day in log.values() for e in day
                   if e.get("clv") is not None and e.get("game_type") != "S"]
    clv_values = [e["clv"] for e in clv_entries]
    clv_stats = {
        "games":    len(clv_values),
        "positive": sum(1 for c in clv_values if c > 0),
        "avg_clv":  round(sum(clv_values) / len(clv_values), 4) if clv_values else None,
    }

    # Run-line accuracy (uses home_cover_prob from existing RL model)
    rl_entries = [e for e in all_entries if e.get("correct_rl") is not None]
    rl_correct  = sum(1 for e in rl_entries if e["correct_rl"])
    rl_stats = {
        "games":    len(rl_entries),
        "correct":  rl_correct,
        "losses":   len(rl_entries) - rl_correct,
        "accuracy": round(rl_correct / len(rl_entries), 3) if rl_entries else None,
    }

    return jsonify({
        "value_bets":      cat_stats(categories["good"]),
        "toss_ups":        cat_stats(categories["unsure"]),
        "no_value":        cat_stats(categories["bad"]),
        "cumulative_pl":   cumulative_pl,
        "recent_bets":     recent_value_bets,
        "team_stats":      team_stats,
        "tracking_start":  tracking_start,
        "clv_stats":       clv_stats,
        "rl_stats":        rl_stats,
        "last_updated":    datetime.now().isoformat(),
    })


@app.route("/api/calibration")
def calibration():
    """Bucket resolved RS predictions by model probability into 10% bins; return actual win rate per bin."""
    log = _load_log()
    entries = [
        e for day in log.values() for e in day
        if e.get("game_type") != "S"
        and e.get("correct") is not None
        and e.get("home_win_prob") is not None
    ]
    bins = [{"range": f"{i*10}–{i*10+10}%", "mid": (i * 10 + 5) / 100, "wins": 0, "total": 0}
            for i in range(10)]
    for e in entries:
        prob = e["home_win_prob"]
        if e.get("predicted_winner") == "Away":
            prob = 1 - prob
        idx = min(int(prob * 10), 9)
        bins[idx]["total"] += 1
        if e["correct"]:
            bins[idx]["wins"] += 1
    result = [
        {
            "label":     b["range"],
            "predicted": b["mid"],
            "actual":    round(b["wins"] / b["total"], 3) if b["total"] else None,
            "count":     b["total"],
        }
        for b in bins
    ]
    return jsonify(result)


@app.route("/api/debug/odds")
def debug_odds():
    """Diagnostic: shows odds API status, key presence, and live game count."""
    key_set = bool(ODDS_API_KEY)
    games_returned = 0
    error = None
    sample = []
    if key_set:
        try:
            odds = get_mlb_odds(ODDS_API_KEY)
            games_returned = len(odds)
            sample = [f"{a}@{h}: away={v['away_ml']}, home={v['home_ml']}"
                      for (a, h), v in list(odds.items())[:3]]
        except Exception as e:
            error = str(e)
    blog_entries = len(_load_betting_log_from_db())
    log = _load_log()
    entries_with_odds = sum(
        1 for day in log.values() for e in day
        if e.get("bet_rating") is not None
    )
    return jsonify({
        "odds_api_key_set":       key_set,
        "games_from_api_now":     games_returned,
        "api_error":              error,
        "sample_games":           sample,
        "betting_log_entries":    blog_entries,
        "predictions_with_odds":  entries_with_odds,
        "today_et":               _today_et().isoformat(),
    })


@app.route("/api/trigger-daily")
def trigger_daily():
    """
    External cron endpoint — wakes the server and runs the daily update + email.
    Requires ?key=TRIGGER_SECRET query param to prevent abuse.
    Set TRIGGER_SECRET env var on Render. Call from cron-job.org at 7:55 AM ET.
    """
    secret = request.args.get("key", "")
    if not TRIGGER_SECRET or secret != TRIGGER_SECRET:
        return jsonify({"error": "forbidden"}), 403
    import threading
    threading.Thread(target=run_daily_update, daemon=True).start()
    return jsonify({"status": "triggered", "time": datetime.now().isoformat()})


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


@app.route("/api/retrain-model")
def retrain_model():
    """
    Trigger weekly model retraining on 2021-2026 data.
    Requires ?key=TRIGGER_SECRET query param.
    Trains ensemble (LR + GB) on 2021-2025, validates on 2026.
    Takes ~5-10 minutes. Call from cron-job.org every Sunday 10 PM ET.
    """
    secret = request.args.get("key", "")
    if not TRIGGER_SECRET or secret != TRIGGER_SECRET:
        return jsonify({"error": "forbidden"}), 403

    import threading
    import update_daily

    def run_retrain():
        try:
            print(f"[retrain] Triggered at {datetime.now().isoformat()}")
            metrics = update_daily.retrain_model()
            if metrics:
                load_artifacts()
                _push_file_to_github(ARTIFACTS_PATH, f"Auto-backup retrained model {_today_et().isoformat()}")
                print(f"[retrain] Success: {metrics}")
            else:
                print(f"[retrain] Failed to retrain model")
        except Exception as e:
            print(f"[retrain] Exception: {e}")
            traceback.print_exc()

    threading.Thread(target=run_retrain, daemon=True).start()
    return jsonify({
        "status": "retrain_triggered",
        "message": "Model retraining started in background (5-10 minutes)",
        "time": datetime.now().isoformat()
    })


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
# Model Explainer (Feature 6)
# ---------------------------------------------------------------------------
@app.route("/explain")
def explain():
    return render_template("explain.html")


@app.route("/api/model/info")
def model_info():
    """Return LR coefficients, GBM importances, and training metadata for the explainer page."""
    from MLBModel import FEATURE_COLS as _FC
    arts       = _artifacts
    lr         = arts.get("lr_model")
    gb         = arts.get("gb_model")
    feat_cols  = arts.get("feature_cols", _FC)

    lr_coefs = {}
    if lr is not None:
        for feat, coef in zip(feat_cols, lr.coef_[0]):
            lr_coefs[feat] = round(float(coef), 4)

    gb_importances = {}
    if gb is not None:
        for feat, imp in zip(feat_cols, gb.feature_importances_):
            gb_importances[feat] = round(float(imp), 4)

    # Human-readable labels from FEATURE_LABELS (already defined in this file)
    labels = {k: v for k, v in FEATURE_LABELS.items() if k in feat_cols}

    return jsonify({
        "features":        feat_cols,
        "feature_labels":  labels,
        "lr_coefs":        lr_coefs,
        "gb_importances":  gb_importances,
        "training_info": {
            "games":         12233,
            "seasons":       "2021–2026",
            "models":        "Logistic Regression (C=0.5) + Gradient Boosting + XGBoost ensemble",
            "ensemble_rule": "Final probability = average of LR, GBM, and XGBoost predictions. "
                             "LR uses scaled features; tree models use raw differentials.",
            "rolling_stats": "Team baselines updated daily from 30-game rolling windows "
                             "(10-game for recent form) computed directly from the game database — "
                             "same pipeline as training, no distribution mismatch.",
            "hyperparams":   "GBM/XGBoost params tuned via 12-combo grid search on 2025 holdout: "
                             "best = n_estimators=200, max_depth=3, learning_rate=0.03.",
            "sp_era_boost":  "SP ERA coefficient boosted 1.4× post-hoc in LR (StandardScaler "
                             "absorbs training-time scaling; post-hoc is the only effective method).",
            "accuracy_note": "See /api/accuracy for live season accuracy",
        },
    })


# ---------------------------------------------------------------------------
# Custom Model Weights (Feature 7)
# ---------------------------------------------------------------------------
@app.route("/api/predict/custom", methods=["POST"])
def predict_custom():
    """
    Re-score today's predictions using user-specified LR feature weight multipliers.
    Body: {"multipliers": {"diff_sp_era": 1.5, "home_park_factor": 0.0, ...}}
    Returns: {"results": [{"game_pk": ..., "home_win_prob": ..., "away_win_prob": ...}, ...]}
    Only the LR portion is adjusted; GBM is not affected.
    """
    body = request.get_json(force=True, silent=True) or {}
    multipliers = body.get("multipliers", {})

    from MLBModel import FEATURE_COLS as _FC
    arts        = _artifacts
    lr          = arts.get("lr_model")
    feat_cols   = arts.get("feature_cols", _FC)

    if lr is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Ensure lr.coef_ is structured correctly. 
    # Binary LogisticRegression in sklearn has shape (1, n_features)
    raw_coef = lr.coef_[0] if lr.coef_.ndim > 1 else lr.coef_

    if len(raw_coef) != len(feat_cols):
        return jsonify({
            "error": f"Model feature count mismatch. Model expected {len(raw_coef)}, but got {len(feat_cols)} features."
        }), 500

    # Clamp multipliers to [0, 3] and map to the exact index order of feat_cols
    mults = {}
    coef_adjusted = np.zeros_like(raw_coef, dtype=float)
    
    for i, f in enumerate(feat_cols):
        # Default to 1.0 if feature not provided in the payload
        m_val = max(0.0, min(3.0, float(multipliers.get(f, 1.0))))
        mults[f] = m_val
        coef_adjusted[i] = raw_coef[i] * m_val

    today = _today_et().isoformat()
    log   = _load_log()
    results  = []
    skipped  = 0
    mismatched = 0

    intercept = float(lr.intercept_[0]) if hasattr(lr.intercept_, "__len__") else float(lr.intercept_)

    for entry in log.get(today, []):
        if entry.get("game_type") == "S":
            continue
        x_scaled = entry.get("x_scaled_features")
        if x_scaled is None:
            skipped += 1
            continue

        x = np.array(x_scaled).flatten()

        if len(x) != len(coef_adjusted):
            mismatched += 1
            continue

        log_odds = float(np.dot(coef_adjusted, x)) + intercept
        prob     = 1.0 / (1.0 + np.exp(-log_odds))
        results.append({
            "game_pk":       entry["game_pk"],
            "home_win_prob": round(prob, 3),
            "away_win_prob": round(1.0 - prob, 3),
        })

    if skipped:
        print(f"[custom weights] {skipped} entries skipped (missing x_scaled_features — re-seed today)", flush=True)
    if mismatched:
        print(f"[custom weights] {mismatched} entries skipped (feature count mismatch — model needs retrain)", flush=True)

    return jsonify({"results": results, "multipliers_applied": mults,
                    "skipped": skipped, "mismatched": mismatched})
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[app] Starting DailyPredictionMLB on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
