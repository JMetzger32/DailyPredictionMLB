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
import time
import traceback
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

def _today_et():
    """Return today's date in US Eastern time (avoids UTC midnight flip on Render)."""
    return datetime.now(tz=_ET).date()

from flask import Flask, jsonify, render_template, request

from MLBModel import predict_game, _default_sp_stats
from schedule_fetcher import get_todays_schedule, get_game_results, get_schedule_and_results, get_mlb_odds, get_team_standings, find_pitcher_by_name, RETRO_TO_FULL_NAME

app = Flask(__name__)

ARTIFACTS_PATH    = os.environ.get("ARTIFACTS_PATH", "mlb_model_artifacts.pkl")
PREDICTIONS_LOG   = os.environ.get("PREDICTIONS_LOG", "predictions_log.json")
SUBSCRIBERS_PATH  = os.environ.get("SUBSCRIBERS_PATH", "subscribers.json")
PICKS_LOG_PATH    = os.environ.get("PICKS_LOG_PATH", "picks_log.json")
RESEND_API_KEY    = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL        = os.environ.get("FROM_EMAIL", "onboarding@resend.dev")
TRIGGER_SECRET    = os.environ.get("TRIGGER_SECRET", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO       = os.environ.get("GITHUB_REPO", "JMetzger32/DailyPredictionMLB")

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
    odds = get_mlb_odds(ODDS_API_KEY)
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
        "bet_rating":        odds_data.get("bet_rating"),
        "predicted_team_ml": odds_data.get("predicted_team_ml"),
        "model_edge":        odds_data.get("model_edge"),
        "actual_winner":     None,
        "away_score":        None,
        "home_score":        None,
        "correct":           None,
        "predicted_total":   result.get("predicted_total"),
        "home_est_score":    result.get("home_est_score"),
        "away_est_score":    result.get("away_est_score"),
        "actual_total":      None,
        "ou_correct":        None,
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
            if not ou_done and entry.get("predicted_total") is not None:
                actual_total = r["home_score"] + r["away_score"]
                entry["actual_total"] = actual_total
                entry["ou_correct"] = abs(actual_total - entry["predicted_total"]) <= 2.0
            changed = True

    if changed:
        _save_log(log)
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
    if lr_model is None:
        return log

    games_raw = get_todays_schedule(target_date)
    # Fetch odds for today/future dates so we can store bet_rating in the log
    odds_map = _get_odds_cached() if target_date >= _today_et() else {}
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
        except Exception as e:
            print(f"[app] Could not resolve results for {date_str}: {e}")

    log[date_str] = entries
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
        print(f"[app] _auto_heal_log: log updated")
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
                result = predict_game(home_ts, away_ts, home_sp, away_sp, lr_model, scaler=scaler)
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
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filepath}"
        r = requests.get(api_url, headers=headers, timeout=10)
        sha = r.json().get("sha") if r.status_code == 200 else None
        with open(filepath, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode()
        payload = {"message": commit_message, "content": content_b64}
        if sha:
            payload["sha"] = sha
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            print(f"[github] {filepath} backed up to GitHub")
        else:
            print(f"[github] backup failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"[github] backup error: {e}")


def _push_log_to_github():
    _push_file_to_github(PREDICTIONS_LOG, f"Auto-backup predictions log {_today_et().isoformat()}")


def _restore_file_from_github(filepath):
    """On startup, pull the latest backed-up file from GitHub if the remote copy is
    larger than the local one. This recovers data that was pushed before a redeploy."""
    if not GITHUB_TOKEN:
        return
    import base64
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filepath}",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            return
        remote_bytes = base64.b64decode(r.json()["content"])
        local_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        if len(remote_bytes) > local_size:
            with open(filepath, "wb") as f:
                f.write(remote_bytes)
            print(f"[github] Restored {filepath} from GitHub ({len(remote_bytes)} bytes)")
        else:
            print(f"[github] {filepath} is up-to-date locally ({local_size} bytes)")
    except Exception as e:
        print(f"[github] restore error for {filepath}: {e}")


# Restore persisted data files from GitHub on startup (recovers data after Render redeploy)
_restore_file_from_github(PICKS_LOG_PATH)
_restore_file_from_github(PREDICTIONS_LOG)


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
        send_daily_email()
        _push_log_to_github()
        print("[app] Daily update complete.")
    except Exception as e:
        print(f"[app] Daily update failed: {e}")
        traceback.print_exc()


def resolve_todays_completed_games():
    """Persist results of any games that finished today. Called every 30 min."""
    try:
        log = _load_log()
        if _resolve_unresolved_for_date(log, _today_et()):
            _save_log(log)
            print(f"[app] Interval job: resolved completed games for {_today_et().isoformat()}")
            _push_log_to_github()
        _resolve_picks_for_date(_today_et())
    except Exception as e:
        print(f"[app] resolve_todays_completed_games failed: {e}")


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_daily_update, "cron", hour=8, minute=0, timezone="America/New_York")
    scheduler.add_job(resolve_todays_completed_games, "interval", minutes=30)
    scheduler.start()
    print("[app] APScheduler started — daily update at 8:00 AM, results every 30 min")
except ImportError:
    print("[app] apscheduler not installed — daily auto-refresh disabled")
    scheduler = None


# ---------------------------------------------------------------------------
# Feature contribution helpers
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "home_park_factor":           "Park Factor",
    "diff_pyth_win_pct":          "Pythagorean Win %",
    "diff_season_win_pct":        "Season Win %",
    "diff_roll30_hits":           "Hits/G (30g)",
    "diff_roll30_walks":          "Walks/G (30g)",
    "diff_roll30_obp":            "Team OBP (30g)",
    "diff_roll30_slg":            "Team SLG (30g)",
    "diff_roll30_iso":            "Isolated Power (30g)",
    "diff_roll10_runs_scored":    "Runs/G (recent 10g)",
    "diff_roll10_homeruns":       "HR/G (recent 10g)",
    "diff_roll30_opp_hits":       "Opp Hits/G (30g)",
    "diff_roll30_opp_walks":      "Opp Walks/G (30g)",
    "diff_roll30_opp_homeruns":   "Opp HR/G (30g)",
    "diff_roll30_opp_strikeouts": "Opp K Rate (30g)",
    "diff_roll30_errors":         "Errors/G (30g)",
    "diff_roll10_win_pct":        "Win % (recent 10g)",
    "diff_roll3_bullpen_used":    "Bullpen Usage (3g)",
    "diff_bullpen_era":           "Bullpen ERA",
    "diff_roll7_bullpen_ip":      "Bullpen IP (7g)",
    "diff_rest_days":             "Rest Days",
    "home_sp_is_lhp":             "Home SP is LHP",
    "away_sp_is_lhp":             "Away SP is LHP",
    "diff_sp_era":                "Starter ERA",
    "diff_sp_whip":               "Starter WHIP",
    "diff_sp_xfip":               "Starter xFIP",
    "diff_sp_ip_gs":              "Starter IP/Start",
    "diff_sp_k_bb":               "Starter K/BB",
    # Extra display-only labels (not active features)
    "diff_sp_siera":              "Starter SIERA",
    "diff_sp_so9":                "Starter K/9",
    "diff_sp_bb9":                "Starter BB/9",
    "diff_sp_hr9":                "Starter HR/9",
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
        "diff_roll30_obp":            home_ts.get("obp", 0.318)                 - away_ts.get("obp", 0.318),
        "diff_roll30_slg":            home_ts.get("slg", 0.400)                 - away_ts.get("slg", 0.400),
        "diff_roll30_iso":            home_ts.get("iso", 0.150)                 - away_ts.get("iso", 0.150),
        "diff_roll30_hits":           home_ts.get("hits_per_game", 8.5)         - away_ts.get("hits_per_game", 8.5),
        "diff_roll30_opp_hits":       home_ts.get("opp_hits_per_game", 8.5)     - away_ts.get("opp_hits_per_game", 8.5),
        "diff_roll30_walks":          home_ts.get("walks_per_game", 3.0)        - away_ts.get("walks_per_game", 3.0),
        "diff_roll30_opp_walks":      home_ts.get("opp_walks_per_game", 3.0)    - away_ts.get("opp_walks_per_game", 3.0),
        "diff_roll30_errors":         home_ts.get("errors_per_game", 0.7)       - away_ts.get("errors_per_game", 0.7),
        "diff_roll30_homeruns":       home_ts.get("hr_per_game", 1.1)           - away_ts.get("hr_per_game", 1.1),
        "diff_roll30_opp_homeruns":   home_ts.get("opp_hr_per_game", 1.1)       - away_ts.get("opp_hr_per_game", 1.1),
        "diff_roll10_win_pct":        home_ts.get("recent_win_pct", 0.5)        - away_ts.get("recent_win_pct", 0.5),
        "diff_roll10_homeruns":       home_ts.get("recent_hr_per_game", 1.1)    - away_ts.get("recent_hr_per_game", 1.1),
        "diff_roll30_opp_strikeouts": home_ts.get("opp_k_per_game", 8.5)        - away_ts.get("opp_k_per_game", 8.5),
        "diff_roll3_bullpen_used":    home_ts.get("bullpen_used", 3.0)          - away_ts.get("bullpen_used", 3.0),
        "diff_bullpen_era":           home_ts.get("bullpen_era", 4.20)          - away_ts.get("bullpen_era", 4.20),
        "diff_roll7_bullpen_ip":      home_ts.get("roll7_bullpen_ip", 15.0)    - away_ts.get("roll7_bullpen_ip", 15.0),
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

    # Fetch odds for today and future dates (The Odds API returns all upcoming games)
    odds_map = _get_odds_cached() if target_date >= _today_et() else {}

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
                                  scaler=scaler, runline_models=runline_models)
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
            "home_cover_prob":  result.get("home_cover_prob"),
            "away_cover_prob":  result.get("away_cover_prob"),
            "predicted_total":  result.get("predicted_total"),
            "home_est_score":   result.get("home_est_score"),
            "away_est_score":   result.get("away_est_score"),
            "actual_winner":         actual_winner,
            "away_score":            away_score,
            "home_score":            home_score,
            "correct":               correct,
            "feature_contributions": _compute_feature_contributions(home_ts, away_ts, home_sp, away_sp),
            **_compute_odds_fields(away, home, result, odds_map),
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
    Auto-heals the log before computing: resolves unresolved entries and
    backfills missing days for the last 7 days.
    """
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

    return jsonify({
        "regular_season":  rs_stats,
        "spring_training": st_stats,
        "by_week":         by_week,
        "ou_stats":        ou_stats,
        "last_updated":    datetime.now().isoformat(),
    })


@app.route("/api/betting")
def betting_stats():
    """Return betting accuracy stats broken down by bet_rating category."""
    log = _load_log()

    # Only include RS entries with odds data that are resolved
    all_entries = [
        e for day in log.values() for e in day
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

    return jsonify({
        "value_bets":      cat_stats(categories["good"]),
        "toss_ups":        cat_stats(categories["unsure"]),
        "no_value":        cat_stats(categories["bad"]),
        "cumulative_pl":   cumulative_pl,
        "recent_bets":     recent_value_bets,
        "team_stats":      team_stats,
        "tracking_start":  tracking_start,
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
