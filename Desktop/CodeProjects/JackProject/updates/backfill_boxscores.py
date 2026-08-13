"""
backfill_boxscores.py — Fetch MLB Stats API boxscores and populate
game_pitcher_lines / game_batter_lines, keyed by the existing Retrosheet
game_id (gamePk is stored alongside for provenance, not as the join key).

Resumability is DB-state-driven (no checkpoint file): games already present
in game_pitcher_lines are skipped, so a crash or Ctrl-C just picks back up.

Usage:
    python3 backfill_boxscores.py               # full backfill, newest-first
    python3 backfill_boxscores.py --limit 25     # bounded batch (daily job uses this)
    python3 backfill_boxscores.py --season 2026  # restrict to one season
"""

import argparse
import os
import sqlite3
import sys
import time

import requests

_UPDATES_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UPDATES_DIR)
sys.path.insert(0, _UPDATES_DIR)

from init_boxscore_tables import init_boxscore_tables
from update_daily import RETRO_TO_MLB_ID

DB_PATH = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
STATS_BASE = "https://statsapi.mlb.com/api/v1"
THROTTLE_SECONDS = 0.12

# The Athletics' Retrosheet code changed (OAK -> ATH) with their 2025
# relocation; games' team codes still store "OAK" for 2020-2024 rows (see
# chore/fix-team-code-mapping), but the MLB Stats API always reports the
# current team id (133) regardless of season. Local alias only — not a
# change to the shared RETRO_TO_MLB_ID map, whose real callers never look
# up "OAK".
_RETRO_TO_MLB_ID = dict(RETRO_TO_MLB_ID)
_RETRO_TO_MLB_ID["OAK"] = _RETRO_TO_MLB_ID["ATH"]
_NORMALIZE_TEAM = {"OAK": "ATH"}


def _parse_innings_pitched(ip_str):
    """MLB's inningsPitched is a string where the decimal digit is THIRDS of
    an inning, not a decimal fraction — '6.1' = 6 + 1/3 = 6.333, not 6.1."""
    if ip_str is None:
        return None
    try:
        whole, _, frac = str(ip_str).partition(".")
        whole = int(whole)
        frac = int(frac) if frac else 0
        return whole + frac / 3.0
    except (ValueError, TypeError):
        return None


def resolve_game_pk(date_iso, home_retro, away_retro, doubleheader_flag, schedule_cache):
    """Resolve a Retrosheet game to its MLB gamePk via the schedule endpoint,
    matched by (home team id, away team id) and tie-broken on gameNumber for
    doubleheaders. schedule_cache is a dict the caller reuses across calls so
    a whole day's games only cost one schedule request."""
    if date_iso not in schedule_cache:
        try:
            resp = requests.get(f"{STATS_BASE}/schedule",
                                 params={"sportId": 1, "startDate": date_iso, "endDate": date_iso},
                                 timeout=15)
            resp.raise_for_status()
            data = resp.json()
            games = [g for d in data.get("dates", []) for g in d.get("games", [])]
            schedule_cache[date_iso] = games
        except Exception:
            schedule_cache[date_iso] = None
        time.sleep(THROTTLE_SECONDS)
    games = schedule_cache[date_iso]
    if games is None:
        return None, "schedule fetch failed"
    home_id = _RETRO_TO_MLB_ID.get(home_retro)
    away_id = _RETRO_TO_MLB_ID.get(away_retro)
    candidates = [g for g in games
                  if g["teams"]["home"]["team"]["id"] == home_id and g["teams"]["away"]["team"]["id"] == away_id]
    if not candidates:
        return None, "no schedule match"
    if len(candidates) == 1:
        return candidates[0]["gamePk"], "unique match"
    game_num = int(doubleheader_flag) if str(doubleheader_flag) in ("1", "2") else 1
    for g in candidates:
        if g.get("gameNumber") == game_num:
            return g["gamePk"], "doubleheader tie-break by gameNumber"
    return candidates[0]["gamePk"], "doubleheader tie-break fallback (first candidate)"


def game_pk_for_row(game_id, season, date_iso, home_retro, away_retro, doubleheader, schedule_cache):
    """2026 game_id values ARE the MLB gamePk (see fetch_2026_games.py) — no
    network needed. Earlier seasons use sequential Retrosheet-style ids and
    need schedule-endpoint resolution."""
    if season == 2026:
        try:
            return int(game_id), "game_id is gamePk (2026)"
        except (TypeError, ValueError):
            pass
    return resolve_game_pk(date_iso, home_retro, away_retro, doubleheader, schedule_cache)


def fetch_boxscore(game_pk):
    try:
        resp = requests.get(f"{STATS_BASE}/game/{game_pk}/boxscore", timeout=20)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def extract_lines(box, team_code):
    """Return (pitcher_lines, batter_lines) for one side of a boxscore."""
    pitcher_lines, batter_lines = [], []
    for pid, p in box.get("players", {}).items():
        pstats = p.get("stats", {})
        person = p.get("person", {})
        name = person.get("fullName")
        retro_id = None  # boxscores are name-matched only; MLB ids aren't Retrosheet ids

        pit = pstats.get("pitching")
        if pit and pit.get("inningsPitched") is not None:
            gs_raw = pit.get("gamesStarted")
            pitcher_lines.append({
                "team": team_code, "retro_pitcher_id": retro_id, "pitcher_name": name,
                "gs": int(gs_raw) if gs_raw is not None else 0,
                "ip": _parse_innings_pitched(pit.get("inningsPitched")),
                "pitches": pit.get("numberOfPitches"),
                "er": pit.get("earnedRuns"),
                "batters_faced": pit.get("battersFaced"),
            })

        bat = pstats.get("batting")
        if bat and bat.get("plateAppearances"):
            batter_lines.append({
                "team": team_code, "retro_batter_id": retro_id, "batter_name": name,
                "pa": bat.get("plateAppearances"), "ab": bat.get("atBats"), "h": bat.get("hits"),
                "doubles": bat.get("doubles"), "triples": bat.get("triples"), "hr": bat.get("homeRuns"),
                "bb": bat.get("baseOnBalls"), "hbp": bat.get("hitByPitch"), "sf": bat.get("sacFlies"),
            })
    return pitcher_lines, batter_lines


def insert_lines(cur, game_id, game_pk, pitcher_lines, batter_lines):
    for pl in pitcher_lines:
        if not pl["pitcher_name"]:
            continue
        cur.execute("""
            INSERT OR IGNORE INTO game_pitcher_lines
                (game_id, retro_pitcher_id, pitcher_name, team, game_pk, gs, ip, pitches, er, batters_faced)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (str(game_id), pl["retro_pitcher_id"], pl["pitcher_name"], pl["team"], game_pk,
              pl["gs"], pl["ip"], pl["pitches"], pl["er"], pl["batters_faced"]))
    for bl in batter_lines:
        if not bl["batter_name"]:
            continue
        cur.execute("""
            INSERT OR IGNORE INTO game_batter_lines
                (game_id, retro_batter_id, batter_name, team, game_pk, pa, ab, h, doubles, triples, hr, bb, hbp, sf)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (str(game_id), bl["retro_batter_id"], bl["batter_name"], bl["team"], game_pk,
              bl["pa"], bl["ab"], bl["h"], bl["doubles"], bl["triples"], bl["hr"], bl["bb"], bl["hbp"], bl["sf"]))


def run_batch(limit=None, season=None, verbose=True):
    """Fetch boxscores for games missing a game_pitcher_lines match,
    newest-first. limit=None means no cap (full backfill); the daily
    incremental job always passes a limit so one tick stays well inside
    Render's request timeout."""
    init_boxscore_tables()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    query = """
        SELECT g.game_id, g.season, g.date, g.doubleheader, g.home_team, g.visiting_team
        FROM games g
        WHERE NOT EXISTS (SELECT 1 FROM game_pitcher_lines gpl WHERE gpl.game_id = CAST(g.game_id AS TEXT))
    """
    params = []
    if season is not None:
        query += " AND g.season = ?"
        params.append(season)
    query += " ORDER BY g.date DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    if verbose:
        print(f"[backfill_boxscores] {len(rows)} games to fetch")

    schedule_cache = {}
    n_resolved = n_fetched = n_unresolved = n_fetch_failed = 0
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for game_id, season_val, date_compact, doubleheader, home_team, visiting_team in rows:
        date_iso = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
        home_norm = _NORMALIZE_TEAM.get(home_team, home_team)
        away_norm = _NORMALIZE_TEAM.get(visiting_team, visiting_team)

        game_pk, note = game_pk_for_row(game_id, season_val, date_iso, home_team, visiting_team,
                                         doubleheader, schedule_cache)
        if game_pk is None:
            n_unresolved += 1
            if verbose:
                print(f"  SKIP {game_id} ({date_iso} {away_norm}@{home_norm}): {note}")
            continue
        n_resolved += 1

        box = fetch_boxscore(game_pk)
        if box is None:
            n_fetch_failed += 1
            continue
        time.sleep(THROTTLE_SECONDS)

        teams = box.get("teams", {})
        home_pitchers, home_batters = extract_lines(teams.get("home", {}), home_norm)
        away_pitchers, away_batters = extract_lines(teams.get("away", {}), away_norm)
        insert_lines(cur, game_id, game_pk, home_pitchers + away_pitchers, home_batters + away_batters)
        conn.commit()
        n_fetched += 1
        if verbose and n_fetched % 25 == 0:
            print(f"  {n_fetched}/{len(rows)} fetched...")

    conn.close()
    if verbose:
        print(f"[backfill_boxscores] done. fetched={n_fetched} unresolved={n_unresolved} "
              f"fetch_failed={n_fetch_failed} (of {len(rows)} candidates)")
    return {"n_candidates": len(rows), "n_fetched": n_fetched,
            "n_unresolved": n_unresolved, "n_fetch_failed": n_fetch_failed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max games to fetch this run")
    parser.add_argument("--season", type=int, default=None, help="Restrict to one season")
    args = parser.parse_args()
    run_batch(limit=args.limit, season=args.season)


if __name__ == "__main__":
    main()
