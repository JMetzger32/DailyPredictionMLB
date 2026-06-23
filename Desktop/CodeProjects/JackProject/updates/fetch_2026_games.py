"""
fetch_2026_games.py — Pull 2026 MLB regular season game results from the
MLB Stats API and insert them into mlb_allseasons.db in Retrosheet format.

Usage:
    python3 fetch_2026_games.py               # fetch all 2026 RS games through yesterday
    python3 fetch_2026_games.py --date 2026-05-20  # fetch a specific date
    python3 fetch_2026_games.py --full        # re-fetch entire 2026 season (skips existing)
"""

import argparse
import os
import sqlite3
import requests
import time
from datetime import date, timedelta

_UPDATES_DIR = os.path.dirname(os.path.abspath(__file__))  # updates/
DB_PATH = os.path.join(_UPDATES_DIR, "..", "Databases_and_logs", "mlb_allseasons.db")
SEASON       = 2026
RS_START     = date(2026, 3, 27)
STATS_BASE   = "https://statsapi.mlb.com/api/v1"

# MLB team ID → Retrosheet code
MLB_ID_TO_RETRO = {
    108: "ANA", 109: "ARI", 133: "ATH", 144: "ATL",
    110: "BAL", 111: "BOS", 145: "CHA", 112: "CHN",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KCA", 119: "LAN", 146: "MIA",
    158: "MIL", 142: "MIN", 147: "NYA", 121: "NYN",
    143: "PHI", 134: "PIT", 135: "SDN", 136: "SEA",
    137: "SFN", 138: "SLN", 139: "TBA", 140: "TEX",
    141: "TOR", 120: "WAS",
}


def _get_schedule(start: date, end: date) -> list:
    url = (f"{STATS_BASE}/schedule?sportId=1&gameType=R"
           f"&startDate={start}&endDate={end}&hydrate=teams")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    games = []
    for day in r.json().get("dates", []):
        for g in day.get("games", []):
            games.append(g)
    return games


def _get_boxscore(game_pk: int) -> dict | None:
    url = f"{STATS_BASE}/game/{game_pk}/boxscore"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _team_stats(box_team: dict) -> dict:
    ts = box_team.get("teamStats", {}).get("batting", {})
    pit = box_team.get("teamStats", {}).get("pitching", {})
    return {
        "at_bats":      ts.get("atBats", 0),
        "hits":         ts.get("hits", 0),
        "doubles":      ts.get("doubles", 0),
        "triples":      ts.get("triples", 0),
        "home_runs":    ts.get("homeRuns", 0),
        "rbi":          ts.get("rbi", 0),
        "walks":        ts.get("baseOnBalls", 0),
        "strikeouts":   ts.get("strikeOuts", 0),
        "earned_runs":  pit.get("earnedRuns", 0),
    }


def _extract_starting_pitcher_name(box_team: dict) -> str | None:
    """Return the starting pitcher's full name from a boxscore team dict."""
    pitchers = box_team.get("pitchers", [])
    if not pitchers:
        return None
    sp_id = pitchers[0]
    players = box_team.get("players", {})
    sp = players.get(f"ID{sp_id}", {})
    return sp.get("person", {}).get("fullName")


def insert_game(cur, game_id: int, game_pk: int, game_date: str,
                away_retro: str, home_retro: str,
                away_score: int, home_score: int,
                away_stats: dict, home_stats: dict,
                doubleheader: str = "0",
                away_sp_name: str | None = None,
                home_sp_name: str | None = None) -> bool:
    cur.execute("SELECT 1 FROM games WHERE game_id=?", (game_id,))
    if cur.fetchone():
        return False  # already exists

    cur.execute("""
        INSERT INTO games (
            game_id, season, date, doubleheader,
            visiting_team, home_team,
            visitor_score, home_score,
            visitor_at_bats, visitor_hits, visitor_doubles, visitor_triples,
            visitor_homeruns, visitor_rbi, visitor_walks, visitor_strikeouts,
            visitor_team_earned_runs,
            home_at_bats, home_hits, home_doubles, home_triples,
            home_homeruns, home_rbi, home_walks, home_strikeouts,
            home_team_earned_runs,
            visitor_starting_pitcher_name, home_starting_pitcher_name
        ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, (
        game_id, SEASON, game_date, doubleheader,
        away_retro, home_retro,
        away_score, home_score,
        away_stats["at_bats"], away_stats["hits"], away_stats["doubles"], away_stats["triples"],
        away_stats["home_runs"], away_stats["rbi"], away_stats["walks"], away_stats["strikeouts"],
        away_stats["earned_runs"],
        home_stats["at_bats"], home_stats["hits"], home_stats["doubles"], home_stats["triples"],
        home_stats["home_runs"], home_stats["rbi"], home_stats["walks"], home_stats["strikeouts"],
        home_stats["earned_runs"],
        away_sp_name, home_sp_name,
    ))
    return True


def backfill_pitcher_names(verbose: bool = True) -> int:
    """Update existing 2026 game rows that are missing starting pitcher names."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.execute("""
        SELECT game_id FROM games
        WHERE season=? AND home_starting_pitcher_name IS NULL
        ORDER BY date
    """, (SEASON,))
    game_pks = [r[0] for r in cur.fetchall()]
    conn.close()

    if verbose:
        print(f"Backfilling pitcher names for {len(game_pks)} games...")

    updated = 0
    for game_pk in game_pks:
        box = _get_boxscore(game_pk)
        if not box:
            continue
        time.sleep(0.15)

        teams = box.get("teams", {})
        home_sp = _extract_starting_pitcher_name(teams.get("home", {}))
        away_sp = _extract_starting_pitcher_name(teams.get("away", {}))

        if home_sp or away_sp:
            conn = sqlite3.connect(DB_PATH)
            cur  = conn.cursor()
            cur.execute("""
                UPDATE games
                SET home_starting_pitcher_name=?, visitor_starting_pitcher_name=?
                WHERE game_id=?
            """, (home_sp, away_sp, game_pk))
            conn.commit()
            conn.close()
            updated += 1
            if verbose and updated % 50 == 0:
                print(f"  {updated}/{len(game_pks)} updated...")

    if verbose:
        print(f"Done. {updated} games updated with pitcher names.")
    return updated


def fetch_and_insert(start: date, end: date, verbose: bool = True) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Get current max game_id for auto-increment
    cur.execute("SELECT MAX(game_id) FROM games WHERE season=?", (SEASON,))
    row = cur.fetchone()
    next_id = (row[0] + 1) if row[0] else 13047  # continues from 2025 max

    # Get already-stored game_pks for this season to avoid re-fetching boxscores
    cur.execute("SELECT game_id FROM games WHERE season=?", (SEASON,))
    # We store game_pk in game_id field for 2026 games
    existing_ids = {r[0] for r in cur.fetchall()}

    if verbose:
        print(f"Fetching schedule {start} → {end}...")
    games = _get_schedule(start, end)
    if verbose:
        print(f"  {len(games)} games on schedule")

    inserted = 0
    # Track doubleheaders: same date + same matchup
    dh_tracker: dict[tuple, int] = {}

    for g in sorted(games, key=lambda x: (x.get("gameDate", ""), x.get("gamePk", 0))):
        status = g.get("status", {}).get("abstractGameState", "")
        if status != "Final":
            continue

        game_pk   = g["gamePk"]
        game_date = g["gameDate"][:10].replace("-", "")  # YYYYMMDD
        away_id   = g["teams"]["away"]["team"]["id"]
        home_id   = g["teams"]["home"]["team"]["id"]
        away_retro = MLB_ID_TO_RETRO.get(away_id)
        home_retro = MLB_ID_TO_RETRO.get(home_id)
        if not away_retro or not home_retro:
            continue

        # Detect doubleheader
        dh_key = (game_date, away_retro, home_retro)
        dh_count = dh_tracker.get(dh_key, 0) + 1
        dh_tracker[dh_key] = dh_count
        doubleheader = str(dh_count - 1) if dh_count == 1 else str(dh_count)

        # Use game_pk as game_id for 2026 (unique, stable)
        if game_pk in existing_ids:
            continue

        box = _get_boxscore(game_pk)
        if not box:
            continue
        time.sleep(0.1)  # be nice to the API

        teams = box.get("teams", {})
        away_box = teams.get("away", {})
        home_box = teams.get("home", {})
        away_score = away_box.get("teamStats", {}).get("batting", {}).get("runs", 0)
        home_score = home_box.get("teamStats", {}).get("batting", {}).get("runs", 0)

        away_stats = _team_stats(away_box)
        home_stats = _team_stats(home_box)
        away_sp_name = _extract_starting_pitcher_name(away_box)
        home_sp_name = _extract_starting_pitcher_name(home_box)

        ok = insert_game(cur, game_pk, game_pk, game_date,
                         away_retro, home_retro,
                         away_score, home_score,
                         away_stats, home_stats, doubleheader,
                         away_sp_name=away_sp_name, home_sp_name=home_sp_name)
        if ok:
            inserted += 1
            if verbose:
                print(f"  + {game_date} {away_retro}@{home_retro} "
                      f"{away_score}-{home_score}")

    conn.commit()
    conn.close()
    if verbose:
        print(f"Done. {inserted} new games inserted.")
    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",  help="Fetch a single date YYYY-MM-DD")
    parser.add_argument("--full",  action="store_true", help="Fetch full 2026 season")
    parser.add_argument("--backfill-pitchers", action="store_true",
                        help="Backfill missing starting pitcher names for existing 2026 games")
    args = parser.parse_args()

    if args.backfill_pitchers:
        backfill_pitcher_names()
        return

    yesterday = date.today() - timedelta(days=1)

    if args.date:
        d = date.fromisoformat(args.date)
        fetch_and_insert(d, d)
    elif args.full:
        fetch_and_insert(RS_START, yesterday)
    else:
        # Default: fetch from last known date to yesterday
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT MAX(date) FROM games WHERE season=?", (SEASON,))
        row = cur.fetchone()
        conn.close()
        if row[0]:
            last_date = date(int(row[0][:4]), int(row[0][4:6]), int(row[0][6:8]))
            start = last_date + timedelta(days=1)
        else:
            start = RS_START
        if start > yesterday:
            print("DB is up to date.")
            return
        fetch_and_insert(start, yesterday)


if __name__ == "__main__":
    main()
