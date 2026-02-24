"""
schedule_fetcher.py — Fetches today's MLB schedule + probable starting pitchers
from the free MLB Stats API (no API key required).

Usage:
    from schedule_fetcher import get_todays_schedule, find_pitcher_by_name
"""

import requests
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
import unicodedata
import re

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# MLB Stats API team ID -> Retrosheet team code
MLB_TEAM_ID_TO_RETRO = {
    108: "ANA",  # Angels
    109: "ARI",  # Diamondbacks
    110: "BAL",  # Orioles
    111: "BOS",  # Red Sox
    112: "CHN",  # Cubs
    113: "CIN",  # Reds
    114: "CLE",  # Guardians
    115: "COL",  # Rockies
    116: "DET",  # Tigers
    117: "HOU",  # Astros
    118: "KCA",  # Royals
    119: "LAN",  # Dodgers
    120: "WAS",  # Nationals
    121: "NYN",  # Mets
    133: "ATH",  # Athletics
    134: "PIT",  # Pirates
    135: "SDN",  # Padres
    136: "SEA",  # Mariners
    137: "SFN",  # Giants
    138: "SLN",  # Cardinals
    139: "TBA",  # Rays
    140: "TEX",  # Rangers
    141: "TOR",  # Blue Jays
    142: "MIN",  # Twins
    143: "PHI",  # Phillies
    144: "ATL",  # Braves
    145: "CHA",  # White Sox
    146: "MIA",  # Marlins
    147: "NYA",  # Yankees
    158: "MIL",  # Brewers
}

RETRO_TO_FULL_NAME = {
    "ANA": "Angels", "ARI": "Diamondbacks", "ATH": "Athletics", "ATL": "Braves",
    "BAL": "Orioles", "BOS": "Red Sox", "CHA": "White Sox", "CHN": "Cubs",
    "CIN": "Reds", "CLE": "Guardians", "COL": "Rockies", "DET": "Tigers",
    "HOU": "Astros", "KCA": "Royals", "LAN": "Dodgers", "MIA": "Marlins",
    "MIL": "Brewers", "MIN": "Twins", "NYA": "Yankees", "NYN": "Mets",
    "PHI": "Phillies", "PIT": "Pirates", "SDN": "Padres", "SEA": "Mariners",
    "SFN": "Giants", "SLN": "Cardinals", "TBA": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WAS": "Nationals",
}


def _normalize_name(name):
    """Normalize a pitcher name for fuzzy matching."""
    if not name:
        return ""
    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Lowercase, strip suffixes
    name = name.lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name).strip()
    name = re.sub(r"[^a-z\s]", "", name)
    return " ".join(name.split())


def find_pitcher_by_name(pitcher_name, sp_baselines):
    """
    Find a pitcher's ID in sp_baselines by name.
    Returns the pitcher ID string or None if not found.

    Strategy:
      1. Exact normalized full name match
      2. Last name match (if unambiguous — exactly one result)
    """
    if not pitcher_name:
        return None

    query_norm = _normalize_name(pitcher_name)
    if not query_norm:
        return None

    # Pass 1: exact full-name match
    for pid, info in sp_baselines.items():
        if _normalize_name(info.get("name", "")) == query_norm:
            return pid

    # Pass 2: last-name match (take last token of query)
    query_last = query_norm.split()[-1]
    matches = [pid for pid, info in sp_baselines.items()
               if query_last in _normalize_name(info.get("name", "")).split()]
    if len(matches) == 1:
        return matches[0]

    return None


def get_todays_schedule(target_date=None):
    """
    Fetch today's MLB schedule from the MLB Stats API.

    Returns a list of game dicts:
      {
        "game_pk":           int,
        "game_time_utc":     str (ISO-8601) or None,
        "game_time_et":      str (e.g. "7:05 PM") or None,
        "status":            str (e.g. "Scheduled", "Final"),
        "away_team":         str (Retrosheet code, e.g. "NYA") or None,
        "home_team":         str (Retrosheet code) or None,
        "away_team_name":    str (full name),
        "home_team_name":    str (full name),
        "away_pitcher_name": str or None,
        "home_pitcher_name": str or None,
      }
    """
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            MLB_SCHEDULE_URL,
            params={
                "sportId": 1,
                "date": date_str,
                "hydrate": "probablePitcher",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[schedule_fetcher] MLB API error: {e}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            away_info = game.get("teams", {}).get("away", {})
            home_info = game.get("teams", {}).get("home", {})

            away_id = away_info.get("team", {}).get("id")
            home_id = home_info.get("team", {}).get("id")

            away_retro = MLB_TEAM_ID_TO_RETRO.get(away_id)
            home_retro = MLB_TEAM_ID_TO_RETRO.get(home_id)

            away_full = away_info.get("team", {}).get("name", RETRO_TO_FULL_NAME.get(away_retro, away_retro or ""))
            home_full = home_info.get("team", {}).get("name", RETRO_TO_FULL_NAME.get(home_retro, home_retro or ""))

            away_pitcher = away_info.get("probablePitcher", {}).get("fullName")
            home_pitcher = home_info.get("probablePitcher", {}).get("fullName")

            # Parse game time (MLB API returns UTC ISO string)
            game_time_utc = game.get("gameDate")  # e.g. "2026-04-01T23:05:00Z"
            game_time_et = None
            if game_time_utc:
                try:
                    dt_utc = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
                    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
                    game_time_et = dt_et.strftime("%-I:%M %p ET")
                except Exception:
                    game_time_et = game_time_utc

            status = game.get("status", {}).get("detailedState", "Scheduled")

            games.append({
                "game_pk":           game.get("gamePk"),
                "game_time_utc":     game_time_utc,
                "game_time_et":      game_time_et,
                "status":            status,
                "game_type":         game.get("gameType", "R"),
                "away_team":         away_retro,
                "home_team":         home_retro,
                "away_team_name":    away_full,
                "home_team_name":    home_full,
                "away_pitcher_name": away_pitcher,
                "home_pitcher_name": home_pitcher,
            })

    # Sort by game time
    games.sort(key=lambda g: g.get("game_time_utc") or "")
    return games
