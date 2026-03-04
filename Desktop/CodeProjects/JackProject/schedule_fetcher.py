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

MLB_SCHEDULE_URL  = "https://statsapi.mlb.com/api/v1/schedule"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

# The Odds API full team name → Retrosheet code
ODDS_API_TEAM_TO_RETRO = {
    "Los Angeles Angels":    "ANA",
    "Arizona Diamondbacks":  "ARI",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHN",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KCA",
    "Los Angeles Dodgers":   "LAN",
    "Washington Nationals":  "WAS",
    "New York Mets":         "NYN",
    "Oakland Athletics":     "ATH",
    "Sacramento Athletics":  "ATH",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SDN",
    "Seattle Mariners":      "SEA",
    "San Francisco Giants":  "SFN",
    "St. Louis Cardinals":   "SLN",
    "Tampa Bay Rays":        "TBA",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Minnesota Twins":       "MIN",
    "Philadelphia Phillies": "PHI",
    "Atlanta Braves":        "ATL",
    "Chicago White Sox":     "CHA",
    "Miami Marlins":         "MIA",
    "New York Yankees":      "NYA",
    "Milwaukee Brewers":     "MIL",
}

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

            away_rec = away_info.get("leagueRecord", {})
            home_rec = home_info.get("leagueRecord", {})

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
                "away_wins":         away_rec.get("wins",   0),
                "away_losses":       away_rec.get("losses", 0),
                "home_wins":         home_rec.get("wins",   0),
                "home_losses":       home_rec.get("losses", 0),
            })

    # Sort by game time
    games.sort(key=lambda g: g.get("game_time_utc") or "")
    return games


def get_schedule_and_results(target_date=None):
    """
    Single API call that fetches schedule, probable pitchers, AND live/final scores.
    Uses hydrate=probablePitcher,linescore to get everything in one round-trip.

    Returns (games_list, results_dict) matching the formats of
    get_todays_schedule() and get_game_results() respectively.
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            MLB_SCHEDULE_URL,
            params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,linescore"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[schedule_fetcher] MLB API error: {e}")
        return [], {}

    games = []
    results = {}

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            away_info = game.get("teams", {}).get("away", {})
            home_info = game.get("teams", {}).get("home", {})

            away_id    = away_info.get("team", {}).get("id")
            home_id    = home_info.get("team", {}).get("id")
            away_retro = MLB_TEAM_ID_TO_RETRO.get(away_id)
            home_retro = MLB_TEAM_ID_TO_RETRO.get(home_id)
            away_full  = away_info.get("team", {}).get("name", RETRO_TO_FULL_NAME.get(away_retro, away_retro or ""))
            home_full  = home_info.get("team", {}).get("name", RETRO_TO_FULL_NAME.get(home_retro, home_retro or ""))

            away_pitcher = away_info.get("probablePitcher", {}).get("fullName")
            home_pitcher = home_info.get("probablePitcher", {}).get("fullName")

            game_time_utc = game.get("gameDate")
            game_time_et  = None
            if game_time_utc:
                try:
                    dt_utc = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
                    dt_et  = dt_utc.astimezone(ZoneInfo("America/New_York"))
                    game_time_et = dt_et.strftime("%-I:%M %p ET")
                except Exception:
                    game_time_et = game_time_utc

            status   = game.get("status", {}).get("detailedState", "Scheduled")
            away_rec = away_info.get("leagueRecord", {})
            home_rec = home_info.get("leagueRecord", {})
            pk       = game.get("gamePk")

            games.append({
                "game_pk":           pk,
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
                "away_wins":         away_rec.get("wins",   0),
                "away_losses":       away_rec.get("losses", 0),
                "home_wins":         home_rec.get("wins",   0),
                "home_losses":       home_rec.get("losses", 0),
            })

            final      = status in ("Final", "Game Over", "Completed Early")
            away_score = away_info.get("score")
            home_score = home_info.get("score")
            results[pk] = {"final": final, "away_score": away_score, "home_score": home_score}

    games.sort(key=lambda g: g.get("game_time_utc") or "")
    return games, results


def get_game_results(target_date):
    """
    Fetch final scores for completed games on target_date.

    Returns a dict keyed by game_pk:
      { game_pk: {"final": bool, "away_score": int|None, "home_score": int|None} }
    """
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            MLB_SCHEDULE_URL,
            params={"sportId": 1, "date": date_str, "hydrate": "linescore"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[schedule_fetcher] get_game_results error: {e}")
        return {}

    results = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            pk     = game.get("gamePk")
            status = game.get("status", {}).get("detailedState", "")
            final  = status in ("Final", "Game Over", "Completed Early")
            away_score = game.get("teams", {}).get("away", {}).get("score")
            home_score = game.get("teams", {}).get("home", {}).get("score")
            results[pk] = {"final": final, "away_score": away_score, "home_score": home_score}
    return results


def get_mlb_odds(api_key):
    """
    Fetch current MLB moneyline odds from The Odds API.
    Costs 1 credit per call. Returns {} on error or missing data.

    Returns dict keyed by (away_retro, home_retro):
      { ("NYA", "BOS"): {"away_ml": +125, "home_ml": -145,
                         "away_implied": 0.444, "home_implied": 0.556} }
    Implied probabilities are vig-adjusted (sum to 1.0).
    """
    if not api_key:
        return {}
    try:
        resp = requests.get(
            ODDS_API_BASE_URL,
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "h2h",
                "oddsFormat":  "american",
            },
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"[odds] API error: {e}")
        return {}

    def _american_to_raw(ml):
        """Convert American moneyline to raw (pre-vig) implied probability."""
        if ml < 0:
            return abs(ml) / (abs(ml) + 100)
        return 100 / (ml + 100)

    odds_map = {}
    for event in events:
        home_name  = event.get("home_team", "")
        away_name  = event.get("away_team", "")
        home_retro = ODDS_API_TEAM_TO_RETRO.get(home_name)
        away_retro = ODDS_API_TEAM_TO_RETRO.get(away_name)
        if not home_retro or not away_retro:
            continue

        # Average odds across all available bookmakers
        away_prices, home_prices = [], []
        for bm in event.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for outcome in mkt.get("outcomes", []):
                    n = outcome.get("name")
                    p = outcome.get("price")
                    if p is None:
                        continue
                    if n == away_name:
                        away_prices.append(p)
                    elif n == home_name:
                        home_prices.append(p)

        if not away_prices or not home_prices:
            continue

        away_ml = round(sum(away_prices) / len(away_prices))
        home_ml = round(sum(home_prices) / len(home_prices))

        away_raw = _american_to_raw(away_ml)
        home_raw = _american_to_raw(home_ml)
        total    = away_raw + home_raw  # >1 due to vig

        odds_map[(away_retro, home_retro)] = {
            "away_ml":      away_ml,
            "home_ml":      home_ml,
            "away_implied": round(away_raw / total, 4),
            "home_implied": round(home_raw / total, 4),
        }

    print(f"[odds] Fetched odds for {len(odds_map)} games")
    return odds_map
