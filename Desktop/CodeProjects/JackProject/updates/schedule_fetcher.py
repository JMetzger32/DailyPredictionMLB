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
ODDS_API_HISTORICAL_URL = ("https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds/")

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
    # --- Aliases for names The Odds API actually returns, current and historical. ---
    # "Athletics" (no city) is the club's CURRENT name and is what MLB StatsAPI and
    # The Odds API both return; its absence meant every A's game resolved to None and
    # was silently dropped (126 game-appearances in predictions_log, 0 with odds,
    # vs 17-22% for every other team). "Cleveland Indians" is the pre-2022 name and
    # is required for any 2021 historical backfill.
    "Athletics":             "ATH",
    "Cleveland Indians":     "CLE",
}

# Nickname -> retro, derived from the full-name map by stripping the city prefix, plus
# retired nicknames. Used as a FALLBACK when an exact full-name match fails, so a future
# relocation/rename (the Athletics moved twice in two years) degrades to a warning
# instead of silently dropping every one of that team's games.
_ODDS_NICKNAME_TO_RETRO = {
    "angels": "ANA", "diamondbacks": "ARI", "orioles": "BAL", "red sox": "BOS",
    "cubs": "CHN", "reds": "CIN", "guardians": "CLE", "indians": "CLE",
    "rockies": "COL", "tigers": "DET", "astros": "HOU", "royals": "KCA",
    "dodgers": "LAN", "nationals": "WAS", "mets": "NYN", "athletics": "ATH",
    "pirates": "PIT", "padres": "SDN", "mariners": "SEA", "giants": "SFN",
    "cardinals": "SLN", "rays": "TBA", "rangers": "TEX", "blue jays": "TOR",
    "twins": "MIN", "phillies": "PHI", "braves": "ATL", "white sox": "CHA",
    "marlins": "MIA", "yankees": "NYA", "brewers": "MIL",
}

# Raw team strings we could not resolve, counted per process. Printed by callers and
# checked by the historical backfill's canary gate -- an unresolved name means paid-for
# games get discarded, so this must never be silently ignored again.
UNMAPPED_ODDS_TEAMS = {}


def resolve_odds_team(name):
    """The Odds API team name -> Retrosheet code, or None (and recorded).

    Exact full-name match first, then longest-matching nickname suffix. Records any
    failure in UNMAPPED_ODDS_TEAMS rather than dropping it on the floor."""
    if not name:
        return None
    retro = ODDS_API_TEAM_TO_RETRO.get(name)
    if retro:
        return retro
    low = name.lower().strip()
    best = None
    for nick, code in _ODDS_NICKNAME_TO_RETRO.items():
        if low.endswith(nick) and (best is None or len(nick) > len(best[0])):
            best = (nick, code)
    if best:
        return best[1]
    UNMAPPED_ODDS_TEAMS[name] = UNMAPPED_ODDS_TEAMS.get(name, 0) + 1
    return None


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

    # Pass 1: exact full-name match. If more than one entry shares the same
    # normalized name (a stale prior-season duplicate lingering under a different
    # key alongside a fresh current-season entry — see merge_sp_baselines_dedup),
    # prefer the freshest: current-season starts (higher gs) and not flagged
    # prior-year, so a leftover duplicate can never shadow live data.
    exact = [(pid, info) for pid, info in sp_baselines.items()
             if _normalize_name(info.get("name", "")) == query_norm]
    if exact:
        def _freshness(item):
            info = item[1]
            return (0 if info.get("is_prior_year") else 1, info.get("gs") or 0)
        return max(exact, key=_freshness)[0]

    # Pass 2: last-name match with first-initial verification
    query_tokens = query_norm.split()
    query_last = query_tokens[-1]
    query_first_initial = query_tokens[0][0] if len(query_tokens) > 1 else None

    matches = [pid for pid, info in sp_baselines.items()
               if query_last in _normalize_name(info.get("name", "")).split()]

    if len(matches) == 1:
        # Verify first initial to avoid false positives (e.g. "Jose" != "Ranger")
        if query_first_initial:
            matched_tokens = _normalize_name(sp_baselines[matches[0]].get("name", "")).split()
            if matched_tokens and not matched_tokens[0].startswith(query_first_initial):
                return None
        return matches[0]

    if len(matches) > 1 and query_first_initial:
        # Multiple last-name matches: narrow by first initial
        initial_matches = [pid for pid in matches
                           if _normalize_name(sp_baselines[pid].get("name", "")).split()[0].startswith(query_first_initial)]
        if len(initial_matches) == 1:
            return initial_matches[0]

        # Still ambiguous (e.g. "Luis Castillo" matches "Luis Miguel Castillo" AND
        # "Luis Felipe Castillo"). Break ties by preferring the pitcher whose
        # stored first name exactly matches the query first name (handles middle names),
        # then fall back to lower ERA (more prominent pitcher).
        if len(initial_matches) > 1:
            query_first = query_tokens[0]
            # Pass 3: exact first-name match (strips middle names from stored name)
            first_name_matches = [pid for pid in initial_matches
                                  if _normalize_name(sp_baselines[pid].get("name", "")).split()[0] == query_first]
            if len(first_name_matches) == 1:
                return first_name_matches[0]
            candidates = first_name_matches if first_name_matches else initial_matches
            # Pass 4: prefer the pitcher with the lower (better) ERA
            def _era(pid):
                try:
                    return float(sp_baselines[pid].get("era", 99))
                except (TypeError, ValueError):
                    return 99.0
            return min(candidates, key=_era)

    return None


def get_team_rest_days(game_date):
    """
    Return a dict of {retro_team_code: rest_days} for all teams playing on game_date.
    Looks back up to 7 days to find each team's most recent prior game.
    Teams with no prior game found default to 1 in the model.
    """
    import datetime as _dt
    start = (game_date - _dt.timedelta(days=7)).strftime("%Y-%m-%d")
    end   = game_date.strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            MLB_SCHEDULE_URL,
            params={"sportId": 1, "startDate": start, "endDate": end,
                    "gameType": "R,S"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[schedule_fetcher] get_team_rest_days error: {e}")
        return {}

    # Build {mlb_team_id: last_game_date} from completed/scheduled games BEFORE today
    last_played = {}
    for date_entry in data.get("dates", []):
        d = _dt.date.fromisoformat(date_entry["date"])
        if d >= game_date:
            continue  # only games strictly before today
        for g in date_entry.get("games", []):
            for side in ("away", "home"):
                tid = g["teams"][side]["team"]["id"]
                if tid not in last_played or d > last_played[tid]:
                    last_played[tid] = d

    # Map MLB team IDs → retro codes and compute rest days.
    # Definition matches TRAINING: days since last game, clipped to [1, 7]
    # (MLBModel.compute_rolling_team_features: (date - prev_date).days, clip(1, 7)).
    # The old "- 1" gave played-yesterday teams 0 while training gave them 1 — a
    # systematic train/live off-by-one on diff_rest_days.
    rest = {}
    for mlb_id, last_d in last_played.items():
        retro = MLB_TEAM_ID_TO_RETRO.get(mlb_id)
        if retro:
            rest[retro] = min(max((game_date - last_d).days, 1), 7)

    return rest


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

            status     = game.get("status", {}).get("detailedState", "Scheduled")
            away_rec   = away_info.get("leagueRecord", {})
            home_rec   = home_info.get("leagueRecord", {})
            pk         = game.get("gamePk")
            final      = status in ("Final", "Game Over", "Completed Early")
            away_score = away_info.get("score")
            home_score = home_info.get("score")

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


def get_team_standings():
    """
    Fetch current MLB standings for all teams.
    Returns dict: retro_code → {"wins": int, "losses": int, "win_pct": float}
    """
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": "103,104", "season": date.today().year, "standingsTypes": "regularSeason"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[schedule_fetcher] get_team_standings error: {e}")
        return {}

    standings = {}
    for record in data.get("records", []):
        for team_rec in record.get("teamRecords", []):
            team_id = team_rec.get("team", {}).get("id")
            retro   = MLB_TEAM_ID_TO_RETRO.get(team_id)
            if retro:
                standings[retro] = {
                    "wins":    team_rec.get("wins",   0),
                    "losses":  team_rec.get("losses", 0),
                    "win_pct": float(team_rec.get("winningPercentage", "0") or "0"),
                }
    return standings


# Last-seen quota from The Odds API response headers. Populated on every
# get_mlb_odds call — INCLUDING 401 "out of credits" responses — so the app can
# surface quota exhaustion instead of a silent empty odds map. Read via
# get_last_odds_quota() (imported by app.py for /api/status and /api/debug/odds).
LAST_ODDS_QUOTA = {"remaining": None, "used": None, "status": None, "checked_at": None}


def get_last_odds_quota():
    """Return a copy of the most recent Odds-API quota snapshot (all None until the
    first call this process). remaining==0 with status==401 means the monthly
    credits ran out — no odds will attach until the plan's cycle resets."""
    return dict(LAST_ODDS_QUOTA)


def _record_quota(resp):
    """Record quota from response headers. The Odds API returns x-requests-remaining
    / x-requests-used on BOTH a 200 AND a 401 quota-exhausted response, so this is
    how the app learns it's out of credits rather than silently seeing 0 games."""
    _rem, _used = resp.headers.get("x-requests-remaining"), resp.headers.get("x-requests-used")
    _last = resp.headers.get("x-requests-last")
    LAST_ODDS_QUOTA.update({
        "remaining":  int(_rem)  if _rem  not in (None, "") else None,
        "used":       int(_used) if _used not in (None, "") else None,
        "last_cost":  int(_last) if _last not in (None, "") else None,
        "status":     resp.status_code,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    })
    return LAST_ODDS_QUOTA


def _american_to_raw(ml):
    """Convert American moneyline to raw (pre-vig) implied probability."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)


def _ml_to_dec(ml):
    return ml / 100 + 1 if ml >= 0 else 100 / abs(ml) + 1


def _raw_to_american(p):
    """Raw implied probability -> American moneyline (inverse of _american_to_raw)."""
    if not 0 < p < 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def parse_odds_events(events):
    """Parse a list of Odds-API event objects into one record per event.

    This is the SINGLE parsing path shared by the live endpoint and the historical
    endpoint — the two differ only in the envelope (historical wraps the same event
    list in {timestamp, previous_timestamp, next_timestamp, data:[...]}), so keeping
    one parser is what stops live and backfilled prices from drifting apart.

    Per-bookmaker prices are averaged, then de-vigged so implied probabilities sum to
    1.0. Events whose teams do not resolve are still returned (with away_team/
    home_team None) so callers can audit them; odds_map_from_event_rows drops them.
    Events with no usable h2h prices, or with a malformed line (|ml| < 100), are
    skipped entirely — matching long-standing get_mlb_odds behaviour.
    """
    rows = []
    for event in events or []:
        home_name = event.get("home_team", "")
        away_name = event.get("away_team", "")
        home_retro = resolve_odds_team(home_name)
        away_retro = resolve_odds_team(away_name)

        away_prices, home_prices = [], []
        books = []
        for bm in event.get("bookmakers", []):
            bm_away = bm_home = None
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for outcome in mkt.get("outcomes", []):
                    n = outcome.get("name")
                    pr = outcome.get("price")
                    if pr is None:
                        continue
                    # Validate each quoted price here rather than after averaging:
                    # a real American line is always <=-100 or >=+100, so anything
                    # inside that band is corrupt input and must not pollute the
                    # consensus. (The post-average guard cannot tell corrupt input
                    # apart from a legitimate pick'em.)
                    if abs(pr) < 100:
                        continue
                    if n == away_name:
                        bm_away = pr
                    elif n == home_name:
                        bm_home = pr
            if bm_away is not None:
                away_prices.append(bm_away)
            if bm_home is not None:
                home_prices.append(bm_home)
            if bm_away is not None and bm_home is not None:
                books.append({
                    "name":    bm.get("title", "Unknown"),
                    "away_ml": round(bm_away),
                    "home_ml": round(bm_home),
                })

        if not away_prices or not home_prices:
            continue

        # Consensus line: average in PROBABILITY space, not raw American odds.
        # American odds are discontinuous across the +/-100 boundary -- a pick'em
        # game where books quote +104 and -101 (both ~50%) averages numerically to
        # about -28, which is not a valid line at all and then trips the malformed
        # guard below, silently dropping the whole game. Observed on 1 of 9 games in
        # the 2026-08-13 probe. Averaging implied probabilities is continuous and
        # gives the right answer (-101, implied 0.502) for exactly that case.
        away_raw = sum(_american_to_raw(x) for x in away_prices) / len(away_prices)
        home_raw = sum(_american_to_raw(x) for x in home_prices) / len(home_prices)
        away_ml = _raw_to_american(away_raw)
        home_ml = _raw_to_american(home_raw)

        # Safety net only: after probability-space averaging a real market can no
        # longer land inside +/-100, so this now catches genuinely corrupt input
        # rather than legitimate pick'em games.
        if away_ml is None or home_ml is None or abs(away_ml) < 100 or abs(home_ml) < 100:
            continue

        total = away_raw + home_raw  # >1 due to vig

        # Arbitrage: best away line + best home line across all books
        arbitrage = None
        if books:
            best_away_dec = max(_ml_to_dec(b["away_ml"]) for b in books)
            best_home_dec = max(_ml_to_dec(b["home_ml"]) for b in books)
            arb_pct = 1 / best_away_dec + 1 / best_home_dec
            if arb_pct < 1.0:
                arbitrage = {"exists": True, "profit_pct": round((1 - arb_pct) * 100, 2)}

        commence = event.get("commence_time")
        rows.append({
            "event_id":       event.get("id"),
            "commence_time":  commence,
            # MLB game dates are ET; commence_time is UTC. Any game starting at or
            # after 8 PM ET carries the NEXT UTC calendar date — roughly half the
            # slate — so every date comparison must go through ET, never the raw
            # UTC prefix.
            "game_date_et":   commence_time_to_et_date(commence),
            "away_team_raw":  away_name,
            "home_team_raw":  home_name,
            "away_team":      away_retro,
            "home_team":      home_retro,
            "away_ml":        away_ml,
            "home_ml":        home_ml,
            "away_implied":   round(away_raw / total, 4),
            "home_implied":   round(home_raw / total, 4),
            "overround":      round(total - 1.0, 6),
            "n_books":        len(books),
            "books":          books[:8],   # cap at 8 for display
            "arbitrage":      arbitrage,
        })
    return rows


def commence_time_to_et_date(commence_time):
    """ISO8601 UTC commence_time -> 'YYYY-MM-DD' in America/New_York, or None."""
    if not commence_time:
        return None
    try:
        dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return None


def odds_map_from_event_rows(rows):
    """Collapse parse_odds_events output to the legacy {(away, home): {...}} shape.

    Rows whose teams did not resolve are dropped. NOTE this key cannot represent a
    doubleheader: two games between the same teams on the same day collapse to one
    entry (last wins). That is pre-existing behaviour, preserved here deliberately;
    the historical backfill keys on event_id instead precisely to avoid it.
    """
    odds_map = {}
    for r in rows:
        if not r["away_team"] or not r["home_team"]:
            continue
        odds_map[(r["away_team"], r["home_team"])] = {
            "away_ml":      r["away_ml"],
            "home_ml":      r["home_ml"],
            "away_implied": r["away_implied"],
            "home_implied": r["home_implied"],
            "books":        r["books"],
            "arbitrage":    r["arbitrage"],
        }
    return odds_map


def get_mlb_odds(api_key):
    """
    Fetch current MLB moneyline odds from The Odds API.
    Costs 1 credit per successful (200) call; a 401/429 costs nothing. Returns {}
    on error or missing data — check get_last_odds_quota() to tell "out of credits"
    (status 401, remaining 0) apart from "no games returned".

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
    except Exception as e:
        print(f"[odds] API request failed: {e}")
        return {}

    _record_quota(resp)

    if resp.status_code == 401:
        print(f"[odds] API 401 — key rejected or monthly quota exhausted "
              f"(used={LAST_ODDS_QUOTA['used']}, remaining={LAST_ODDS_QUOTA['remaining']})",
              flush=True)
        return {}
    try:
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"[odds] API error: {e}")
        return {}

    odds_map = odds_map_from_event_rows(parse_odds_events(events))
    if UNMAPPED_ODDS_TEAMS:
        print(f"[odds] WARNING unresolved team names (games dropped): "
              f"{dict(UNMAPPED_ODDS_TEAMS)}", flush=True)
    print(f"[odds] Fetched odds for {len(odds_map)} games")
    return odds_map


def get_historical_mlb_odds(api_key, iso_ts):
    """Fetch a HISTORICAL odds snapshot at iso_ts (e.g. '2026-08-13T14:00:00Z').

    Costs 10 credits per successful call — 10x the live endpoint (cost = 10 x markets
    x regions) — and is available only on paid plans. The API returns the closest
    snapshot at or earlier than iso_ts.

    Returns (meta, raw_json, event_rows):
      meta       {timestamp, previous_timestamp, next_timestamp, status, http_ok}
      raw_json   the exact response body, for archiving so re-parsing is free forever
      event_rows parse_odds_events output (may span several game dates — the endpoint
                 returns a rolling window of UPCOMING events, not a single day, so
                 callers MUST filter on game_date_et)
    Returns (meta, None, []) on any failure; inspect meta['status'].
    """
    if not api_key:
        return {"status": None, "http_ok": False, "error": "no api key"}, None, []
    try:
        resp = requests.get(
            ODDS_API_HISTORICAL_URL,
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "h2h",
                "oddsFormat":  "american",
                "date":        iso_ts,
            },
            timeout=30,
        )
    except Exception as e:
        return {"status": None, "http_ok": False, "error": str(e)}, None, []

    _record_quota(resp)
    meta = {
        "status":   resp.status_code,
        "http_ok":  resp.status_code == 200,
        "timestamp": None, "previous_timestamp": None, "next_timestamp": None,
    }
    if resp.status_code != 200:
        meta["error"] = resp.text[:300]
        return meta, None, []
    try:
        payload = resp.json()
    except Exception as e:
        meta["error"] = f"unparseable json: {e}"
        return meta, None, []

    meta["timestamp"]          = payload.get("timestamp")
    meta["previous_timestamp"] = payload.get("previous_timestamp")
    meta["next_timestamp"]     = payload.get("next_timestamp")
    # The ONLY structural difference from the live endpoint: events are under "data".
    return meta, payload, parse_odds_events(payload.get("data") or [])


def get_odds_quota(api_key):
    """Read the account's remaining/used credits WITHOUT spending one.

    /v4/sports is documented as a free endpoint but still returns the quota headers,
    so this is how a budget guard checks its ceiling before committing to a run.
    Returns the LAST_ODDS_QUOTA snapshot; 'remaining' is None if the call failed.
    """
    if not api_key:
        return dict(LAST_ODDS_QUOTA)
    try:
        resp = requests.get("https://api.the-odds-api.com/v4/sports/",
                            params={"apiKey": api_key}, timeout=15)
    except Exception as e:
        print(f"[odds] quota check failed: {e}", flush=True)
        return dict(LAST_ODDS_QUOTA)
    _record_quota(resp)
    return dict(LAST_ODDS_QUOTA)
