"""
update_daily.py — Daily baseline refresh for MLB Game Predictor.

Pulls live 2026 game-by-game data via pybaseball, recomputes rolling
team and pitcher baselines, and overwrites mlb_model_artifacts.pkl.
The trained model weights are NOT retrained — only the baselines change.

Data sources:
  - team_game_logs()  -> batting + pitching game logs (Baseball Reference)
  - pitching_stats()  -> season pitcher stats (FanGraphs)

To schedule at 8 AM daily via cron:
  crontab -e
  0 8 * * * cd /Users/jackmetzger/Desktop/CodeProjects/JackProject && python3 update_daily.py >> update_log.txt 2>&1
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pybaseball import team_game_logs, pitching_stats, cache

warnings.filterwarnings("ignore")
cache.enable()  # Cache API responses to avoid hammering servers on repeated runs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEASON        = 2026
ARTIFACTS_PATH = "mlb_model_artifacts.pkl"
PYTH_EXP      = 1.83
MIN_GAMES_FOR_LIVE_UPDATE = 1   # process team if any RS games exist; thresholds handled inside
RS_START_DATE = pd.Timestamp(f"{SEASON}-03-25")  # regular season never starts before this

# Games thresholds for switching from prior-year to live stats
BATTING_LIVE_THRESHOLD = 5    # hitting stats (OBP, SLG, ISO, hits, runs, etc.)
RECORD_LIVE_THRESHOLD  = 12   # win%, pythagorean win%, recent win%

# Retrosheet team code -> MLB Stats API team ID (for hitting/pitching fallback)
RETRO_TO_MLB_ID = {
    "ANA": 108, "ARI": 109, "ATH": 133, "ATL": 144,
    "BAL": 110, "BOS": 111, "CHA": 145, "CHN": 112,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KCA": 118, "LAN": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYA": 147, "NYN": 121,
    "PHI": 143, "PIT": 134, "SDN": 135, "SEA": 136,
    "SFN": 137, "SLN": 138, "TBA": 139, "TEX": 140,
    "TOR": 141, "WAS": 120,
}

# Retrosheet team code -> Baseball Reference team code
RETRO_TO_BR = {
    "ANA": "LAA",   # Angels
    "ARI": "ARI",
    "ATH": "OAK",   # Athletics (Sacramento)
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHA": "CWS",   # White Sox
    "CHN": "CHC",   # Cubs
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KCA": "KCR",   # Royals
    "LAN": "LAD",   # Dodgers
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYA": "NYY",
    "NYN": "NYM",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDN": "SDP",   # Padres
    "SEA": "SEA",
    "SFN": "SFG",
    "SLN": "STL",
    "TBA": "TBR",   # Rays
    "TEX": "TEX",
    "TOR": "TOR",
    "WAS": "WSH",   # Nationals
}

# League-average fallbacks used when pybaseball data is unavailable
FALLBACK_TEAM = {
    "pyth_win_pct":          0.500,
    "win_pct":               0.500,
    "hits_per_game":         8.5,
    "opp_hits_per_game":     8.5,
    "walks_per_game":        3.3,
    "opp_walks_per_game":    3.3,
    "errors_per_game":       0.6,
    "hr_per_game":           1.1,
    "opp_hr_per_game":       1.1,
    "recent_win_pct":        0.500,
    "recent_hr_per_game":    1.1,
    "bullpen_used":          4.0,
    # New rate stats
    "obp":                   0.318,
    "slg":                   0.400,
    "iso":                   0.150,   # league-average ISO (SLG - AVG)
    "recent_runs_per_game":  4.5,
    "opp_k_per_game":        8.5,
}


# ---------------------------------------------------------------------------
# Date parsing helper (BR game logs use formats like "Apr  1" or "Apr 1(1)")
# ---------------------------------------------------------------------------
def _parse_br_dates(date_series, season):
    """Parse Baseball Reference date strings into datetime objects."""
    # Strip doubleheader suffixes like "(1)" or "(2)"
    cleaned = date_series.astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True).str.strip()
    # Try "Mon DD YYYY" (add season year)
    parsed = pd.to_datetime(cleaned + f" {season}", format="%b %d %Y", errors="coerce")
    # Fallback: try ISO format
    mask = parsed.isna()
    if mask.any():
        parsed[mask] = pd.to_datetime(cleaned[mask], errors="coerce")
    return parsed


# ---------------------------------------------------------------------------
# Fetch batting game log (offensive stats per game)
# ---------------------------------------------------------------------------
def _fetch_batting_log(br_team, season):
    try:
        df = team_game_logs(season, br_team, log_type="batting")
    except Exception as e:
        print(f"[WARN] batting log failed for {br_team}: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    # Keep completed games only
    df = df[df["W/L"].astype(str).str.match(r"^[WL]", na=False)].copy()
    if len(df) == 0:
        return None

    df["date"] = _parse_br_dates(df["Date"], season)
    df = df.dropna(subset=["date"])

    # Strip spring training — keep only regular season games
    df = df[df["date"] >= RS_START_DATE]
    if len(df) == 0:
        return None

    df["win"]          = df["W/L"].astype(str).str.startswith("W").astype(int)
    df["runs_scored"]  = pd.to_numeric(df.get("R"),   errors="coerce")
    df["runs_allowed"] = pd.to_numeric(df.get("RA"),  errors="coerce")
    df["hits"]         = pd.to_numeric(df.get("H"),   errors="coerce")
    df["walks"]        = pd.to_numeric(df.get("BB"),  errors="coerce")
    df["homeruns"]     = pd.to_numeric(df.get("HR"),  errors="coerce")
    # For OBP/SLG rate stats
    df["at_bats"]      = pd.to_numeric(df.get("AB"),  errors="coerce")
    df["doubles"]      = pd.to_numeric(df.get("2B"),  errors="coerce")
    df["triples"]      = pd.to_numeric(df.get("3B"),  errors="coerce")
    df["hit_by_pitch"] = pd.to_numeric(df.get("HBP"), errors="coerce").fillna(0)
    df["sac_flies"]    = pd.to_numeric(df.get("SF"),  errors="coerce").fillna(0)

    return df[["date", "win", "runs_scored", "runs_allowed", "hits", "walks", "homeruns",
               "at_bats", "doubles", "triples", "hit_by_pitch", "sac_flies"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fetch pitching game log (defensive stats + bullpen workload per game)
# ---------------------------------------------------------------------------
def _fetch_pitching_log(br_team, season):
    try:
        df = team_game_logs(season, br_team, log_type="pitching")
    except Exception as e:
        print(f"[WARN] pitching log failed for {br_team}: {e}")
        return None

    if df is None or len(df) == 0:
        return None

    df = df[df["W/L"].astype(str).str.match(r"^[WL]", na=False)].copy()
    if len(df) == 0:
        return None

    df["date"] = _parse_br_dates(df["Date"], season)
    df = df.dropna(subset=["date"])

    # Strip spring training — keep only regular season games
    df = df[df["date"] >= RS_START_DATE]
    if len(df) == 0:
        return None

    # Pitchers used — BR column is "#P"
    pused_col = next((c for c in df.columns if str(c).strip() == "#P"), None)
    df["pitchers_used"] = pd.to_numeric(df[pused_col], errors="coerce") if pused_col else np.nan

    df["opp_hits"]       = pd.to_numeric(df.get("H"),  errors="coerce")
    df["opp_walks"]      = pd.to_numeric(df.get("BB"), errors="coerce")
    df["opp_homeruns"]   = pd.to_numeric(df.get("HR"), errors="coerce")
    df["errors"]         = pd.to_numeric(df.get("E"),  errors="coerce")
    df["opp_strikeouts"] = pd.to_numeric(df.get("SO"), errors="coerce")

    return df[["date", "opp_hits", "opp_walks", "opp_homeruns",
               "errors", "pitchers_used", "opp_strikeouts"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combine batting + pitching logs for one team
# ---------------------------------------------------------------------------
def fetch_team_games(br_team, season):
    bat = _fetch_batting_log(br_team, season)
    if bat is None:
        return None

    pit = _fetch_pitching_log(br_team, season)
    if pit is not None:
        merged = bat.merge(pit, on="date", how="left")
    else:
        merged = bat.copy()
        for col in ["opp_hits", "opp_walks", "opp_homeruns", "errors", "pitchers_used"]:
            merged[col] = np.nan

    return merged.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# MLB Stats API fallback — used when Baseball Reference is blocked
# ---------------------------------------------------------------------------
def fetch_team_baseline_from_mlb_api(retro_code, season, old_baseline=None):
    """
    Fetch season-to-date team stats from the free MLB Stats API.
    Used as fallback when pybaseball/Baseball Reference is unavailable.
    Returns a baseline dict in the same format as compute_team_baseline(),
    or None on failure.
    """
    import requests as _req
    mlb_id = RETRO_TO_MLB_ID.get(retro_code)
    if not mlb_id:
        return None
    fallback = old_baseline or dict(FALLBACK_TEAM)

    try:
        # Hitting stats
        hit_url = (f"https://statsapi.mlb.com/api/v1/teams/{mlb_id}/stats"
                   f"?stats=season&group=hitting&season={season}")
        hit_r = _req.get(hit_url, timeout=10)
        hit_r.raise_for_status()
        hit_splits = hit_r.json().get("stats", [{}])[0].get("splits", [{}])
        if not hit_splits:
            return None
        h = hit_splits[0].get("stat", {})

        games_played = int(h.get("gamesPlayed", 0))
        if games_played < BATTING_LIVE_THRESHOLD:
            return None  # not enough games, keep prior baseline

        ab   = float(h.get("atBats",    1)) or 1
        hits = float(h.get("hits",      0))
        bb   = float(h.get("baseOnBalls", 0))
        hbp  = float(h.get("hitByPitch",  0))
        sf   = float(h.get("sacFlies",    0))
        hr   = float(h.get("homeRuns",    0))
        slg_str = h.get("slg", "0") or "0"
        obp_str = h.get("obp", "0") or "0"
        avg_str = h.get("avg", "0") or "0"
        slg  = float(slg_str)
        obp  = float(obp_str)
        avg  = float(avg_str)
        iso  = round(slg - avg, 4)
        runs = float(h.get("runs",    0))

        # Pitching/defense stats
        pit_url = (f"https://statsapi.mlb.com/api/v1/teams/{mlb_id}/stats"
                   f"?stats=season&group=pitching&season={season}")
        pit_r = _req.get(pit_url, timeout=10)
        pit_r.raise_for_status()
        pit_splits = pit_r.json().get("stats", [{}])[0].get("splits", [{}])
        p = pit_splits[0].get("stat", {}) if pit_splits else {}

        opp_hits   = float(p.get("hits",        0))
        opp_bb     = float(p.get("baseOnBalls", 0))
        opp_hr     = float(p.get("homeRuns",    0))
        opp_k      = float(p.get("strikeOuts",  0))
        era_str    = p.get("era", "4.50") or "4.50"
        bullpen_era = float(era_str)

        g = max(games_played, 1)

        # Win% — use standings API
        win_pct    = fallback.get("win_pct",      0.500)
        pyth       = fallback.get("pyth_win_pct", 0.500)
        recent_wp  = fallback.get("recent_win_pct", 0.500)
        if games_played >= RECORD_LIVE_THRESHOLD:
            try:
                std_url = (f"https://statsapi.mlb.com/api/v1/standings"
                           f"?leagueId=103,104&season={season}&standingsTypes=regularSeason")
                std_r = _req.get(std_url, timeout=10)
                std_r.raise_for_status()
                for div in std_r.json().get("records", []):
                    for team_rec in div.get("teamRecords", []):
                        if team_rec.get("team", {}).get("id") == mlb_id:
                            w  = float(team_rec.get("wins",   0))
                            l  = float(team_rec.get("losses", 0))
                            rs = float(team_rec.get("runsScored",   0) or 0)
                            ra = float(team_rec.get("runsAllowed",  0) or 0)
                            if w + l > 0:
                                win_pct   = round(w / (w + l), 4)
                                recent_wp = win_pct
                            if rs > 0 and ra > 0:
                                pyth = round(rs**PYTH_EXP / (rs**PYTH_EXP + ra**PYTH_EXP), 4)
            except Exception:
                pass

        return {
            "pyth_win_pct":         pyth,
            "win_pct":              win_pct,
            "hits_per_game":        round(hits / g, 3),
            "opp_hits_per_game":    round(opp_hits / g, 3),
            "walks_per_game":       round(bb / g, 3),
            "opp_walks_per_game":   round(opp_bb / g, 3),
            "errors_per_game":      fallback.get("errors_per_game", 0.6),
            "hr_per_game":          round(hr / g, 3),
            "opp_hr_per_game":      round(opp_hr / g, 3),
            "recent_win_pct":       recent_wp,
            "recent_hr_per_game":   round(hr / g, 3),
            "bullpen_used":         fallback.get("bullpen_used", 4.0),
            "bullpen_era":          round(bullpen_era, 3),
            "obp":                  obp,
            "slg":                  slg,
            "iso":                  iso,
            "recent_runs_per_game": round(runs / g, 3),
            "opp_k_per_game":       round(opp_k / g, 3),
        }
    except Exception as e:
        print(f"[WARN] MLB API stats failed for {retro_code}: {e}")
        return None


# ---------------------------------------------------------------------------
# Compute rolling baselines from a team's game log
# ---------------------------------------------------------------------------
def compute_team_baseline(games, old_baseline=None):
    """
    Compute current-state rolling stats from a per-game DataFrame.
    Falls back to old_baseline for any stat that lacks enough data.
    """
    fallback = old_baseline or dict(FALLBACK_TEAM)

    if games is None or len(games) == 0:
        return fallback

    games = games.sort_values("date").reset_index(drop=True)
    n = len(games)

    # Graduated thresholds: use live stats only once enough RS games played
    use_batting_live = n >= BATTING_LIVE_THRESHOLD   # hitting, pitching defense stats
    use_record_live  = n >= RECORD_LIVE_THRESHOLD    # win%, pythagorean, recent win%

    def fb(key):
        """Return fallback value for a stat key."""
        return fallback.get(key, FALLBACK_TEAM.get(key, 0))

    def roll_last(col, window, min_periods):
        """Return the last value in a rolling mean series, or season mean."""
        s = games[col].rolling(window, min_periods=min_periods).mean()
        val = s.iloc[-1]
        if pd.isna(val):
            val = games[col].mean()
        return float(val) if not pd.isna(val) else fb(col)

    # Season-to-date Pythagorean and win% (only used if use_record_live)
    cum_rs = games["runs_scored"].fillna(0).cumsum()
    cum_ra = games["runs_allowed"].fillna(0).cumsum()
    rs = max(float(cum_rs.iloc[-1]), 1)
    ra = max(float(cum_ra.iloc[-1]), 1)
    pyth    = rs ** PYTH_EXP / (rs ** PYTH_EXP + ra ** PYTH_EXP)
    win_pct = float(games["win"].sum()) / n

    # Compute per-game OBP, SLG, ISO (only used if use_batting_live)
    ab  = games.get("at_bats",     pd.Series(30, index=games.index)).fillna(30).clip(lower=1)
    hbp = games.get("hit_by_pitch",pd.Series(0,  index=games.index)).fillna(0)
    sf  = games.get("sac_flies",   pd.Series(0,  index=games.index)).fillna(0)
    d   = games.get("doubles",     pd.Series(0,  index=games.index)).fillna(0)
    t   = games.get("triples",     pd.Series(0,  index=games.index)).fillna(0)
    h   = games["hits"].fillna(0)
    bb  = games["walks"].fillna(0)
    hr  = games["homeruns"].fillna(0)
    games = games.copy()
    games["obp_game"] = (h + bb + hbp) / (ab + bb + hbp + sf).clip(lower=1)
    games["slg_game"] = (h + d + 2 * t + 3 * hr) / ab
    games["iso_game"] = (d + 2 * t + 3 * hr) / ab

    errors_ok  = int(games["errors"].notna().sum()) >= 3
    bullpen_ok = int(games["pitchers_used"].notna().sum()) >= 1
    ok_k       = int(games.get("opp_strikeouts", pd.Series()).notna().sum()) >= 3

    return {
        # Win/record stats — live at 12+ games, else prior-year fallback
        "pyth_win_pct":          round(pyth, 4)    if use_record_live  else fb("pyth_win_pct"),
        "win_pct":               round(win_pct, 4) if use_record_live  else fb("win_pct"),
        "recent_win_pct":        roll_last("win", 10, 3)        if use_record_live  else fb("recent_win_pct"),
        "recent_hr_per_game":    roll_last("homeruns", 10, 3)   if use_record_live  else fb("recent_hr_per_game"),
        # Batting/pitching stats — live at 5+ games, else prior-year fallback
        "hits_per_game":         roll_last("hits",           30, 5) if use_batting_live else fb("hits_per_game"),
        "opp_hits_per_game":     roll_last("opp_hits",       30, 5) if use_batting_live else fb("opp_hits_per_game"),
        "walks_per_game":        roll_last("walks",          30, 5) if use_batting_live else fb("walks_per_game"),
        "opp_walks_per_game":    roll_last("opp_walks",      30, 5) if use_batting_live else fb("opp_walks_per_game"),
        "errors_per_game":       roll_last("errors",         30, 5) if (use_batting_live and errors_ok)  else fb("errors_per_game"),
        "hr_per_game":           roll_last("homeruns",       30, 5) if use_batting_live else fb("hr_per_game"),
        "opp_hr_per_game":       roll_last("opp_homeruns",   30, 5) if use_batting_live else fb("opp_hr_per_game"),
        "bullpen_used":          roll_last("pitchers_used",   3, 1) if (use_batting_live and bullpen_ok) else fb("bullpen_used"),
        "obp":                   roll_last("obp_game",       30, 5) if use_batting_live else fb("obp"),
        "slg":                   roll_last("slg_game",       30, 5) if use_batting_live else fb("slg"),
        "iso":                   roll_last("iso_game",       30, 5) if use_batting_live else fb("iso"),
        "recent_runs_per_game":  roll_last("runs_scored",    10, 3) if use_batting_live else fb("recent_runs_per_game"),
        "opp_k_per_game":        roll_last("opp_strikeouts", 30, 5) if (use_batting_live and ok_k)       else fb("opp_k_per_game"),
    }


# ---------------------------------------------------------------------------
# Fetch starting pitcher season stats from FanGraphs
# ---------------------------------------------------------------------------
def fetch_sp_baselines(season, games_played):
    """
    Pull FanGraphs pitcher stats and return sp_baselines dict.
    games_played: estimated games per team so far (to set GS threshold).
    """
    # Early season: lower GS threshold
    # Require 3+ GS before trusting 2026 ERA (1-2 start samples too noisy: 0.00/9.00 ERA)
    # 3 starts ≈ 18 innings, reasonably stable. Switch to 5 after ~1 month.
    min_gs = 3 if games_played < 20 else 5

    try:
        sp = pitching_stats(season, qual=1)
    except Exception as e:
        print(f"  [WARN] FanGraphs pitcher stats failed: {e}")
        return None

    if sp is None or len(sp) == 0:
        return None

    # Filter to starters
    sp = sp[pd.to_numeric(sp.get("GS", 0), errors="coerce").fillna(0) >= min_gs].copy()
    if len(sp) == 0:
        return None

    print(f"  Found {len(sp)} starters (GS >= {min_gs})")

    def safe(row, col, default):
        v = pd.to_numeric(row.get(col, default), errors="coerce")
        return float(v) if not pd.isna(v) else default

    sp_baselines = {}
    for _, row in sp.iterrows():
        name = str(row.get("Name", "Unknown")).strip()
        # Build a slug key for name-based lookup (predict.py searches by partial name)
        key = (name.lower()
               .replace(" ", "_")
               .replace(".", "")
               .replace("'", "")
               .replace("-", "_"))

        sp_baselines[key] = {
            "name":   name,
            "era":    safe(row, "ERA",  4.0),
            "whip":   safe(row, "WHIP", 1.3),
            "xfip":   safe(row, "xFIP", 4.0),
            "siera":  safe(row, "SIERA",4.0),
            "so9":    safe(row, "K/9",  8.0),
            "bb9":    safe(row, "BB/9", 3.0),
            "hr9":    safe(row, "HR/9", 1.2),
            "wins":   int(safe(row, "W", 0)),
            "losses": int(safe(row, "L", 0)),
        }

    return sp_baselines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] MLB Daily Update — {SEASON} season")
    print("=" * 60)

    # Load existing artifacts — preserve the trained model and scaler
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    old_team_baselines = artifacts.get("team_baselines", {})
    old_sp_baselines   = artifacts.get("sp_baselines",   {})
    print(f"  Loaded artifacts  ({len(old_team_baselines)} teams, {len(old_sp_baselines)} pitchers)")

    # ---- Team baselines ------------------------------------------------
    print(f"\n  Fetching {SEASON} game logs for all 30 teams...")
    new_team_baselines = {}
    games_fetched = []
    updated_count = 0

    for retro_code, br_code in RETRO_TO_BR.items():
        print(f"    {retro_code} ({br_code:<3s}) ", end="", flush=True)
        games = fetch_team_games(br_code, SEASON)

        if games is not None and len(games) >= MIN_GAMES_FOR_LIVE_UPDATE:
            games_fetched.append(len(games))
            baseline = compute_team_baseline(games, old_team_baselines.get(retro_code))
            new_team_baselines[retro_code] = baseline
            updated_count += 1
            print(f"-> {len(games):3d} games  W={baseline['win_pct']:.3f}  "
                  f"BP={baseline.get('bullpen_used', 0):.1f}")
        else:
            # BR unavailable — try MLB Stats API fallback
            api_baseline = fetch_team_baseline_from_mlb_api(
                retro_code, SEASON, old_team_baselines.get(retro_code)
            )
            if api_baseline:
                new_team_baselines[retro_code] = api_baseline
                updated_count += 1
                games_fetched.append(9)  # approximate
                print(f"-> MLB API (season stats)  W={api_baseline['win_pct']:.3f}  "
                      f"OBP={api_baseline['obp']:.3f}")
            else:
                new_team_baselines[retro_code] = old_team_baselines.get(
                    retro_code, dict(FALLBACK_TEAM)
                )
                print(f"-> kept prior baseline (BR + MLB API both failed)")

    avg_games = int(sum(games_fetched) / len(games_fetched)) if games_fetched else 0
    print(f"\n  Updated {updated_count}/30 teams with live {SEASON} data "
          f"(avg {avg_games} games)")

    # ---- SP baselines --------------------------------------------------
    print()
    new_sp_baselines = fetch_sp_baselines(SEASON, games_played=avg_games)

    if new_sp_baselines and len(new_sp_baselines) >= 10:
        print(f"  Built {len(new_sp_baselines)} SP baselines from FanGraphs")
        # Fetch prior-year data as fallback for pitchers not yet in 2026
        prior_sp = fetch_sp_baselines(SEASON - 1, games_played=162)
        if prior_sp:
            merged_sp = {**prior_sp, **new_sp_baselines}
            print(f"  Merged to {len(merged_sp)} total pitchers (2026 data + {SEASON-1} fallback)")
        else:
            merged_sp = {**old_sp_baselines, **new_sp_baselines}
            print(f"  Merged to {len(merged_sp)} total pitchers (2026 data + prior fallback)")
        new_sp_baselines = merged_sp
    else:
        print(f"  Insufficient {SEASON} pitcher data — keeping prior SP baselines")
        new_sp_baselines = old_sp_baselines

    # ---- Save updated artifacts ----------------------------------------
    artifacts["team_baselines"] = new_team_baselines
    artifacts["sp_baselines"]   = new_sp_baselines
    # Model and scaler are preserved untouched

    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n  Saved updated artifacts -> {ARTIFACTS_PATH}")
    print(f"  Teams with live data: {sorted(new_team_baselines.keys())}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Done!")


if __name__ == "__main__":
    main()
