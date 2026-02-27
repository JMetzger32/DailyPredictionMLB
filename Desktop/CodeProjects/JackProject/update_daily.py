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
MIN_GAMES_FOR_LIVE_UPDATE = 3   # below this, keep prior-year baseline

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

    # Season-to-date Pythagorean and win%
    cum_rs = games["runs_scored"].fillna(0).cumsum()
    cum_ra = games["runs_allowed"].fillna(0).cumsum()
    rs = max(float(cum_rs.iloc[-1]), 1)
    ra = max(float(cum_ra.iloc[-1]), 1)
    pyth     = rs ** PYTH_EXP / (rs ** PYTH_EXP + ra ** PYTH_EXP)
    win_pct  = float(games["win"].sum()) / n

    def roll_last(col, window, min_periods):
        """Return the last value in a rolling mean series, or fallback."""
        s = games[col].rolling(window, min_periods=min_periods).mean()
        val = s.iloc[-1]
        if pd.isna(val):
            # Use season mean if rolling window isn't full yet
            val = games[col].mean()
        return float(val) if not pd.isna(val) else fallback.get(col, FALLBACK_TEAM.get(col, 0))

    # Compute per-game OBP and SLG (need at_bats, doubles, triples, etc.)
    ab  = games.get("at_bats",     pd.Series(30,  index=games.index)).fillna(30).clip(lower=1)
    hbp = games.get("hit_by_pitch",pd.Series(0,   index=games.index)).fillna(0)
    sf  = games.get("sac_flies",   pd.Series(0,   index=games.index)).fillna(0)
    d   = games.get("doubles",     pd.Series(0,   index=games.index)).fillna(0)
    t   = games.get("triples",     pd.Series(0,   index=games.index)).fillna(0)
    h   = games["hits"].fillna(0)
    bb  = games["walks"].fillna(0)
    hr  = games["homeruns"].fillna(0)
    games = games.copy()
    games["obp_game"] = (h + bb + hbp) / (ab + bb + hbp + sf).clip(lower=1)
    games["slg_game"] = (h + d + 2 * t + 3 * hr) / ab

    # For errors and pitchers_used, only use rolling if we have enough data points
    errors_ok  = int(games["errors"].notna().sum()) >= 3
    bullpen_ok = int(games["pitchers_used"].notna().sum()) >= 1
    ok_k       = int(games.get("opp_strikeouts", pd.Series()).notna().sum()) >= 3

    return {
        "pyth_win_pct":          round(pyth, 4),
        "win_pct":               round(win_pct, 4),
        "hits_per_game":         roll_last("hits",            30, 5),
        "opp_hits_per_game":     roll_last("opp_hits",        30, 5),
        "walks_per_game":        roll_last("walks",           30, 5),
        "opp_walks_per_game":    roll_last("opp_walks",       30, 5),
        "errors_per_game":       roll_last("errors",          30, 5) if errors_ok
                                 else fallback["errors_per_game"],
        "hr_per_game":           roll_last("homeruns",        30, 5),
        "opp_hr_per_game":       roll_last("opp_homeruns",    30, 5),
        "recent_win_pct":        roll_last("win",             10, 3),
        "recent_hr_per_game":    roll_last("homeruns",        10, 3),
        "bullpen_used":          roll_last("pitchers_used",    3, 1) if bullpen_ok
                                 else fallback["bullpen_used"],
        # New rate stats
        "obp":                   roll_last("obp_game",        30, 5),
        "slg":                   roll_last("slg_game",        30, 5),
        "recent_runs_per_game":  roll_last("runs_scored",     10, 3),
        "opp_k_per_game":        roll_last("opp_strikeouts",  30, 5) if ok_k
                                 else fallback["opp_k_per_game"],
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
    min_gs = 2 if games_played < 15 else 5

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
                  f"BP={baseline['bullpen_used']:.1f}")
        else:
            # Not enough live data — keep prior-year baseline
            new_team_baselines[retro_code] = old_team_baselines.get(
                retro_code, dict(FALLBACK_TEAM)
            )
            print(f"-> kept prior baseline (< {MIN_GAMES_FOR_LIVE_UPDATE} games)")

    avg_games = int(sum(games_fetched) / len(games_fetched)) if games_fetched else 0
    print(f"\n  Updated {updated_count}/30 teams with live {SEASON} data "
          f"(avg {avg_games} games)")

    # ---- SP baselines --------------------------------------------------
    print()
    new_sp_baselines = fetch_sp_baselines(SEASON, games_played=avg_games)

    if new_sp_baselines and len(new_sp_baselines) >= 10:
        print(f"  Built {len(new_sp_baselines)} SP baselines from FanGraphs")
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
