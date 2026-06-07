"""
fetch_advanced_pitching.py — Pull xFIP, SIERA, and bullpen ERA from FanGraphs
via pybaseball and store in mlb_allseasons.db.

Run this once before training the model:
  python3 fetch_advanced_pitching.py

What it adds to the database:
  - pitcher_stats.xfip  — xFIP per pitcher-season (better than FIP for future prediction)
  - pitcher_stats.siera — SIERA per pitcher-season (contact-quality adjusted)
  - team_bullpen_stats  — new table: (team, season, bullpen_era, bullpen_ip)
"""

import sqlite3
import unicodedata
import re
import warnings
import numpy as np
import pandas as pd
from pybaseball import pitching_stats, cache

warnings.filterwarnings("ignore")
cache.enable()

DB_PATH = "mlb_allseasons.db"
SEASONS = [2021, 2022, 2023, 2024, 2025]

# FanGraphs -> Retrosheet team code mapping
FG_TO_RETRO = {
    "LAA": "ANA",  "Angels":        "ANA",
    "ARI": "ARI",  "Diamondbacks":  "ARI",
    "ATL": "ATL",  "Braves":        "ATL",
    "BAL": "BAL",  "Orioles":       "BAL",
    "BOS": "BOS",  "Red Sox":       "BOS",
    "CHC": "CHN",  "Cubs":          "CHN",
    "CWS": "CHA",  "White Sox":     "CHA",
    "CIN": "CIN",  "Reds":          "CIN",
    "CLE": "CLE",  "Guardians":     "CLE",  "Indians": "CLE",
    "COL": "COL",  "Rockies":       "COL",
    "DET": "DET",  "Tigers":        "DET",
    "HOU": "HOU",  "Astros":        "HOU",
    "KCR": "KCA",  "Royals":        "KCA",
    "LAD": "LAN",  "Dodgers":       "LAN",
    "MIA": "MIA",  "Marlins":       "MIA",
    "MIL": "MIL",  "Brewers":       "MIL",
    "MIN": "MIN",  "Twins":         "MIN",
    "NYM": "NYN",  "Mets":          "NYN",
    "NYY": "NYA",  "Yankees":       "NYA",
    "OAK": "ATH",  "Athletics":     "ATH",
    "PHI": "PHI",  "Phillies":      "PHI",
    "PIT": "PIT",  "Pirates":       "PIT",
    "SDP": "SDN",  "Padres":        "SDN",
    "SEA": "SEA",  "Mariners":      "SEA",
    "SFG": "SFN",  "Giants":        "SFN",
    "STL": "SLN",  "Cardinals":     "SLN",
    "TBR": "TBA",  "Rays":          "TBA",
    "TEX": "TEX",  "Rangers":       "TEX",
    "TOR": "TOR",  "Blue Jays":     "TOR",
    "WSN": "WAS",  "Nationals":     "WAS",
}


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.replace("*", "").replace("#", "").strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    name = name.replace(".", "")
    return name


def fetch_season_pitching(season):
    """Pull all pitcher stats from FanGraphs for a season."""
    print(f"  Fetching FanGraphs pitcher stats for {season}...", end=" ", flush=True)
    try:
        df = pitching_stats(season, qual=1)
        print(f"{len(df)} pitchers")
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def add_advanced_columns_to_pitcher_stats(conn):
    """Add xfip and siera columns to pitcher_stats if they don't exist."""
    cursor = conn.cursor()
    existing = [row[1] for row in cursor.execute("PRAGMA table_info(pitcher_stats)").fetchall()]
    if "xfip" not in existing:
        cursor.execute("ALTER TABLE pitcher_stats ADD COLUMN xfip REAL")
        print("  Added xfip column to pitcher_stats")
    if "siera" not in existing:
        cursor.execute("ALTER TABLE pitcher_stats ADD COLUMN siera REAL")
        print("  Added siera column to pitcher_stats")
    conn.commit()


def update_starter_advanced_stats(conn, season, fg_df):
    """Match FanGraphs starters to pitcher_stats and update xFIP/SIERA."""
    # Filter to starters
    starters = fg_df[pd.to_numeric(fg_df.get("GS", 0), errors="coerce").fillna(0) >= 3].copy()

    # Pull existing pitcher_stats for this season to match against
    ps = pd.read_sql_query(
        f"SELECT id, retro_pitcher_id, player_name, retro_team FROM pitcher_stats WHERE season = {season}",
        conn
    )
    if len(ps) == 0:
        print(f"    No pitcher_stats rows for {season} — skipping update")
        return 0

    ps["name_clean"] = ps["player_name"].apply(normalize_name)

    # Clean FanGraphs names
    starters = starters.copy()
    starters["name_clean"] = starters["Name"].apply(normalize_name)
    starters["retro_team"] = starters["Team"].map(FG_TO_RETRO)

    def safe_float(val, default=None):
        try:
            v = float(val)
            return v if not np.isnan(v) else default
        except (TypeError, ValueError):
            return default

    updated = 0
    cursor = conn.cursor()
    for _, fg_row in starters.iterrows():
        # Match by name + team first, then name only
        match = ps[
            (ps["name_clean"] == fg_row["name_clean"]) &
            (ps["retro_team"] == fg_row.get("retro_team"))
        ]
        if len(match) == 0:
            match = ps[ps["name_clean"] == fg_row["name_clean"]]
        if len(match) == 0:
            continue

        xfip  = safe_float(fg_row.get("xFIP"))
        siera = safe_float(fg_row.get("SIERA"))

        if xfip is None and siera is None:
            continue

        for _, ps_row in match.iterrows():
            cursor.execute(
                "UPDATE pitcher_stats SET xfip = ?, siera = ? WHERE id = ?",
                (xfip, siera, ps_row["id"])
            )
            updated += 1

    conn.commit()
    return updated


def build_bullpen_stats(season, fg_df):
    """
    Compute team-level bullpen ERA from FanGraphs reliever data.
    Relievers = pitchers with GS == 0 (or very few starts).
    Returns a DataFrame with (retro_team, season, bullpen_era, bullpen_ip).
    """
    # Filter to relievers (0 GS)
    gs_col = pd.to_numeric(fg_df.get("GS", 0), errors="coerce").fillna(0)
    relievers = fg_df[gs_col == 0].copy()

    if len(relievers) == 0:
        return pd.DataFrame()

    relievers["retro_team"] = relievers["Team"].map(FG_TO_RETRO)
    relievers["ip"]  = pd.to_numeric(relievers.get("IP",  0), errors="coerce").fillna(0)
    relievers["era"] = pd.to_numeric(relievers.get("ERA", np.nan), errors="coerce")
    relievers = relievers.dropna(subset=["era"])
    relievers = relievers[relievers["retro_team"].notna() & (relievers["ip"] > 0)]

    if len(relievers) == 0:
        return pd.DataFrame()

    # IP-weighted ERA per team
    def weighted_era(group):
        total_ip = group["ip"].sum()
        if total_ip == 0:
            return np.nan
        return (group["era"] * group["ip"]).sum() / total_ip

    team_bp = relievers.groupby("retro_team").apply(weighted_era).reset_index()
    team_bp.columns = ["retro_team", "bullpen_era"]
    team_bp["bullpen_ip"] = relievers.groupby("retro_team")["ip"].sum().values
    team_bp["season"] = season
    return team_bp[["retro_team", "season", "bullpen_era", "bullpen_ip"]]


def create_bullpen_table(conn):
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS team_bullpen_stats")
    cursor.execute("""
        CREATE TABLE team_bullpen_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            retro_team  TEXT,
            season      INTEGER,
            bullpen_era REAL,
            bullpen_ip  REAL
        )
    """)
    conn.commit()


def main():
    print("=" * 60)
    print("Fetch Advanced Pitching Stats (FanGraphs via pybaseball)")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # Prepare pitcher_stats table
    add_advanced_columns_to_pitcher_stats(conn)

    all_bullpen = []
    total_updated = 0

    for season in SEASONS:
        print(f"\n[{season}]")
        fg_df = fetch_season_pitching(season)
        if fg_df is None:
            print(f"  Skipping {season}")
            continue

        # Update xFIP/SIERA in pitcher_stats
        n = update_starter_advanced_stats(conn, season, fg_df)
        total_updated += n
        print(f"  Updated xFIP/SIERA for {n} pitcher-season rows")

        # Build bullpen stats
        bp = build_bullpen_stats(season, fg_df)
        if len(bp) > 0:
            all_bullpen.append(bp)
            print(f"  Bullpen ERA computed for {len(bp)} teams")
            print(f"    Sample: {bp[['retro_team','bullpen_era']].sort_values('retro_team').to_string(index=False)}")

    # Save bullpen stats table
    print(f"\n  Creating team_bullpen_stats table...")
    create_bullpen_table(conn)
    if all_bullpen:
        bullpen_df = pd.concat(all_bullpen, ignore_index=True)
        bullpen_df.to_sql("team_bullpen_stats", conn, if_exists="append", index=False)
        print(f"  Inserted {len(bullpen_df)} team-season bullpen records")
    else:
        print("  No bullpen data — table is empty")

    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pitcher_stats WHERE xfip IS NOT NULL")
    xfip_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM team_bullpen_stats")
    bp_count = cursor.fetchone()[0]
    print(f"\n  Verification:")
    print(f"    pitcher_stats rows with xFIP: {xfip_count}")
    print(f"    team_bullpen_stats rows:       {bp_count}")
    print(f"    Total starter rows updated:    {total_updated}")

    conn.close()
    print("\nDone! Now run: python3 MLBModel.py")


if __name__ == "__main__":
    main()
