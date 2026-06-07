import pandas as pd
import sqlite3
import unicodedata
import re

# BR team code -> Retrosheet team code mapping
BR_TO_RETRO_TEAM = {
    "ANA": "ANA", "LAA": "ANA",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN", "Cubs": "CHN",
    "CHW": "CHA", "CWS": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KCR": "KCA", "KAN": "KCA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "OAK": "ATH", "ATH": "ATH",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDP": "SDN", "SAN": "SDN",
    "SEA": "SEA",
    "SFG": "SFN",
    "STL": "SLN",
    "TBR": "TBA", "TAM": "TBA", "TBD": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSN": "WAS", "WSH": "WAS",
}


def normalize_name(name):
    """Normalize a player name for matching."""
    if not isinstance(name, str):
        return ""
    # Remove asterisks (lefty indicator) and # (switch hitter)
    name = name.replace("*", "").replace("#", "").strip()
    # Normalize unicode (accented characters -> base form)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Lowercase
    name = name.lower().strip()
    # Remove Jr., Sr., II, III etc.
    name = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", "", name)
    # Remove periods
    name = name.replace(".", "")
    return name


def read_pitching_file(filepath, season):
    """Read a pitching stats file (CSV or XLS) and return a clean DataFrame."""
    if filepath.endswith(".csv"):
        # The 2024 CSV has a weird duplicated first column — skip it
        df = pd.read_csv(filepath)
        # Drop the first combined column if it exists
        first_col = df.columns[0]
        if "," in str(first_col):
            df = df.drop(columns=[first_col])
    else:
        # XLS files from BR are HTML tables
        dfs = pd.read_html(filepath)
        df = dfs[0]

    # Filter to starters only (GS > 0)
    df = df[df["GS"] > 0].copy()

    # Skip "2TM", "3TM" etc. (aggregate rows for traded players)
    df = df[~df["Team"].str.contains("TM", na=False)].copy()

    # Clean player names
    df["name_clean"] = df["Player"].apply(normalize_name)

    # Map team codes
    df["retro_team"] = df["Team"].map(BR_TO_RETRO_TEAM)

    # Add season
    df["season"] = season

    # Keep only relevant columns
    cols = ["name_clean", "Player", "retro_team", "season", "Age",
            "W", "L", "ERA", "G", "GS", "IP", "H", "R", "ER",
            "HR", "BB", "SO", "WHIP", "FIP", "ERA+", "WAR",
            "H9", "HR9", "BB9", "SO9"]

    available = [c for c in cols if c in df.columns]
    df = df[available].copy()

    # Convert numeric columns
    for col in ["ERA", "WHIP", "FIP", "ERA+", "WAR", "H9", "HR9", "BB9", "SO9"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def match_to_retrosheet(pitching_df, conn):
    """Match BR pitcher names to Retrosheet pitcher IDs."""
    # Get all Retrosheet pitchers
    retro = pd.read_sql_query("""
        SELECT DISTINCT home_starting_pitcher_id as pid,
               home_starting_pitcher_name as name,
               home_team as team, season
        FROM games WHERE season >= 2021
        UNION
        SELECT DISTINCT visitor_starting_pitcher_id,
               visitor_starting_pitcher_name,
               visiting_team, season
        FROM games WHERE season >= 2021
    """, conn)
    retro["name_clean"] = retro["name"].apply(normalize_name)

    # Match by name + team + season
    merged = pitching_df.merge(
        retro,
        left_on=["name_clean", "retro_team", "season"],
        right_on=["name_clean", "team", "season"],
        how="left"
    )

    # For unmatched, try name + season only (handles team mismatches)
    unmatched = merged[merged["pid"].isna()]
    if len(unmatched) > 0:
        name_only = pitching_df[pitching_df.index.isin(unmatched.index)].merge(
            retro.drop_duplicates(subset=["name_clean", "season"]),
            on=["name_clean", "season"],
            how="left",
            suffixes=("", "_retro")
        )
        # Fill in the missing PIDs
        for idx in unmatched.index:
            match = name_only[name_only.index == idx]
            if len(match) > 0 and pd.notna(match.iloc[0].get("pid")):
                merged.loc[idx, "pid"] = match.iloc[0]["pid"]

    return merged


def create_pitching_table(conn):
    """Create the pitcher_stats table."""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS pitcher_stats")
    cursor.execute("""
        CREATE TABLE pitcher_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            retro_pitcher_id TEXT,
            player_name TEXT,
            retro_team TEXT,
            season INTEGER,
            age INTEGER,
            wins INTEGER,
            losses INTEGER,
            era REAL,
            games INTEGER,
            games_started INTEGER,
            innings_pitched REAL,
            hits_allowed INTEGER,
            runs_allowed INTEGER,
            earned_runs INTEGER,
            homeruns_allowed INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            whip REAL,
            fip REAL,
            era_plus REAL,
            war REAL,
            h9 REAL,
            hr9 REAL,
            bb9 REAL,
            so9 REAL
        )
    """)
    conn.commit()


if __name__ == "__main__":
    files = {
        2021: "2021-Pitching.xls",
        2022: "2022-Pitching.xls",
        2023: "2023-Pitching.xls",
        2024: "2024-Pitching.csv",
        2025: "2025-Pitching.xls",
    }

    # Read all pitching files
    all_pitching = []
    for season, filepath in files.items():
        print(f"Reading {filepath}...")
        df = read_pitching_file(filepath, season)
        print(f"  {len(df)} pitchers with starts")
        all_pitching.append(df)

    pitching_df = pd.concat(all_pitching, ignore_index=True)
    print(f"\nTotal pitcher-seasons: {len(pitching_df)}")

    # Match to Retrosheet IDs
    conn = sqlite3.connect("mlb_allseasons.db")
    print("\nMatching to Retrosheet pitcher IDs...")
    matched = match_to_retrosheet(pitching_df, conn)

    matched_count = matched["pid"].notna().sum()
    total = len(matched)
    print(f"  Matched: {matched_count}/{total} ({matched_count/total:.1%})")

    # Show unmatched
    unmatched = matched[matched["pid"].isna()]
    if len(unmatched) > 0 and len(unmatched) <= 30:
        print(f"\n  Unmatched pitchers:")
        for _, row in unmatched.iterrows():
            print(f"    {row['Player']} ({row['retro_team']}, {row['season']})")

    # Insert into database
    print("\nCreating pitcher_stats table...")
    create_pitching_table(conn)

    cursor = conn.cursor()
    inserted = 0
    for _, row in matched[matched["pid"].notna()].iterrows():
        cursor.execute("""
            INSERT INTO pitcher_stats
            (retro_pitcher_id, player_name, retro_team, season, age,
             wins, losses, era, games, games_started, innings_pitched,
             hits_allowed, runs_allowed, earned_runs, homeruns_allowed,
             walks, strikeouts, whip, fip, era_plus, war, h9, hr9, bb9, so9)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["pid"], row["Player"], row["retro_team"], row["season"],
            row.get("Age"), row.get("W"), row.get("L"),
            row.get("ERA"), row.get("G"), row.get("GS"),
            row.get("IP"), row.get("H"), row.get("R"), row.get("ER"),
            row.get("HR"), row.get("BB"), row.get("SO"),
            row.get("WHIP"), row.get("FIP"), row.get("ERA+"), row.get("WAR"),
            row.get("H9"), row.get("HR9"), row.get("BB9"), row.get("SO9"),
        ))
        inserted += 1

    conn.commit()
    print(f"  Inserted {inserted} pitcher-season records")

    # Verify
    cursor.execute("SELECT season, COUNT(*) FROM pitcher_stats GROUP BY season ORDER BY season")
    print("\n  Pitcher stats per season:")
    for season, count in cursor.fetchall():
        print(f"    {season}: {count} pitchers")

    cursor.execute("""
        SELECT retro_pitcher_id, player_name, season, era, whip, fip, so9
        FROM pitcher_stats
        WHERE games_started >= 20
        ORDER BY era
        LIMIT 10
    """)
    print("\n  Top 10 qualified starters by ERA:")
    print(f"    {'Name':<25s} {'Season':>6} {'ERA':>6} {'WHIP':>6} {'FIP':>6} {'SO9':>6}")
    for row in cursor.fetchall():
        print(f"    {row[1]:<25s} {row[2]:>6} {row[3]:>6.2f} {row[4]:>6.3f} {row[5]:>6.2f} {row[6]:>6.1f}")

    conn.close()
    print(f"\nDone! Pitcher stats added to mlb_allseasons.db")
