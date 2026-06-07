import csv
import sqlite3

# Retrosheet game log column definitions
COLUMNS = [
    ("date", "TEXT"),
    ("doubleheader", "TEXT"),
    ("day_of_week", "TEXT"),
    ("visiting_team", "TEXT"),
    ("visiting_league", "TEXT"),
    ("visiting_game_number", "INTEGER"),
    ("home_team", "TEXT"),
    ("home_league", "TEXT"),
    ("home_game_number", "INTEGER"),
    ("visitor_score", "INTEGER"),
    ("home_score", "INTEGER"),
    ("length_outs", "INTEGER"),
    ("day_night", "TEXT"),
    ("completion_info", "TEXT"),
    ("forfeit_info", "TEXT"),
    ("protest_info", "TEXT"),
    ("park_id", "TEXT"),
    ("attendance", "INTEGER"),
    ("time_of_game_minutes", "INTEGER"),
    ("visitor_line_score", "TEXT"),
    ("home_line_score", "TEXT"),
    # Visitor batting stats
    ("visitor_at_bats", "INTEGER"),
    ("visitor_hits", "INTEGER"),
    ("visitor_doubles", "INTEGER"),
    ("visitor_triples", "INTEGER"),
    ("visitor_homeruns", "INTEGER"),
    ("visitor_rbi", "INTEGER"),
    ("visitor_sac_hits", "INTEGER"),
    ("visitor_sac_flies", "INTEGER"),
    ("visitor_hit_by_pitch", "INTEGER"),
    ("visitor_walks", "INTEGER"),
    ("visitor_intentional_walks", "INTEGER"),
    ("visitor_strikeouts", "INTEGER"),
    ("visitor_stolen_bases", "INTEGER"),
    ("visitor_caught_stealing", "INTEGER"),
    ("visitor_grounded_into_dp", "INTEGER"),
    ("visitor_awarded_first_on_ci", "INTEGER"),
    ("visitor_left_on_base", "INTEGER"),
    # Visitor pitching stats
    ("visitor_pitchers_used", "INTEGER"),
    ("visitor_individual_earned_runs", "INTEGER"),
    ("visitor_team_earned_runs", "INTEGER"),
    ("visitor_wild_pitches", "INTEGER"),
    ("visitor_balks", "INTEGER"),
    # Visitor fielding stats
    ("visitor_putouts", "INTEGER"),
    ("visitor_assists", "INTEGER"),
    ("visitor_errors", "INTEGER"),
    ("visitor_passed_balls", "INTEGER"),
    ("visitor_double_plays", "INTEGER"),
    ("visitor_triple_plays", "INTEGER"),
    # Home batting stats
    ("home_at_bats", "INTEGER"),
    ("home_hits", "INTEGER"),
    ("home_doubles", "INTEGER"),
    ("home_triples", "INTEGER"),
    ("home_homeruns", "INTEGER"),
    ("home_rbi", "INTEGER"),
    ("home_sac_hits", "INTEGER"),
    ("home_sac_flies", "INTEGER"),
    ("home_hit_by_pitch", "INTEGER"),
    ("home_walks", "INTEGER"),
    ("home_intentional_walks", "INTEGER"),
    ("home_strikeouts", "INTEGER"),
    ("home_stolen_bases", "INTEGER"),
    ("home_caught_stealing", "INTEGER"),
    ("home_grounded_into_dp", "INTEGER"),
    ("home_awarded_first_on_ci", "INTEGER"),
    ("home_left_on_base", "INTEGER"),
    # Home pitching stats
    ("home_pitchers_used", "INTEGER"),
    ("home_individual_earned_runs", "INTEGER"),
    ("home_team_earned_runs", "INTEGER"),
    ("home_wild_pitches", "INTEGER"),
    ("home_balks", "INTEGER"),
    # Home fielding stats
    ("home_putouts", "INTEGER"),
    ("home_assists", "INTEGER"),
    ("home_errors", "INTEGER"),
    ("home_passed_balls", "INTEGER"),
    ("home_double_plays", "INTEGER"),
    ("home_triple_plays", "INTEGER"),
    # Umpires
    ("umpire_home_id", "TEXT"),
    ("umpire_home_name", "TEXT"),
    ("umpire_1b_id", "TEXT"),
    ("umpire_1b_name", "TEXT"),
    ("umpire_2b_id", "TEXT"),
    ("umpire_2b_name", "TEXT"),
    ("umpire_3b_id", "TEXT"),
    ("umpire_3b_name", "TEXT"),
    ("umpire_lf_id", "TEXT"),
    ("umpire_lf_name", "TEXT"),
    ("umpire_rf_id", "TEXT"),
    ("umpire_rf_name", "TEXT"),
    # Managers
    ("visitor_manager_id", "TEXT"),
    ("visitor_manager_name", "TEXT"),
    ("home_manager_id", "TEXT"),
    ("home_manager_name", "TEXT"),
    # Winning/losing pitchers
    ("winning_pitcher_id", "TEXT"),
    ("winning_pitcher_name", "TEXT"),
    ("losing_pitcher_id", "TEXT"),
    ("losing_pitcher_name", "TEXT"),
    # Save pitcher
    ("saving_pitcher_id", "TEXT"),
    ("saving_pitcher_name", "TEXT"),
    # Game-winning RBI batter
    ("gw_rbi_batter_id", "TEXT"),
    ("gw_rbi_batter_name", "TEXT"),
    # Starting pitchers
    ("visitor_starting_pitcher_id", "TEXT"),
    ("visitor_starting_pitcher_name", "TEXT"),
    ("home_starting_pitcher_id", "TEXT"),
    ("home_starting_pitcher_name", "TEXT"),
    # Visitor lineup (9 batters: id, name, defensive position)
    ("visitor_batter_1_id", "TEXT"),
    ("visitor_batter_1_name", "TEXT"),
    ("visitor_batter_1_pos", "INTEGER"),
    ("visitor_batter_2_id", "TEXT"),
    ("visitor_batter_2_name", "TEXT"),
    ("visitor_batter_2_pos", "INTEGER"),
    ("visitor_batter_3_id", "TEXT"),
    ("visitor_batter_3_name", "TEXT"),
    ("visitor_batter_3_pos", "INTEGER"),
    ("visitor_batter_4_id", "TEXT"),
    ("visitor_batter_4_name", "TEXT"),
    ("visitor_batter_4_pos", "INTEGER"),
    ("visitor_batter_5_id", "TEXT"),
    ("visitor_batter_5_name", "TEXT"),
    ("visitor_batter_5_pos", "INTEGER"),
    ("visitor_batter_6_id", "TEXT"),
    ("visitor_batter_6_name", "TEXT"),
    ("visitor_batter_6_pos", "INTEGER"),
    ("visitor_batter_7_id", "TEXT"),
    ("visitor_batter_7_name", "TEXT"),
    ("visitor_batter_7_pos", "INTEGER"),
    ("visitor_batter_8_id", "TEXT"),
    ("visitor_batter_8_name", "TEXT"),
    ("visitor_batter_8_pos", "INTEGER"),
    ("visitor_batter_9_id", "TEXT"),
    ("visitor_batter_9_name", "TEXT"),
    ("visitor_batter_9_pos", "INTEGER"),
    # Home lineup (9 batters: id, name, defensive position)
    ("home_batter_1_id", "TEXT"),
    ("home_batter_1_name", "TEXT"),
    ("home_batter_1_pos", "INTEGER"),
    ("home_batter_2_id", "TEXT"),
    ("home_batter_2_name", "TEXT"),
    ("home_batter_2_pos", "INTEGER"),
    ("home_batter_3_id", "TEXT"),
    ("home_batter_3_name", "TEXT"),
    ("home_batter_3_pos", "INTEGER"),
    ("home_batter_4_id", "TEXT"),
    ("home_batter_4_name", "TEXT"),
    ("home_batter_4_pos", "INTEGER"),
    ("home_batter_5_id", "TEXT"),
    ("home_batter_5_name", "TEXT"),
    ("home_batter_5_pos", "INTEGER"),
    ("home_batter_6_id", "TEXT"),
    ("home_batter_6_name", "TEXT"),
    ("home_batter_6_pos", "INTEGER"),
    ("home_batter_7_id", "TEXT"),
    ("home_batter_7_name", "TEXT"),
    ("home_batter_7_pos", "INTEGER"),
    ("home_batter_8_id", "TEXT"),
    ("home_batter_8_name", "TEXT"),
    ("home_batter_8_pos", "INTEGER"),
    ("home_batter_9_id", "TEXT"),
    ("home_batter_9_name", "TEXT"),
    ("home_batter_9_pos", "INTEGER"),
    # Additional info
    ("additional_info", "TEXT"),
    ("acquisition_info", "TEXT"),
]


def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    col_defs = ", ".join(f"{name} {dtype}" for name, dtype in COLUMNS)
    cursor.execute(f"DROP TABLE IF EXISTS games")
    cursor.execute(f"CREATE TABLE games (game_id INTEGER PRIMARY KEY AUTOINCREMENT, season INTEGER, {col_defs})")

    conn.commit()
    return conn


def parse_and_insert(conn, file_path, season):
    cursor = conn.cursor()
    col_names = [name for name, _ in COLUMNS]
    all_cols = ["season"] + col_names
    placeholders = ", ".join("?" for _ in all_cols)
    insert_sql = f"INSERT INTO games ({', '.join(all_cols)}) VALUES ({placeholders})"

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        rows_inserted = 0
        for row in reader:
            # Pad or trim row to match expected column count
            if len(row) < len(COLUMNS):
                row.extend([""] * (len(COLUMNS) - len(row)))
            elif len(row) > len(COLUMNS):
                row = row[:len(COLUMNS)]

            # Convert integer fields
            values = [season]
            for i, (_, dtype) in enumerate(COLUMNS):
                val = row[i].strip('"').strip()
                if dtype == "INTEGER":
                    try:
                        val = int(val) if val else None
                    except ValueError:
                        val = None
                values.append(val)

            cursor.execute(insert_sql, values)
            rows_inserted += 1

    conn.commit()
    return rows_inserted


def print_summary(conn):
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM games")
    total = cursor.fetchone()[0]
    print(f"\nTotal games loaded: {total}")

    cursor.execute("SELECT MIN(date), MAX(date) FROM games")
    min_date, max_date = cursor.fetchone()
    print(f"Date range: {min_date} to {max_date}")

    print("\nSample games (first 10):")
    print(f"{'Date':<12} {'Visitor':<6} {'Score':>5}  {'Home':<6} {'Score':>5}  {'Park':<8} {'Attendance':>10}")
    print("-" * 65)

    cursor.execute("""
        SELECT date, visiting_team, visitor_score, home_team, home_score, park_id, attendance
        FROM games
        LIMIT 10
    """)
    for row in cursor.fetchall():
        date, vis, vs, home, hs, park, att = row
        att_str = f"{att:,}" if att else "N/A"
        print(f"{date:<12} {vis:<6} {vs:>5}  {home:<6} {hs:>5}  {park:<8} {att_str:>10}")

    print("\nGames per season:")
    cursor.execute("SELECT season, COUNT(*) FROM games GROUP BY season ORDER BY season")
    for season, count in cursor.fetchall():
        print(f"  {season}: {count} games")


if __name__ == "__main__":
    import glob
    import os

    db_path = "mlb_allseasons.db"
    game_log_files = sorted(glob.glob("gl20*.txt"))

    print(f"Found {len(game_log_files)} game log files: {[os.path.basename(f) for f in game_log_files]}")
    conn = create_database(db_path)

    total = 0
    for file_path in game_log_files:
        # Extract season year from filename (e.g. gl2021.txt -> 2021)
        season = int(os.path.basename(file_path)[2:6])
        rows = parse_and_insert(conn, file_path, season)
        total += rows
        print(f"  {file_path}: {rows} games (season {season})")

    print(f"\nTotal: {total} games inserted into {db_path}")
    print_summary(conn)
    conn.close()
    print(f"\nDatabase saved to {db_path}")
