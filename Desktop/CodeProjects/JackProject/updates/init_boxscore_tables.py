"""
Initialize game_pitcher_lines / game_batter_lines tables in mlb_allseasons.db.
Run this once to create the schema, then it's safe to run repeatedly (no-op if tables exist).
"""
import os
import sqlite3

_UPDATES_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UPDATES_DIR)
DB_PATH = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

_TABLES = {
    "game_pitcher_lines": """
    CREATE TABLE game_pitcher_lines (
        game_id TEXT NOT NULL,
        retro_pitcher_id TEXT,
        pitcher_name TEXT,
        team TEXT NOT NULL,
        game_pk INTEGER,
        gs INTEGER,
        ip REAL,
        pitches INTEGER,
        er INTEGER,
        batters_faced INTEGER,
        PRIMARY KEY (game_id, team, pitcher_name)
    );
    """,
    "game_batter_lines": """
    CREATE TABLE game_batter_lines (
        game_id TEXT NOT NULL,
        retro_batter_id TEXT,
        batter_name TEXT,
        team TEXT NOT NULL,
        game_pk INTEGER,
        pa INTEGER,
        ab INTEGER,
        h INTEGER,
        doubles INTEGER,
        triples INTEGER,
        hr INTEGER,
        bb INTEGER,
        hbp INTEGER,
        sf INTEGER,
        PRIMARY KEY (game_id, team, batter_name)
    );
    """,
}

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_gpl_game ON game_pitcher_lines(game_id);",
    "CREATE INDEX IF NOT EXISTS idx_gbl_game ON game_batter_lines(game_id);",
]


def init_boxscore_tables():
    """Create game_pitcher_lines/game_batter_lines tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for table_name, create_sql in _TABLES.items():
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cur.fetchone():
            print(f"[init] {table_name} table already exists in {DB_PATH}")
            continue
        cur.execute(create_sql)
        print(f"[init] Created {table_name} table in {DB_PATH}")

    for idx_sql in _INDEXES:
        try:
            cur.execute(idx_sql)
        except sqlite3.OperationalError as e:
            print(f"[init] index creation failed ({idx_sql.split()[5]}): {e}", flush=True)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_boxscore_tables()
