"""
Initialize betting_log table in mlb_allseasons.db.
Run this once to create the schema, then it's safe to run repeatedly (no-op if table exists).
"""
import os
import sqlite3
from pathlib import Path

_UPDATES_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UPDATES_DIR)
DB_PATH = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

def init_betting_log_table():
    """Create betting_log table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check if table already exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='betting_log'")
    if cur.fetchone():
        print(f"[init] betting_log table already exists in {DB_PATH}")
        conn.close()
        return

    # Create the betting_log table
    create_table_sql = """
    CREATE TABLE betting_log (
        game_pk INTEGER PRIMARY KEY,
        date TEXT NOT NULL,
        game_type TEXT,
        away_team TEXT,
        home_team TEXT,
        predicted_winner TEXT,
        away_win_prob REAL,
        home_win_prob REAL,
        away_ml INTEGER,
        home_ml INTEGER,
        away_implied REAL,
        home_implied REAL,
        bet_rating TEXT,
        model_edge REAL,
        predicted_team_ml INTEGER,
        predicted_total REAL,
        actual_winner TEXT,
        away_score INTEGER,
        home_score INTEGER,
        correct INTEGER,
        closing_away_ml INTEGER,
        closing_home_ml INTEGER,
        clv REAL,
        created_at TEXT,
        updated_at TEXT
    );
    """

    cur.execute(create_table_sql)

    # Create indexes for fast queries
    indexes = [
        "CREATE INDEX idx_betting_date ON betting_log(date);",
        "CREATE INDEX idx_betting_correct ON betting_log(correct);",
        "CREATE INDEX idx_betting_rating ON betting_log(bet_rating);",
    ]

    for idx_sql in indexes:
        try:
            cur.execute(idx_sql)
        except sqlite3.OperationalError:
            pass  # Index might already exist

    conn.commit()
    conn.close()
    print(f"[init] Created betting_log table in {DB_PATH} with indexes")

if __name__ == "__main__":
    init_betting_log_table()
