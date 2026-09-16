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

# Columns added after the table's original creation. Applied by ALTER TABLE on every
# init so an existing production DB picks them up without a rebuild; SQLite has no
# "ADD COLUMN IF NOT EXISTS", hence the PRAGMA check in _migrate_columns.
#
# home_cover_prob/away_cover_prob/correct_rl were computed in app.py from the start but
# never had anywhere to land, which quietly made /api/betting's rl_stats a no-op on the
# DB path. The spread_* columns and edge_method arrived with the joint edge.
ADDED_COLUMNS = {
    "home_cover_prob":     "REAL",
    "away_cover_prob":     "REAL",
    "away_win_by2_prob":   "REAL",     # P(away wins by 2+); NOT 1 - home_cover_prob
    "correct_rl":          "INTEGER",
    "away_spread_point":   "REAL",
    "home_spread_point":   "REAL",
    "away_spread_ml":      "INTEGER",
    "home_spread_ml":      "INTEGER",
    "away_spread_implied": "REAL",
    "home_spread_implied": "REAL",
    "joint_home_win_prob": "REAL",
    # 'moneyline' | 'joint_ml_rl' -- which formula produced this row's model_edge. The
    # two are on ~6x different scales, so a row's edge is uninterpretable without it.
    "edge_method":         "TEXT",
}


def _migrate_columns(cur):
    """Add any ADDED_COLUMNS the table is missing. Idempotent."""
    existing = {r[1] for r in cur.execute("PRAGMA table_info(betting_log)")}
    added = []
    for col, coltype in ADDED_COLUMNS.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE betting_log ADD COLUMN {col} {coltype}")
            added.append(col)
    if added:
        print(f"[init] betting_log: added {len(added)} column(s): {', '.join(added)}")
    return added


def init_betting_log_table():
    """Create betting_log table if absent, then bring its columns up to date."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check if table already exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='betting_log'")
    if cur.fetchone():
        _migrate_columns(cur)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_betting_edge_method "
                    "ON betting_log(edge_method);")
        conn.commit()
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
    _migrate_columns(cur)

    # Create indexes for fast queries
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_betting_date ON betting_log(date);",
        "CREATE INDEX IF NOT EXISTS idx_betting_correct ON betting_log(correct);",
        "CREATE INDEX IF NOT EXISTS idx_betting_rating ON betting_log(bet_rating);",
        "CREATE INDEX IF NOT EXISTS idx_betting_edge_method ON betting_log(edge_method);",
    ]

    for idx_sql in indexes:
        try:
            cur.execute(idx_sql)
        except sqlite3.OperationalError as e:
            # IF NOT EXISTS makes "already exists" impossible — anything caught here is
            # a real problem (locked DB, corrupt index) and must be visible.
            print(f"[init] index creation failed ({idx_sql.split()[5]}): {e}", flush=True)

    conn.commit()
    conn.close()
    print(f"[init] Created betting_log table in {DB_PATH} with indexes")

if __name__ == "__main__":
    init_betting_log_table()
