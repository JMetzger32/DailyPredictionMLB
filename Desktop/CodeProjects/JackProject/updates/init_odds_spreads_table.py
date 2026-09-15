"""
Initialize the historical run-line (spreads) odds table in mlb_allseasons.db.
Safe to run repeatedly (no-op if the table already exists).

Deliberately a SEPARATE table from odds_snapshots rather than extra columns on it:
the h2h table is load-bearing and exhaustively validated, and the two markets are
fetched by separate API calls (10 credits each) that can succeed independently, so
keeping them apart means a spreads bug can never corrupt moneyline prices that cost
real money and can no longer be re-bought.

Prices are stored ONLY at the standard +/-1.5 run line (see
schedule_fetcher.STANDARD_RUN_LINE): home_covers is defined as margin > 1.5, so any
other point prices a different event. away_point/home_point are stored anyway, as an
audit trail that the invariant held on the data actually ingested.
"""
import os
import sqlite3

_UPDATES_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UPDATES_DIR)
DB_PATH = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

SCHEMA = {
    "odds_snapshots_spreads": """
    CREATE TABLE odds_snapshots_spreads (
        game_date_et   TEXT    NOT NULL,   -- ET date of the game (NOT the UTC date)
        event_id       TEXT    NOT NULL,   -- Odds API event id; unique per game
        requested_date TEXT    NOT NULL,   -- the date we asked for (provenance)
        requested_ts   TEXT    NOT NULL,   -- ISO8601 Z we asked for
        snapshot_ts    TEXT,               -- envelope 'timestamp' actually returned
        commence_time  TEXT,               -- ISO8601 Z first pitch
        horizon_days   INTEGER NOT NULL DEFAULT 0,  -- game_date_et - requested_date
        away_team_raw  TEXT    NOT NULL,   -- EXACT API string, never normalized away
        home_team_raw  TEXT    NOT NULL,
        away_team      TEXT,               -- retro code; NULL means unresolved
        home_team      TEXT,
        away_point     REAL,               -- always +/-1.5; stored for audit
        home_point     REAL,
        away_spread_ml INTEGER,            -- consensus price for away at its point
        home_spread_ml INTEGER,
        away_implied   REAL,               -- de-vigged, sums to 1.0 with home_implied
        home_implied   REAL,
        overround      REAL,
        n_books        INTEGER,
        books_json     TEXT,
        started_before_snapshot INTEGER NOT NULL DEFAULT 0,
        source         TEXT    NOT NULL,   -- historical_api|live_log
        fetched_at     TEXT    NOT NULL,
        PRIMARY KEY (game_date_et, event_id)
    )""",

    # Sibling of odds_game_link, identical shape. A separate table rather than a
    # `market` column on the existing one: that table is working and already
    # populated, and an ALTER + backfill there would risk the h2h joins for no gain.
    "odds_game_link_spreads": """
    CREATE TABLE odds_game_link_spreads (
        game_date_et TEXT NOT NULL,
        event_id     TEXT NOT NULL,
        target       TEXT NOT NULL,        -- 'games' | 'predictions_log'
        game_id      TEXT,                 -- games.game_id (2021-2025)
        game_pk      INTEGER,              -- predictions_log game_pk (2026)
        match_method TEXT NOT NULL,        -- unique_date_teams|dh_by_commence_order|manual
        confidence   TEXT NOT NULL,        -- exact|ambiguous|unmatched
        linked_at    TEXT NOT NULL,
        PRIMARY KEY (game_date_et, event_id, target)
    )""",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_odds_spreads_join "
    "ON odds_snapshots_spreads(game_date_et, away_team, home_team)",
    "CREATE INDEX IF NOT EXISTS idx_odds_spreads_horizon "
    "ON odds_snapshots_spreads(horizon_days)",
    "CREATE INDEX IF NOT EXISTS idx_odds_link_spreads_game "
    "ON odds_game_link_spreads(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_odds_link_spreads_pk "
    "ON odds_game_link_spreads(game_pk)",
]


def init_odds_spreads_table(db_path=DB_PATH, verbose=True):
    """Create the spreads table if absent. Returns the list of tables created."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    created = []
    for name, ddl in SCHEMA.items():
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
        if cur.fetchone():
            if verbose:
                print(f"[init] {name} already exists")
            continue
        cur.execute(ddl)
        created.append(name)
        if verbose:
            print(f"[init] created {name}")
    for idx in INDEXES:
        cur.execute(idx)
    conn.commit()
    conn.close()
    return created


if __name__ == "__main__":
    init_odds_spreads_table()
