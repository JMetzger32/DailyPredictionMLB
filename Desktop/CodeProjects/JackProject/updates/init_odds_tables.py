"""
Initialize the historical-odds tables in mlb_allseasons.db.
Safe to run repeatedly (no-op if the tables already exist).

Three tables, deliberately kept separate:

  odds_snapshots  the prices themselves. IRREPLACEABLE -- these cost 10 credits per
                  date on a paid plan and cannot be re-bought after a downgrade.
                  Keyed (game_date_et, event_id), NOT (date, away, home): two games
                  of a doubleheader are two events with two commence_times, and a
                  team-tuple key silently collapses them (which is exactly what the
                  live odds_map does today).

  odds_game_link  the join from an odds event to a DB games row / predictions_log
                  entry. A DERIVED artifact, so it lives apart from the prices and
                  can be dropped and rebuilt when the join logic improves or the
                  DB grows past its 2026-07-07 cutoff, with zero risk to the data
                  that actually cost money.

  odds_fetch_log  every fetch attempt including failures -- the ledger used to
                  reconcile credits spent against dates actually retrieved.
"""
import os
import sqlite3

_UPDATES_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UPDATES_DIR)
DB_PATH = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")

SCHEMA = {
    "odds_snapshots": """
    CREATE TABLE odds_snapshots (
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
        away_ml        INTEGER,
        home_ml        INTEGER,
        away_implied   REAL,
        home_implied   REAL,
        overround      REAL,
        n_books        INTEGER,
        books_json     TEXT,
        arbitrage_pct  REAL,
        started_before_snapshot INTEGER NOT NULL DEFAULT 0,
        source         TEXT    NOT NULL,   -- historical_api|closing_archive|live_log
        fetched_at     TEXT    NOT NULL,
        PRIMARY KEY (game_date_et, event_id)
    )""",
    "odds_game_link": """
    CREATE TABLE odds_game_link (
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
    "odds_fetch_log": """
    CREATE TABLE odds_fetch_log (
        requested_date TEXT NOT NULL,
        requested_ts   TEXT NOT NULL,
        attempt        INTEGER NOT NULL,
        http_status    INTEGER,
        snapshot_ts    TEXT,
        n_events       INTEGER,
        n_primary      INTEGER,            -- events with horizon_days = 0
        n_unmapped     INTEGER,
        credits_remaining_before INTEGER,
        credits_remaining_after  INTEGER,
        credits_charged INTEGER,
        error          TEXT,
        fetched_at     TEXT NOT NULL,
        PRIMARY KEY (requested_date, requested_ts, attempt)
    )""",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_odds_snap_join "
    "ON odds_snapshots(game_date_et, away_team, home_team)",
    "CREATE INDEX IF NOT EXISTS idx_odds_snap_horizon "
    "ON odds_snapshots(horizon_days)",
    "CREATE INDEX IF NOT EXISTS idx_odds_link_game "
    "ON odds_game_link(game_id)",
    "CREATE INDEX IF NOT EXISTS idx_odds_link_pk "
    "ON odds_game_link(game_pk)",
]


def init_odds_tables(db_path=DB_PATH, verbose=True):
    """Create the odds tables if absent. Returns the list of tables created."""
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
    init_odds_tables()
