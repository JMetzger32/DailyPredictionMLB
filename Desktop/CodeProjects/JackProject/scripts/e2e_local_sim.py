#!/usr/bin/env python3
"""
e2e_local_sim.py — end-to-end pipeline simulation, no server and no network.

Walks one fake game through every stage of the real pipeline using the REAL
code units (extracted from Main/app.py source, same technique as tests/):

  predict -> log entry -> odds capture -> resolution -> betting_log upsert
          -> qualifying filter -> P/L + Kelly -> weekly aggregation -> accuracy

Asserts at each stage; prints PASS/FAIL per stage. Uses a throwaway in-memory
SQLite DB — touches nothing in Databases_and_logs/.

Run: .venv/bin/python scripts/e2e_local_sim.py
"""
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP_SRC = open(os.path.join(_ROOT, "Main", "app.py")).read()

_STAGES = []


def stage(name):
    def deco(fn):
        _STAGES.append((name, fn))
        return fn
    return deco


def _extract(*func_names):
    """Exec selected top-level function defs from app.py source (real code)."""
    ns = {"os": os, "json": json, "datetime": datetime,
          "_ET": ZoneInfo("America/New_York"),
          "_artifacts": {"model_version": "e2e-sim"}}
    for name in func_names:
        m = re.search(rf"\ndef {name}\(.*?\n(?=\ndef |\nclass |\n# |\n@app)", _APP_SRC, re.S)
        assert m, f"could not extract {name} from Main/app.py"
        exec(m.group(0), ns)
    return ns


NS = _extract("_calibration_bucket", "_compute_error_metrics", "_build_prediction_entry",
              "_compute_odds_fields", "_pl_for_bet", "_qualifying_bets",
              "_kelly_stake", "_bet_row", "_week_key", "_synth_results_from_log")

GAME = {
    "game_pk": 999001, "game_type": "R",
    "away_team": "NYA", "away_team_name": "Yankees",
    "home_team": "BOS", "home_team_name": "Red Sox",
    "game_time_utc": "2026-07-01T23:10:00Z",
}
RESULT = {"predicted_winner": "Home", "home_win_prob": 0.62, "away_win_prob": 0.38,
          "predicted_total": 8.5}
ODDS_MAP = {("NYA", "BOS"): {"away_ml": 120, "home_ml": -140,
                             "away_implied": 0.44, "home_implied": 0.56}}

CTX = {}  # carries state between stages


@stage("1. prediction entry built from game + model result")
def s1():
    entry = NS["_build_prediction_entry"](GAME, RESULT)
    assert entry["game_pk"] == 999001 and entry["predicted_winner"] == "Home"
    assert entry["date"] == "2026-07-01" and entry["home_win_prob"] == 0.62
    assert entry["correct"] is None and entry["bet_rating"] is None
    CTX["entry"] = entry


@stage("2. odds captured pre-game (bet_rating assigned)")
def s2():
    fields = NS["_compute_odds_fields"]("NYA", "BOS", RESULT, ODDS_MAP)
    assert fields["bet_rating"] == "good", f"edge 0.06 must rate 'good', got {fields['bet_rating']}"
    assert fields["predicted_team_ml"] == -140
    CTX["entry"].update(fields)


@stage("3. game resolves (result scored against stored pick)")
def s3():
    e = CTX["entry"]
    e["away_score"], e["home_score"], e["actual_winner"] = 3, 5, "Home"
    e["correct"] = (e["predicted_winner"] == e["actual_winner"])
    brier, ll = NS["_compute_error_metrics"](e["home_win_prob"], e["actual_winner"])
    assert e["correct"] is True and 0 < brier < 0.25
    e["brier_score"], e["log_loss"] = brier, ll


@stage("4. betting_log upsert (real SQL, throwaway DB) + NULL re-upsert survival")
def s4():
    m = re.search(r'cur\.execute\("""\s*(INSERT INTO betting_log.*?)"""', _APP_SRC, re.S)
    init = re.search(r"CREATE TABLE betting_log.*?\);",
                     open(os.path.join(_ROOT, "updates", "init_betting_log.py")).read(), re.S)
    assert m and init
    conn = sqlite3.connect(":memory:")
    conn.execute(init.group(0))
    cols = ("game_pk", "date", "game_type", "away_team", "home_team", "predicted_winner",
            "away_win_prob", "home_win_prob", "away_ml", "home_ml", "away_implied",
            "home_implied", "bet_rating", "model_edge", "predicted_team_ml", "predicted_total",
            "actual_winner", "away_score", "home_score", "correct", "closing_away_ml",
            "closing_home_ml", "clv")
    e = CTX["entry"]
    vals = tuple(int(e[c]) if c == "correct" and e.get(c) is not None else e.get(c) for c in cols)
    conn.execute(m.group(1), vals)
    # simulate the restart-then-re-upsert-with-NULLs pattern that used to wipe odds
    conn.execute(m.group(1), tuple(e.get(c) if c in ("game_pk", "date", "away_team",
                 "home_team", "predicted_winner", "home_win_prob", "away_win_prob",
                 "game_type") else None for c in cols))
    conn.commit()
    row = dict(zip(("bet_rating", "correct", "predicted_team_ml"), conn.execute(
        "SELECT bet_rating, correct, predicted_team_ml FROM betting_log WHERE game_pk=999001").fetchone()))
    assert row["bet_rating"] == "good" and row["correct"] == 1 and row["predicted_team_ml"] == -140, \
        f"odds/result must survive NULL re-upsert, got {row}"
    CTX["db_row"] = {**e}


@stage("5. qualifying filter admits the row (needs BOTH odds and result)")
def s5():
    rows = NS["_qualifying_bets"]([CTX["db_row"],
                                   {"bet_rating": None, "correct": 1, "game_type": "R"},
                                   {"bet_rating": "good", "correct": None, "game_type": "R"}])
    assert len(rows) == 1 and rows[0]["game_pk"] == 999001


@stage("6. P/L + quarter-Kelly stake computed")
def s6():
    row = NS["_bet_row"](CTX["db_row"], kelly=(100.0, 0.25, 0.05))
    assert row["pl"] == 7.14, f"-140 winner pays $7.14 on $10, got {row['pl']}"
    assert abs(row["kelly_stake"] - 2.20) < 0.01 and row["kelly_pl"] == 1.57
    CTX["bet_row"] = row


@stage("7. weekly aggregation buckets the bet")
def s7():
    wk = NS["_week_key"](CTX["db_row"]["date"])
    assert wk == "2026-W27", f"2026-07-01 is ISO week 27, got {wk}"


@stage("8. accuracy pipeline scores the resolved entry")
def s8():
    entries = [CTX["db_row"]]
    resolved = [e for e in entries if e.get("correct") is not None and e.get("game_type") != "S"]
    assert len(resolved) == 1 and sum(e["correct"] for e in resolved) / len(resolved) == 1.0


@stage("9. past-date fast path synthesizes this game from the log")
def s9():
    out = NS["_synth_results_from_log"]([{"game": GAME}], {999001: CTX["db_row"]})
    assert out is not None and out[0]["predicted_winner"] == "Home"
    assert out[0]["home_win_prob"] == 0.62 and out[0]["predicted_total"] == 8.5


if __name__ == "__main__":
    failed = 0
    for name, fn in _STAGES:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(_STAGES) - failed}/{len(_STAGES)} stages passed")
    sys.exit(1 if failed else 0)
