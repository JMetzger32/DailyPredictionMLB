#!/usr/bin/env python3
"""
Unit tests for pure helpers. Plain-assert, runnable with either:
    .venv/bin/python tests/test_units.py        (no pytest needed)
    .venv/bin/python -m pytest tests/           (if pytest is installed)

Importing Main/app.py executes network-heavy startup, so app.py functions are extracted
from SOURCE and exec'd in an isolated namespace — the tests still run the real code.
schedule_fetcher imports cheaply and is imported normally.
"""
import os
import re
import sys
import json
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "updates"))
sys.path.insert(0, os.path.join(_ROOT, "Main"))

_APP_SRC = open(os.path.join(_ROOT, "Main", "app.py")).read()


def _extract(*func_names):
    """Exec selected top-level function defs from app.py source into a namespace."""
    ns = {"os": os, "json": json}
    for name in func_names:
        m = re.search(rf"\ndef {name}\(.*?\n(?=\ndef |\nclass |\n# |\n@app)", _APP_SRC, re.S)
        assert m, f"could not extract {name} from Main/app.py"
        exec(m.group(0), ns)
    return ns


# ---------------------------------------------------------------------------
def test_calibration_bucket():
    ns = _extract("_calibration_bucket")
    f = ns["_calibration_bucket"]
    assert f(0.50) == "50-60%"
    assert f(0.549) == "50-60%"
    assert f(0.62) == "60-70%"
    assert f(0.799) == "70-80%"
    assert f(0.85) == "80-90%"
    assert f(0.999) == "90-100%"
    assert f(1.0) == "90-100%"          # top edge clamps into last bin
    # must match /api/calibration's binning exactly
    for p in (0.5, 0.55, 0.61, 0.7, 0.83, 0.94, 1.0):
        idx = min(int(p * 10), 9)
        assert f(p) == f"{idx*10}-{idx*10+10}%"


def test_compute_odds_fields():
    ns = _extract("_compute_odds_fields")
    f = ns["_compute_odds_fields"]
    pred = {"predicted_winner": "Home", "home_win_prob": 0.62, "away_win_prob": 0.38}
    odds = {("NYA", "BOS"): {"away_ml": 120, "home_ml": -140,
                             "away_implied": 0.44, "home_implied": 0.56}}
    r = f("NYA", "BOS", pred, odds)
    assert r["home_ml"] == -140 and r["away_ml"] == 120
    assert r["predicted_team_ml"] == -140            # follows the predicted side
    assert r["bet_rating"] == "good"                 # edge 0.62-0.56 = +0.06 > 0.05
    assert abs(r["model_edge"] - 0.06) < 1e-9
    # no odds for the matchup -> all None
    r2 = f("SEA", "TEX", pred, odds)
    assert r2["away_ml"] is None and r2["bet_rating"] is None
    # negative edge -> bad
    pred_bad = {"predicted_winner": "Away", "home_win_prob": 0.62, "away_win_prob": 0.38}
    r3 = f("NYA", "BOS", pred_bad, odds)
    assert r3["bet_rating"] == "bad"                 # 0.38-0.44 = -0.06 < -0.05


def test_find_pitcher_by_name():
    from schedule_fetcher import find_pitcher_by_name
    sp = {
        "colej001": {"name": "Gerrit Cole"},
        "degrj001": {"name": "Jacob deGrom"},
        "smitj001": {"name": "Joe Smith"},
        "smitw001": {"name": "Will Smith"},
    }
    assert find_pitcher_by_name("Gerrit Cole", sp) == "colej001"
    assert find_pitcher_by_name("gerrit cole", sp) == "colej001"      # case-insensitive
    assert find_pitcher_by_name("Jacob deGrom", sp) == "degrj001"
    assert find_pitcher_by_name("deGrom", sp) == "degrj001"           # unambiguous last name
    assert find_pitcher_by_name("Smith", sp) is None                  # ambiguous last name
    assert find_pitcher_by_name("Nobody Here", sp) is None
    assert find_pitcher_by_name(None, sp) is None


def test_should_restore():
    ns = _extract("_latest_date_key", "_should_restore")
    should = ns["_should_restore"]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "log.json")
        json.dump({"2026-07-01": [1], "2026-07-05": [2]}, open(logp, "w"))
        # stale-but-larger remote -> skip
        stale = (json.dumps({"2026-06-01": [1]}) + " " * 4096).encode()
        assert should(logp, stale)[0] is False
        # newer-but-smaller remote -> restore
        newer = json.dumps({"2026-07-06": [1]}).encode()
        assert should(logp, newer)[0] is True
        # empty remote -> skip
        assert should(logp, b"")[0] is False
        # pkl present -> never; pkl missing -> restore
        pkl = os.path.join(td, "m.pkl")
        open(pkl, "wb").write(b"x")
        assert should(pkl, b"y" * 999)[0] is False
        assert should(os.path.join(td, "gone.pkl"), b"y")[0] is True


def test_betting_upsert_coalesce():
    """Execute the REAL upsert SQL from app.py against a temp DB: created_at preserved,
    NULL result fields never clobber previously-resolved values, updated_at advances."""
    m = re.search(r'cur\.execute\("""\s*(INSERT INTO betting_log.*?)"""', _APP_SRC, re.S)
    assert m, "could not extract betting upsert SQL from Main/app.py"
    sql = m.group(1)
    init_sql = re.search(r'CREATE TABLE betting_log.*?\);',
                         open(os.path.join(_ROOT, "updates", "init_betting_log.py")).read(),
                         re.S)
    assert init_sql, "could not extract betting_log CREATE TABLE from init_betting_log.py"
    conn = sqlite3.connect(":memory:")
    conn.execute(init_sql.group(0))
    vals = lambda **kw: tuple(kw.get(c) for c in (
        "game_pk", "date", "game_type", "away_team", "home_team", "predicted_winner",
        "away_win_prob", "home_win_prob", "away_ml", "home_ml", "away_implied",
        "home_implied", "bet_rating", "model_edge", "predicted_team_ml", "predicted_total",
        "actual_winner", "away_score", "home_score", "correct", "closing_away_ml",
        "closing_home_ml", "clv"))
    conn.execute(sql, vals(game_pk=1, date="2026-07-01", predicted_winner="Home",
                           home_win_prob=0.6, away_ml=120, bet_rating="good",
                           actual_winner="Home", correct=1, clv=0.03))
    conn.commit()
    row1 = conn.execute("SELECT created_at, updated_at, correct, clv, bet_rating "
                        "FROM betting_log WHERE game_pk=1").fetchone()
    conn.execute("UPDATE betting_log SET updated_at='2000-01-01' WHERE game_pk=1")
    # re-upsert with NULL results/odds
    conn.execute(sql, vals(game_pk=1, date="2026-07-01", predicted_winner="Home",
                           home_win_prob=0.6))
    conn.commit()
    row2 = conn.execute("SELECT created_at, updated_at, correct, clv, bet_rating "
                        "FROM betting_log WHERE game_pk=1").fetchone()
    assert row2[0] == row1[0], "created_at must be preserved"
    assert row2[1] != "2000-01-01", "updated_at must advance on re-upsert"
    assert row2[2] == 1 and row2[3] == 0.03 and row2[4] == "good", \
        "resolved/odds columns must survive a NULL re-upsert"
    conn.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
