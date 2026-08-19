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
    ns = _extract("_rate_edge", "_implied_probs", "_compute_odds_fields")
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
    # away_ml/home_ml present but away_implied/home_implied missing (the
    # _build_prediction_entry storage bug, fixed 2026-08-13) -> must reconstruct
    # implied probs from the moneylines rather than leaving bet_rating stuck None
    odds_no_impl = {("NYA", "BOS"): {"away_ml": 120, "home_ml": -140}}
    r4 = f("NYA", "BOS", pred, odds_no_impl)
    assert r4["bet_rating"] == "good"
    assert abs(r4["model_edge"] - 0.058) < 1e-9   # de-vigged from 120/-140, not the odds dict's rounded 0.56


def test_rate_edge():
    """Single source of truth for bet_rating thresholds — must match
    _compute_odds_fields's persisted values AND be safe to call at READ time on
    old model_edge values from before the 'extreme' tier existed (2026-07-23),
    since betting_stats/betting_weekly re-derive the category this way instead of
    trusting the frozen bet_rating column. See CLAUDE.md's frozen-vintage note."""
    ns = _extract("_rate_edge")
    f = ns["_rate_edge"]
    assert f(None) is None
    assert f(0.13) == "extreme"      # > 0.12
    assert f(0.12) == "good"         # boundary is exclusive on the extreme side
    assert f(0.06) == "good"         # 0.05 < edge <= 0.12
    assert f(0.05) == "unsure"       # boundary is exclusive on the good side
    assert f(0.0) == "unsure"
    assert f(-0.05) == "unsure"      # boundary is exclusive on the bad side
    assert f(-0.06) == "bad"
    # a pre-07-23 row stored bet_rating='good' despite edge=0.20 (extreme didn't exist
    # yet) -- re-rating it under today's rules must reclassify it, that's the whole point
    assert f(0.20) == "extreme"


def test_implied_probs():
    ns = _extract("_implied_probs")
    f = ns["_implied_probs"]
    # de-vigged pair must sum to exactly 1.0
    a, h = f(120, -140)
    assert abs((a + h) - 1.0) < 1e-9
    assert a < h                    # -140 (home favorite) implies the higher probability
    assert f(None, -140) == (None, None)
    assert f(120, None) == (None, None)
    # even-money both sides -> 50/50 regardless of vig
    a2, h2 = f(100, 100)
    assert abs(a2 - 0.5) < 1e-9 and abs(h2 - 0.5) < 1e-9


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


def test_odds_team_resolution():
    """The Odds API team name -> retro code. Regression test for the bug where
    "Athletics" (the club's CURRENT name, no city) was absent from the map, so
    resolution returned None and get_mlb_odds silently `continue`d past every A's
    game -- 126 game-appearances in predictions_log with 0 odds, vs 17-22% for
    every other team. "Cleveland Indians" is the pre-2022 name, needed for 2021."""
    import schedule_fetcher as sf
    r = sf.resolve_odds_team
    assert r("Athletics")            == "ATH"   # the live bug
    assert r("Oakland Athletics")    == "ATH"
    assert r("Sacramento Athletics") == "ATH"
    assert r("Cleveland Indians")    == "CLE"   # pre-2022, needed for the 2021 backfill
    assert r("Cleveland Guardians")  == "CLE"
    assert r("Chicago White Sox")    == "CHA"   # two-word nickname
    assert r("New York Yankees")     == "NYA"
    assert r("New York Mets")        == "NYN"   # same city, must not collide
    # unknown names resolve to None AND get recorded, never silently dropped
    sf.UNMAPPED_ODDS_TEAMS.pop("Utica Pierogies", None)
    assert r("Utica Pierogies") is None
    assert sf.UNMAPPED_ODDS_TEAMS.get("Utica Pierogies") == 1
    sf.UNMAPPED_ODDS_TEAMS.pop("Utica Pierogies", None)
    assert r(None) is None
    # a future relocation degrades to the nickname fallback instead of dropping games
    assert r("Las Vegas Athletics")  == "ATH"


def _odds_event(eid, away, home, commence, prices=((120, -140), (125, -145))):
    """Minimal Odds-API event fixture with two bookmakers."""
    return {
        "id": eid, "commence_time": commence, "away_team": away, "home_team": home,
        "bookmakers": [
            {"title": f"Book{i}", "markets": [{"key": "h2h", "outcomes": [
                {"name": away, "price": a}, {"name": home, "price": h}]}]}
            for i, (a, h) in enumerate(prices)
        ],
    }


def test_parse_odds_events_envelope():
    """THE regression test for the live/historical split: the same events parsed as a
    bare list (live endpoint) and unwrapped from {"data": [...]} (historical endpoint)
    must produce identical prices. If these ever diverge, backfilled odds stop being
    comparable to live-captured odds and the whole edge calibration is invalid."""
    import schedule_fetcher as sf
    events = [_odds_event("e1", "New York Yankees", "Boston Red Sox",
                          "2026-06-16T00:10:00Z")]
    live = sf.parse_odds_events(events)                       # live: bare list
    hist = sf.parse_odds_events({"data": events}["data"])     # historical: under "data"
    assert live == hist
    assert len(live) == 1
    row = live[0]
    assert row["away_team"] == "NYA" and row["home_team"] == "BOS"
    # mean of the two books, via round() — note Python's banker's rounding takes
    # 122.5 -> 122 and -142.5 -> -142, which is pre-existing get_mlb_odds behaviour
    assert row["away_ml"] == 122 and row["home_ml"] == -142
    # de-vigged implieds sum to exactly 1.0
    assert abs(row["away_implied"] + row["home_implied"] - 1.0) < 1e-9
    assert row["n_books"] == 2
    assert row["overround"] > 0                               # vig is positive
    # the legacy map shape is unchanged
    m = sf.odds_map_from_event_rows(live)
    assert m[("NYA", "BOS")]["away_ml"] == 122


def test_game_date_et():
    """commence_time is UTC but MLB game dates are ET. A 8:10 PM ET first pitch is
    already the NEXT day in UTC — roughly half the slate — so parsing the UTC date
    prefix would misfile those games by one day."""
    import schedule_fetcher as sf
    f = sf.commence_time_to_et_date
    assert f("2026-06-16T00:10:00Z") == "2026-06-15"   # 8:10 PM ET, next UTC day
    assert f("2026-06-15T17:05:00Z") == "2026-06-15"   # 1:05 PM ET, same UTC day
    assert f("2026-06-16T02:40:00Z") == "2026-06-15"   # 10:40 PM ET west coast
    assert f(None) is None and f("garbage") is None


def test_parse_odds_events_doubleheader():
    """Two games, same teams, same day, different event ids and start times. The
    per-event rows keep both; the legacy (away, home) map can only hold one. That
    collision is why the historical store keys on event_id instead."""
    import schedule_fetcher as sf
    evs = [_odds_event("g1", "New York Yankees", "Boston Red Sox", "2026-06-15T17:05:00Z"),
           _odds_event("g2", "New York Yankees", "Boston Red Sox", "2026-06-15T23:05:00Z",
                       prices=((150, -170), (155, -175)))]
    rows = sf.parse_odds_events(evs)
    assert len(rows) == 2
    assert {r["event_id"] for r in rows} == {"g1", "g2"}
    assert rows[0]["away_ml"] != rows[1]["away_ml"]        # genuinely different prices
    assert all(r["game_date_et"] == "2026-06-15" for r in rows)
    assert len(sf.odds_map_from_event_rows(rows)) == 1     # documented collapse


def test_parse_odds_events_unresolved_team_kept():
    """An unresolvable team yields a row (so it can be audited) but is dropped from
    the legacy map — the old code dropped it with no trace at all."""
    import schedule_fetcher as sf
    sf.UNMAPPED_ODDS_TEAMS.pop("Utica Pierogies", None)
    rows = sf.parse_odds_events([
        _odds_event("x1", "Utica Pierogies", "Boston Red Sox", "2026-06-15T17:05:00Z")])
    assert len(rows) == 1 and rows[0]["away_team"] is None
    assert rows[0]["away_team_raw"] == "Utica Pierogies"    # raw name preserved
    assert sf.odds_map_from_event_rows(rows) == {}
    assert "Utica Pierogies" in sf.UNMAPPED_ODDS_TEAMS
    sf.UNMAPPED_ODDS_TEAMS.pop("Utica Pierogies", None)


def test_budget_guard():
    """The historical backfill must never spend below the reserve kept for live games.

    Evaluated against the SERVER's reported remaining before every call, because the
    Render app shares the same key and a locally-tracked counter would drift."""
    sys.path.insert(0, os.path.join(_ROOT, "updates"))
    from backfill_historical_odds import would_breach_floor as g, HISTORICAL_COST
    assert HISTORICAL_COST == 10          # historical = 10x the live endpoint
    assert g(5011, 5000) is False         # 5011-10 = 5001, still above the floor
    assert g(5010, 5000) is False         # exactly on the floor is allowed
    assert g(5009, 5000) is True          # 5009-10 = 4999 -> would breach
    assert g(5000, 5000) is True
    assert g(0, 5000) is True
    assert g(None, 5000) is True          # unknown remaining is treated as unsafe
    assert g(19884, 5000) is False        # the real pre-flight state


def test_parse_odds_events_pickem_straddle():
    """Books straddling the +/-100 boundary on a pick'em game must still price.

    Regression for a bug that silently dropped near-even games from BOTH the live
    product and the historical backfill: American odds are discontinuous across
    +/-100, so averaging +104 and -101 (both ~50%) numerically gives about -28 --
    not a valid line -- which then tripped the malformed-odds guard and discarded
    the game entirely. Observed on 1 of 9 games in the 2026-08-13 probe. Averaging
    implied probabilities instead is continuous and yields ~-101.
    """
    import schedule_fetcher as sf
    rows = sf.parse_odds_events([_odds_event(
        "pk1", "Cleveland Guardians", "Detroit Tigers", "2026-08-13T17:11:00Z",
        prices=((-114, 104), (-109, -101), (-110, -102)))])
    assert len(rows) == 1, "pick'em game must not be dropped"
    r = rows[0]
    assert r["home_team"] == "DET"
    # home side is a true coin flip; the naive numeric average would be ~+0
    assert abs(r["home_implied"] - 0.5) < 0.03
    assert abs(r["home_ml"]) >= 100, "must emit a valid American line"
    assert abs(r["away_implied"] + r["home_implied"] - 1.0) < 1e-9


def test_parse_odds_events_malformed_skipped():
    """|ml| < 100 is not a valid American line; such events are skipped entirely,
    matching long-standing get_mlb_odds behaviour."""
    import schedule_fetcher as sf
    rows = sf.parse_odds_events([
        _odds_event("m1", "New York Yankees", "Boston Red Sox",
                    "2026-06-15T17:05:00Z", prices=((50, -60),))])
    assert rows == []


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


def test_pl_for_bet():
    ns = _extract("_pl_for_bet")
    f = ns["_pl_for_bet"]
    assert f({"predicted_team_ml": None, "correct": 1}) is None      # no odds -> no P/L
    assert f({"predicted_team_ml": 120, "correct": 1}) == 12.0       # +120 win pays $12
    assert f({"predicted_team_ml": -150, "correct": 1}) == 6.67      # -150 win pays $6.67
    assert f({"predicted_team_ml": -150, "correct": 0}) == -10.0     # loss = -stake
    assert f({"predicted_team_ml": 120, "correct": 0, }, stake=25) == -25.0


def test_qualifying_bets():
    """Regression for the empty-betting-page failure: rows where bet_rating and correct
    never coexist must yield zero qualifying bets; rows with both must qualify."""
    ns = _extract("_qualifying_bets")
    f = ns["_qualifying_bets"]
    broken_state = (
        [{"bet_rating": None, "correct": 1, "game_type": "R"}] * 3 +   # resolved, no odds
        [{"bet_rating": "good", "correct": None, "game_type": "R"}] * 2  # odds, unresolved
    )
    assert f(broken_state) == [], "odds-XOR-resolved rows must not qualify (the live bug)"
    healthy = {"bet_rating": "good", "correct": 1, "game_type": "R"}
    spring  = {"bet_rating": "good", "correct": 1, "game_type": "S"}
    assert f(broken_state + [healthy, spring]) == [healthy], "RS rows with both must qualify; ST excluded"


def test_kelly_stake():
    ns = _extract("_kelly_stake")
    f = ns["_kelly_stake"]
    assert f(None, -140, 100, 0.25, 0.05) is None       # missing prob
    assert f(0.62, None, 100, 0.25, 0.05) is None       # missing odds
    # -140, p=0.62: b=100/140, f*=(0.62b-0.38)/b=0.088 -> quarter=0.022 -> $2.20
    assert abs(f(0.62, -140, 100, 0.25, 0.05) - 2.20) < 0.01
    assert f(0.40, 120, 100, 0.25, 0.05) == 0.0         # negative edge -> no bet
    assert f(0.90, 200, 100, 0.25, 0.05) == 5.0         # f*=0.85 -> capped at 5% of bankroll
    assert abs(f(0.62, -140, 1000, 0.25, 0.05) - 22.0) < 0.1  # scales with bankroll


def test_week_key():
    import datetime as _dt
    ns = _extract("_week_key")
    ns["datetime"] = _dt.datetime
    f = ns["_week_key"]
    assert f("2024-12-30") == "2025-W01"    # ISO year rollover
    assert f("2025-01-08") == "2025-W02"    # zero-padded week number
    assert f("garbage") is None
    assert f(None) is None


def test_bet_row_kelly():
    ns = _extract("_pl_for_bet", "_kelly_stake", "_bet_row")
    f = ns["_bet_row"]
    b = {"predicted_winner": "Home", "home_win_prob": 0.62, "predicted_team_ml": -140,
         "correct": 1, "date": "2026-07-06", "game_pk": 1, "model_edge": 0.06}
    row = f(b, kelly=(100, 0.25, 0.05))
    assert abs(row["kelly_stake"] - 2.20) < 0.01
    assert row["kelly_pl"] == round(2.20 * 100 / 140, 2)   # win at -140
    assert f(b).get("kelly_stake") is None                  # no kelly ctx -> no kelly fields
    b_loss = dict(b, correct=0)
    assert f(b_loss, kelly=(100, 0.25, 0.05))["kelly_pl"] == -2.20
    b_no_odds = dict(b, predicted_team_ml=None)
    r = f(b_no_odds, kelly=(100, 0.25, 0.05))
    assert r["kelly_stake"] is None and r["kelly_pl"] is None


def test_synth_results_from_log():
    """Past-date fast path: full synthesis when every game has a stored prediction;
    None (-> fall back to inference) when any game lacks one."""
    ns = _extract("_synth_results_from_log")
    f = ns["_synth_results_from_log"]
    ctx = [{"game": {"game_pk": 1}}, {"game": {"game_pk": 2}}]
    log = {1: {"home_win_prob": 0.62, "away_win_prob": 0.38, "predicted_winner": "Home",
               "predicted_total": 8.5},
           2: {"home_win_prob": 0.45, "predicted_winner": "Away"}}
    out = f(ctx, log)
    assert set(out) == {0, 1}
    assert out[0]["predicted_winner"] == "Home" and out[0]["home_win_prob"] == 0.62
    assert out[0]["confidence"] == 0.24 and out[0]["predicted_total"] == 8.5
    assert out[1]["away_win_prob"] == 0.55            # derived from 1 - hwp
    assert out[1]["est_components"] is None           # not logged -> None, UI hides it
    assert f(ctx, {1: log[1]}) is None                # game 2 unstored -> no fast path
    assert f(ctx, {1: log[1], 2: {"home_win_prob": None, "predicted_winner": "Away"}}) is None
    assert f([], {}) == {}                            # empty slate -> empty dict


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
