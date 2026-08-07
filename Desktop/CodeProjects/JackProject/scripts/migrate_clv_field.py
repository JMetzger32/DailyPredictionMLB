"""
migrate_clv_field.py
---------------------
One-time backfill for the CLV field bug documented in
scripts/results/clv_and_home_skew.md: Main/app.py used to store
`clv = model_prob - closing_implied` (the model's own edge re-measured against a
later line), which is not closing line value. The fix (this branch) computes real
CLV = closing_implied - bet_implied going forward.

This migrates every already-resolved entry that has closing odds attached:
  - moves the old (mislabeled) value into `edge_vs_close`, unchanged
  - computes and stores the correct `clv` from data already on the entry
    (away_ml/home_ml/away_implied/home_implied, closing_away_ml/closing_home_ml)

No refetching, no data loss — pure recomputation from already-stored fields. Touches
Databases_and_logs/predictions_log.json and betting_log.json. The local SQLite
betting_log table holds zero rows in the affected date range (verified separately)
so it is not touched.

Usage:
    python3 scripts/migrate_clv_field.py [--dry-run]
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
BET_LOG  = os.path.join(_ROOT, "Databases_and_logs", "betting_log.json")


def _raw(ml):
    return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)


def implied_probs(away_ml, home_ml):
    if away_ml is None or home_ml is None:
        return None, None
    a, h = _raw(away_ml), _raw(home_ml)
    t = a + h
    return round(a / t, 4), round(h / t, 4)


def migrate_entry(e):
    """Mutate one entry in place. Returns True if it changed."""
    if e.get("clv") is None or e.get("closing_away_ml") is None:
        return False
    if "edge_vs_close" in e:
        return False  # already migrated

    predicted = e.get("predicted_winner")
    closing_impl = (e.get("closing_home_implied") if predicted == "Home"
                    else e.get("closing_away_implied"))
    if closing_impl is None:
        return False

    # 1) preserve the old (mislabeled) quantity under its honest name
    e["edge_vs_close"] = e["clv"]

    # 2) compute true CLV from data already on the entry
    bet_away_impl = e.get("away_implied")
    bet_home_impl = e.get("home_implied")
    if bet_away_impl is None or bet_home_impl is None:
        bet_away_impl, bet_home_impl = implied_probs(e.get("away_ml"), e.get("home_ml"))
    bet_impl = bet_home_impl if predicted == "Home" else bet_away_impl
    if bet_impl is None:
        del e["edge_vs_close"]  # can't compute true clv; don't half-migrate
        return False

    e["clv"] = round(closing_impl - bet_impl, 4)
    return True


def migrate_log(path, dry_run):
    with open(path) as f:
        log = json.load(f)
    changed = 0
    for day in log.values():
        for e in day:
            if migrate_entry(e):
                changed += 1
    if changed and not dry_run:
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for path in (PRED_LOG, BET_LOG):
        n = migrate_log(path, args.dry_run)
        verb = "would migrate" if args.dry_run else "migrated"
        print(f"{os.path.basename(path)}: {verb} {n} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
