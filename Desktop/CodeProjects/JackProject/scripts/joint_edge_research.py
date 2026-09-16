"""
joint_edge_research.py
----------------------
Does combining the moneyline and run-line markets produce an edge signal that beats
the moneyline-only edge the product ships today?

Everything here is out-of-sample: walk-forward OOF model probabilities
(oof_predictions + oof_runline_predictions, season S trained only on seasons < S)
joined to de-vigged market prices archived at a fixed pre-game snapshot. Scoring the
shipped artifact on its own training seasons would measure memorised outcomes, not
skill -- same discipline as build_oof_predictions.py.

The run-line orientation problem, and why it is solved rather than dodged:
the market quotes -1.5 on whichever side is favoured, so `home_implied` on the
spreads table is P(home covers ITS OWN line) -- P(margin > 1.5) when home lays -1.5,
but P(margin > -1.5) when home takes +1.5. Those are different events. The model can
now price both, because train_away_runline.py added the away side:
    home lays -1.5 :  model P = home_cover_prob          (P(margin > 1.5))
    home takes +1.5:  model P = 1 - away_by2_prob        (P(margin > -1.5))
so model and market are compared on the SAME event for every game, with no games
dropped and no silent threshold mismatch.

Baselines this must be judged against (CLAUDE.md, measured on the moneyline model):
model AUC 0.598 vs market AUC 0.594; edge alone is not reliably predictive
(logit(win) ~ edge, p=0.42 at n=222). A new signal that merely matches those is not
an improvement.

Usage:
    .venv/bin/python scripts/joint_edge_research.py [--scheme walkforward]
"""
import argparse
import os
import sqlite3
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
RESULTS = os.path.join(_ROOT, "scripts", "results")


def lg(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, s):
    """Rank-based AUC; no sklearn dependency needed for one number."""
    y = np.asarray(y)
    order = np.argsort(np.asarray(s, dtype=float))
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(scheme):
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT o.game_id, o.season, o.home_win_prob, o.home_win,
               h.away_ml, h.home_ml, h.away_implied, h.home_implied,
               r.home_cover_prob, r.away_by2_prob, r.home_covers, r.away_covers,
               s.home_point, s.away_implied, s.home_implied
        FROM oof_predictions o
        JOIN oof_runline_predictions r
              ON r.game_id = o.game_id AND r.scheme = o.scheme
        JOIN odds_game_link l
              ON l.game_id = o.game_id AND l.target='games' AND l.confidence='exact'
        JOIN odds_snapshots h
              ON h.game_date_et = l.game_date_et AND h.event_id = l.event_id
             AND h.horizon_days = 0
        JOIN odds_game_link_spreads ls
              ON ls.game_id = o.game_id AND ls.target='games' AND ls.confidence='exact'
        JOIN odds_snapshots_spreads s
              ON s.game_date_et = ls.game_date_et AND s.event_id = ls.event_id
             AND s.horizon_days = 0
        WHERE o.scheme = ?
          AND h.home_implied IS NOT NULL AND s.home_implied IS NOT NULL
          AND o.home_win IS NOT NULL
    """, (scheme,)).fetchall()
    conn.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="walkforward",
                    help="walkforward (headline) or loso (optimistic upper bound)")
    args = ap.parse_args()

    rows = load(args.scheme)
    print(f"scheme={args.scheme}  joined games: {len(rows)}")
    if not rows:
        print("no rows -- did build_runline_oof.py and the spreads link run?")
        return 1

    season = np.array([r[1] for r in rows])
    mdl_ml = np.array([r[2] for r in rows], dtype=float)
    home_win = np.array([r[3] for r in rows], dtype=int)
    mkt_ml = np.array([r[7] for r in rows], dtype=float)       # h2h home_implied
    p_home_by2 = np.array([r[8] for r in rows], dtype=float)
    p_away_by2 = np.array([r[9] for r in rows], dtype=float)
    home_point = np.array([r[12] for r in rows], dtype=float)
    mkt_rl = np.array([r[14] for r in rows], dtype=float)      # spreads home_implied

    # Model's probability for the SAME event the spreads market priced for home.
    home_lays = home_point == -1.5
    mdl_rl = np.where(home_lays, p_home_by2, 1.0 - p_away_by2)

    print(f"  home lays -1.5: {home_lays.sum()}   home takes +1.5: {(~home_lays).sum()}")
    print(f"  seasons: {sorted(set(season.tolist()))}")

    # ---- sanity: is each market calibrated on its own event? -------------------
    actual_home_covers_line = np.where(
        home_lays,
        np.array([r[10] for r in rows], dtype=int),                 # margin > 1.5
        1 - np.array([r[11] for r in rows], dtype=int))             # margin > -1.5
    print("\n" + "=" * 72)
    print("1) DOES EACH SIDE PRICE ITS OWN EVENT HONESTLY? (OOF, no leakage)")
    print("=" * 72)
    print(f"  moneyline   model mean {mdl_ml.mean():.4f}  market mean {mkt_ml.mean():.4f}"
          f"  actual {home_win.mean():.4f}")
    print(f"  run line    model mean {mdl_rl.mean():.4f}  market mean {mkt_rl.mean():.4f}"
          f"  actual {actual_home_covers_line.mean():.4f}")
    print(f"  AUC  model_ml={auc(home_win, mdl_ml):.4f}   market_ml={auc(home_win, mkt_ml):.4f}")
    print(f"       model_rl={auc(actual_home_covers_line, mdl_rl):.4f}   "
          f"market_rl={auc(actual_home_covers_line, mkt_rl):.4f}")

    # ---- 2) does the run line add anything to the moneyline picture? -----------
    print("\n" + "=" * 72)
    print("2) JOINT REGRESSION: logit(home_win) ~ ml + rl terms")
    print("=" * 72)
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  statsmodels not installed -- skipping regressions")
        return 1

    X_base = sm.add_constant(np.column_stack([lg(mdl_ml), lg(mkt_ml)]))
    m_base = sm.Logit(home_win, X_base).fit(disp=0)
    X_full = sm.add_constant(np.column_stack([lg(mdl_ml), lg(mkt_ml),
                                              lg(mdl_rl), lg(mkt_rl)]))
    m_full = sm.Logit(home_win, X_full).fit(disp=0)

    names_b = ["const", "model_ml", "market_ml"]
    names_f = ["const", "model_ml", "market_ml", "model_rl", "market_rl"]
    print("  baseline (moneyline only):")
    for n, c, p in zip(names_b, m_base.params, m_base.pvalues):
        print(f"    {n:<11} coef={c:+.4f}  p={p:.4f}")
    print(f"    pseudo-R2={m_base.prsquared:.5f}  loglik={m_base.llf:.2f}")
    print("  with run-line terms:")
    for n, c, p in zip(names_f, m_full.params, m_full.pvalues):
        print(f"    {n:<11} coef={c:+.4f}  p={p:.4f}")
    print(f"    pseudo-R2={m_full.prsquared:.5f}  loglik={m_full.llf:.2f}")

    lr_stat = 2 * (m_full.llf - m_base.llf)
    from scipy import stats as st
    lr_p = st.chi2.sf(lr_stat, 2)
    print(f"\n  LR test (do the 2 run-line terms add anything?): "
          f"chi2={lr_stat:.3f}  p={lr_p:.4f}")
    print("  -> " + ("run-line terms ADD information beyond the moneyline pair"
                     if lr_p < 0.05 else
                     "run-line terms add NOTHING detectable beyond the moneyline pair"))

    # ---- 3) walk-forward second stage -> a 'true edge' ------------------------
    print("\n" + "=" * 72)
    print("3) SECOND-STAGE MODEL, FIT WALK-FORWARD (no season sees its own fit)")
    print("=" * 72)
    feats = np.column_stack([lg(mdl_ml), lg(mkt_ml), lg(mdl_rl), lg(mkt_rl)])
    seasons_sorted = sorted(set(season.tolist()))
    true_prob = np.full(len(rows), np.nan)
    for s in seasons_sorted:
        tr, te = season < s, season == s
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = sm.Logit(home_win[tr], sm.add_constant(feats[tr])).fit(disp=0)
        true_prob[te] = m.predict(sm.add_constant(feats[te]))
    ok = ~np.isnan(true_prob)
    print(f"  scored out-of-sample: {ok.sum()} games "
          f"(seasons {sorted(set(season[ok].tolist()))})")

    old_edge = mdl_ml[ok] - mkt_ml[ok]
    new_edge = true_prob[ok] - mkt_ml[ok]
    y = home_win[ok]
    print(f"  AUC  old model_ml={auc(y, mdl_ml[ok]):.4f}   "
          f"market={auc(y, mkt_ml[ok]):.4f}   second-stage={auc(y, true_prob[ok]):.4f}")
    print(f"  mean |edge|  old={np.abs(old_edge).mean():.4f}  new={np.abs(new_edge).mean():.4f}")

    # Does either edge actually predict winning the bet?
    print("\n  logit(bet won) ~ edge, for each edge definition:")
    for label, e in (("old (ml only)", old_edge), ("new (ml+rl)", new_edge)):
        picked_home = e > 0
        won = np.where(picked_home, y == 1, y == 0).astype(int)
        mag = np.abs(e)
        mm = sm.Logit(won, sm.add_constant(mag)).fit(disp=0)
        print(f"    {label:<14} coef={mm.params[1]:+.4f}  p={mm.pvalues[1]:.4f}  "
              f"win%={won.mean()*100:.1f}  n={len(won)}")

    np.save(os.path.join(RESULTS, "_joint_edge_cache.npy"),
            np.column_stack([season[ok], y, mkt_ml[ok], mdl_ml[ok],
                             true_prob[ok], old_edge, new_edge]))
    print(f"\n  cached per-game arrays -> scripts/results/_joint_edge_cache.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
