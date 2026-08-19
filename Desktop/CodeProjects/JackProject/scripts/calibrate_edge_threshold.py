"""
calibrate_edge_threshold.py
---------------------------
Find the edge threshold that should drive "value bet" classification, using ALL past
seasons rather than the ~222-bet 2026 slice every prior analysis was stuck with.

Costs zero credits. Reuses scripts/backtest_threshold.py's pl_for/sweep_thresholds so
the repo keeps exactly one threshold sweep.

Three independent datasets, deliberately reported side by side:

  walkforward  season S scored by a model trained ONLY on seasons < S. HEADLINE.
               The one scheme that matches how a bettor actually experiences a
               market -- no future information, ever.
  loso         season S scored by a model trained on all OTHER seasons. Optimistic
               by construction (a 2021 fold knows 2022-2025), so it is a robustness
               check, never the decision input.
  live2026     the probabilities actually served pre-game in 2026, straight from
               predictions_log. Genuinely out-of-sample and genuinely shipped, but
               small. post_game_created rows are EXCLUDED -- those were generated
               after the final whistle using post-game baselines.

Every reported win% carries a Wilson 95% CI and every ROI a bootstrap 95% CI, because
point estimates at these sample sizes are how "n=38 says >0.12 loses 40%" ended up
frozen into two docstrings.

Usage:
    .venv/bin/python scripts/calibrate_edge_threshold.py [--stake 10] [--out FILE]
"""
import argparse
import json
import math
import os
import random
import sqlite3
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "updates"))

DB = os.path.join(_ROOT, "Databases_and_logs", "mlb_allseasons.db")
PRED_LOG = os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json")
OUT_DEFAULT = os.path.join(_ROOT, "scripts", "results", "edge_threshold_calibration.md")

from backtest_threshold import pl_for, sweep_thresholds  # noqa: E402

GRID = [round(x * 0.005, 3) for x in range(0, 41)]      # 0.000 .. 0.200 step 0.005
BUCKETS = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.08),
           (0.08, 0.12), (0.12, 0.20), (0.20, 1.01)]


def wilson(k, n, z=1.96):
    """Wilson score interval — correct at small n and near 0/1, unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(vals, n_boot=4000, seed=42):
    """Percentile bootstrap CI for the mean (used for ROI)."""
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def required_n(delta):
    """Bets per bucket to detect a `delta` win-rate gap at 80% power, a=0.05.
    n = 16*p(1-p)/d^2 with p(1-p) pinned at its 0.25 maximum (conservative)."""
    return int(16 * 0.25 / delta ** 2)


def devig(away_ml, home_ml):
    if away_ml is None or home_ml is None:
        return None, None
    raw = lambda m: abs(m) / (abs(m) + 100) if m < 0 else 100 / (m + 100)
    a, h = raw(away_ml), raw(home_ml)
    t = a + h
    return a / t, h / t


def _mk(p_home, away_ml, home_ml, home_won, season):
    """Build one bet row from a model home-prob and a priced game."""
    ai, hi = devig(away_ml, home_ml)
    if ai is None or home_won is None:
        return None
    home_pick = p_home > 0.5
    model_p = p_home if home_pick else 1 - p_home
    mkt_p = hi if home_pick else ai
    return {
        "season": season,
        "model_edge": model_p - mkt_p,
        "predicted_team_ml": home_ml if home_pick else away_ml,
        "correct": bool(home_won) == home_pick,
    }


def load_oof(conn, scheme):
    """OOF probabilities joined to primary odds via odds_game_link."""
    rows = conn.execute("""
        SELECT o.home_win_prob, s.away_ml, s.home_ml, o.home_win, o.season
        FROM oof_predictions o
        JOIN odds_game_link l ON l.game_id = o.game_id AND l.target='games'
        JOIN odds_snapshots s ON s.game_date_et = l.game_date_et
                             AND s.event_id = l.event_id
        WHERE o.scheme = ? AND s.horizon_days = 0
          AND s.away_ml IS NOT NULL AND l.confidence = 'exact'
    """, (scheme,)).fetchall()
    out = [_mk(p, a, h, w, s) for p, a, h, w, s in rows]
    return [r for r in out if r]


def load_live2026():
    """The probabilities actually served pre-game in 2026."""
    log = json.load(open(PRED_LOG))
    out = []
    for date_str, entries in log.items():
        for e in entries:
            if e.get("game_type") == "S" or e.get("post_game_created"):
                continue
            if e.get("away_ml") is None or e.get("actual_winner") in (None, "Tie"):
                continue
            p = e.get("home_win_prob")
            if p is None:
                continue
            r = _mk(p, e["away_ml"], e["home_ml"],
                    e["actual_winner"] == "Home", int(date_str[:4]))
            if r:
                out.append(r)
    return out


def fmt_sweep(rows, stake, title, lines):
    lines.append(f"\n### {title}  (n={len(rows)})\n")
    if not rows:
        lines.append("_no data_\n")
        return None
    res = sweep_thresholds(rows, grid=GRID, min_sample=50, stake=stake)
    if not res:
        lines.append("_no threshold reached the 50-bet minimum_\n")
        return None
    lines.append("| thresh | bets | win% | win% 95% CI | ROI | ROI 95% CI | Sharpe |")
    lines.append("|---:|---:|---:|:--|---:|:--|---:|")
    for r in res:
        bets = [b for b in rows if b["model_edge"] >= r["threshold"]]
        pls = [pl_for(b["predicted_team_ml"], b["correct"], stake) / stake for b in bets]
        lo, hi = wilson(r["wins"], r["bets"])
        rlo, rhi = boot_ci(pls)
        lines.append(f"| {r['threshold']:.3f} | {r['bets']} | {r['win_pct']*100:.1f}% "
                     f"| [{lo*100:.1f}, {hi*100:.1f}] | {r['roi']*100:+.1f}% "
                     f"| [{rlo*100:+.1f}, {rhi*100:+.1f}] | {r['sharpe'] or 0:.3f} |")
    best = max(res, key=lambda r: r["sharpe"] or -9)
    lines.append(f"\nBest by Sharpe: **{best['threshold']:.3f}** "
                 f"(n={best['bets']}, win {best['win_pct']*100:.1f}%, "
                 f"ROI {best['roi']*100:+.1f}%)")
    return best


def fmt_buckets(rows, stake, title, lines):
    lines.append(f"\n### Buckets — {title}\n")
    lines.append("| bucket | bets | win% | win% 95% CI | ROI |")
    lines.append("|:--|---:|---:|:--|---:|")
    for lo_e, hi_e in BUCKETS:
        b = [r for r in rows if lo_e <= r["model_edge"] < hi_e]
        if not b:
            continue
        w = sum(1 for r in b if r["correct"])
        pl = sum(pl_for(r["predicted_team_ml"], r["correct"], stake) for r in b)
        lo, hi = wilson(w, len(b))
        lines.append(f"| {lo_e:.2f}–{hi_e:.2f} | {len(b)} | {w/len(b)*100:.1f}% "
                     f"| [{lo*100:.1f}, {hi*100:.1f}] | {pl/(len(b)*stake)*100:+.1f}% |")


def fmt_per_season(rows, stake, lines):
    lines.append("\n### Per-season stability (walk-forward)\n")
    lines.append("A threshold that wins pooled but in only 2 of 6 seasons is noise.\n")
    seasons = sorted({r["season"] for r in rows})
    picks = [0.03, 0.05, 0.08, 0.12]
    lines.append("| season | n | " + " | ".join(f"ROI@{p:.2f}" for p in picks) + " |")
    lines.append("|:--|---:|" + "---:|" * len(picks))
    argmax = defaultdict(int)
    for s in seasons:
        sr = [r for r in rows if r["season"] == s]
        cells, best, bestv = [], None, -9
        for p in picks:
            b = [r for r in sr if r["model_edge"] >= p]
            if len(b) < 30:
                cells.append("–")
                continue
            roi = sum(pl_for(r["predicted_team_ml"], r["correct"], stake)
                      for r in b) / (len(b) * stake)
            cells.append(f"{roi*100:+.1f}%")
            if roi > bestv:
                bestv, best = roi, p
        if best is not None:
            argmax[best] += 1
        lines.append(f"| {s} | {len(sr)} | " + " | ".join(cells) + " |")
    lines.append(f"\nArgmax-season counts: "
                 f"{ {f'{k:.2f}': v for k, v in sorted(argmax.items())} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    have_oof = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name='oof_predictions'").fetchone()[0]
    datasets = {}
    if have_oof:
        for scheme in ("walkforward", "loso"):
            datasets[scheme] = load_oof(conn, scheme)
    datasets["live2026"] = load_live2026()

    L = ["# Edge Threshold Calibration", ""]
    L.append("Generated by `scripts/calibrate_edge_threshold.py`. Zero API credits.")
    L.append("")
    L.append("**Pre-registered decision rule (written before reading the tables):** "
             "adopt a new `GOOD_EDGE` only if it (a) beats 0.05 on the pooled "
             "walk-forward estimate with a bootstrap CI excluding zero improvement, "
             "(b) is non-inferior in at least 4 of 6 seasons, and (c) agrees "
             "directionally with the live-2026 sweep. Otherwise keep 0.05.")
    L.append("")
    L.append(f"Power reference: detecting a 10pp win-rate gap needs ~{required_n(0.10)} "
             f"bets per bucket; 5pp needs ~{required_n(0.05)}; 3pp needs "
             f"~{required_n(0.03)}.")
    for name in ("walkforward", "loso", "live2026"):
        if name not in datasets:
            continue
        rows = datasets[name]
        fmt_sweep(rows, args.stake, name, L)
        if rows:
            fmt_buckets(rows, args.stake, name, L)
    if datasets.get("walkforward"):
        fmt_per_season(datasets["walkforward"], args.stake, L)

    L.append("\n### Dataset sizes\n")
    for k, v in datasets.items():
        L.append(f"- `{k}`: {len(v)} resolved priced games")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
