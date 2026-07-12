# Phase 6 evidence — backtest scripts implemented from their pseudocode (2026-07-12)

## What was built
- `scripts/backtest_threshold.py` — grid-searches edge thresholds 0.01–0.15 over
  resolved bets (MIN_SAMPLE=30 per threshold), reports Win%/ROI/Sharpe/net P/L,
  recommends by Sharpe, saves an ROI-vs-sample plot. REPORT-ONLY: never touches the
  live 0.05 threshold in Main/app.py.
- `scripts/backtest_kelly.py` — simulates flat $10 vs quarter-Kelly (production
  default) vs half-Kelly (skeleton's suggestion) on resolved value bets; reports
  staked/net/ROI/Sharpe/max-drawdown/avg-stake + cumulative-P/L and stake plots.
  Kelly math mirrors Main/app.py `_kelly_stake` (comment marks the sync point).

## Deviations from the pseudocode (deliberate)
- Data source is the SQLite `betting_log` table, not `betting_log.json` — the DB is
  the authoritative store; threshold script falls back to JSON if the DB is unreadable.
- Both scripts gate on data volume: exit 1 with an explanation at 0 rows; loud
  "directional only" warning under ~200 resolved bets (the skeleton's own bar).
- Kelly script adds max drawdown + Sharpe (risk-adjusted view the pseudocode lacked)
  and uses the production quarter-Kelly/5%-cap parameters alongside half-Kelly.

## Verification (2026-07-12)
- Real DB (0 odds+result rows): both scripts exit 1 gracefully with the
  GITHUB_TOKEN/OPERATIONS.md pointer. No crash, no plot.
- Seeded 120 synthetic value bets (random.seed=42, edge-correlated outcomes),
  then reverted the DB via git checkout before commit:
  - threshold grid: sample shrinks monotonically 120→31 across 0.01→0.11;
    thresholds ≥0.12 correctly dropped (< MIN_SAMPLE). Best-by-Sharpe was 0.11
    (N=31, ROI 32.3%) — exactly the small-N mirage the HOLD rule exists for.
  - kelly: quarter-Kelly placed 89/120 bets (passed on f*≤0), staked $326.74
    vs flat $1200; max drawdown $44.83 vs flat $119.48. Half-Kelly staked only
    ~1.17x quarter (not 2x) because the 5% cap binds on large edges.

## Decision status: HOLD
Live data has 0 qualifying bets (odds/results gap; persistence fix awaits deploy).
No threshold or sizing change is recommended or made. Re-run both scripts once
`/api/betting` diagnostics show rows_qualifying ≥ ~200.
