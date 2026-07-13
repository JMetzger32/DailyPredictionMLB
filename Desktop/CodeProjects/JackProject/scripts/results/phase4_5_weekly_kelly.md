# Phase 4+5 evidence — weekly betting view + quarter-Kelly sizing (2026-07-11)

## What was added
- `GET /api/betting/weekly` — value bets grouped by ISO week (Python `isocalendar()`;
  SQLite `strftime %G/%V` needs >= 3.46 so grouping stays in Python). Per week:
  bets/wins/losses/win_rate/net_pl/avg_edge/avg_clv. `?week=2026-Wnn` returns that
  week's individual bet rows.
- `_bet_row()` — module-level bet-row builder shared by /api/betting recent list and
  weekly detail (was inline in betting_stats).
- `_kelly_stake(win_prob, ml, bankroll, fraction, max_stake_pct)` — fractional Kelly,
  American odds; None on missing inputs, 0.0 when f* <= 0; stake capped at
  `max_stake_pct` of bankroll. Flat, non-compounding.
- `/api/betting` query params (clamped): `bankroll` (default 100, 1–1e6),
  `kelly_fraction` (default 0.25, 0–1), `max_stake_pct` (default 0.05, 0.001–1).
  Response gains `kelly` summary (bets/total_staked/net_pl/roi + params) and per-row
  `kelly_stake`/`kelly_pl` on recent bets. Default is quarter-Kelly because live
  calibration error (ECE ~ 0.079) makes full Kelly overbet.
- betting.html: weekly results table (click a week -> its bets render in the table
  below, "show recent" restores), ¼K Stake / ¼K P/L columns, Kelly tile + footnote.

## Verification (local boot, port 5099, 2026-07-11)
- Unit tests: 10/10 pass (3 new: test_kelly_stake, test_week_key, test_bet_row_kelly —
  includes ISO year rollover 2024-12-30 -> 2025-W01 and the f*<=0 no-bet case).
- Local DB has 0 qualifying rows (known odds/results gap, resolves in prod now that
  GITHUB_TOKEN is set) so 8 rows were temporarily seeded across two ISO weeks, then
  the DB was reverted (git checkout) before commit:
  - `/api/betting/weekly` -> W27: 4 bets 3-1, net +$21.23 (hand-check: +7.14 +9.09
    +15.00 -10.00 = +21.23 OK); W28: 4 bets 2-2, net -$1.17 (-10 +10.50 +8.33 -10 OK).
  - `?week=2026-W28` -> 4 rows; sample -120 win: pl +$8.33, kelly_stake $3.00
    (p=0.6, b=0.833, f*=0.12, quarter -> 0.03 x $100), kelly_pl +$2.50. All hand-check.
  - `/api/betting` -> kelly: 8 bets, $30.00 staked, net +$7.60, ROI 25.3% (synthetic).
  - `?bankroll=500` reflected in response params (clamping path exercised).
- Empty-data path: with the real (unseeded) DB both endpoints return 200 with empty
  weeks/zero kelly — no 500s, page unaffected.
