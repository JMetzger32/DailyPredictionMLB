# Phase 1 — Edge calculation audit (fix/edge-calibration, 2026-08-06)

Question: is `model_edge` (and therefore the Value Bet / Toss-Up / No Value split)
computed from raw, uncalibrated probabilities, and is the market side carrying
bookmaker vig? Both sub-questions answered below by direct code read + empirical
check against the live logs.

## 1a. Which probability feeds `model_edge` — CONFIRMED: a flat-blended, NOT fitted-calibrated, probability

Single computation site: `_compute_odds_fields()`, **Main/app.py:1454-1502**.

```python
# Main/app.py:1469-1474
if away_impl is not None and home_impl is not None:
    predicted  = pred_result["predicted_winner"]
    model_prob  = pred_result["home_win_prob"] if predicted == "Home" else pred_result["away_win_prob"]
    market_prob = home_impl                    if predicted == "Home" else away_impl
    edge = model_prob - market_prob
    model_edge = round(edge, 4)
```

`pred_result` always originates from `predict_game()` or `predict_games_batch()`
(`Main/MLBModel.py`). Both apply one fixed transformation to the raw ensemble average
before returning `home_win_prob`/`away_win_prob`:

```python
# Main/MLBModel.py:940-944 (predict_game); identical at 1073-1076 (predict_games_batch)
# Soft recalibration: nudge 4% toward MLB home win prior (53%)
_HOME_PRIOR  = 0.53
_RECAL_BLEND = 0.04
prob = (1 - _RECAL_BLEND) * prob + _RECAL_BLEND * _HOME_PRIOR
```

**Full chain:** `ensemble.predict_proba()` (LR + GB + XGB + 50 bootstrap XGBs, averaged)
→ flat `0.96·p + 0.04·0.53` blend → `home_win_prob` → `edge = model_prob − market_prob`.

The nuance that matters: the probability is **not** literally the raw ensemble output —
but the only thing applied is a **fixed linear shrink of 4% toward 0.53**. That is a
constant, chosen a priori, that shifts every probability by at most 2 percentage points
and shrinks spread by 4%. It is **not** a fitted calibrator, and it cannot correct a
probability-dependent miscalibration (the model being too confident specifically in the
70-80%+ range), because it applies the same shrink everywhere.

**No fitted calibrator exists in the live path.** Platt/isotonic fitting appears only in
`scripts/calibration_live_check.py`, which prints a recommendation and never persists or
wires in a calibrator — verified by grep for `platt` across `Main/`, `updates/`,
`scripts/`: `scripts/calibration_live_check.py` is the sole hit. `Main/app.py:2465`'s own
comment concedes the point: `# 1/4 Kelly: live calibration error (ECE ~ 0.079) means full
Kelly would overbet.` — the app knows its probabilities are miscalibrated and compensates
at the *staking* layer, while `model_edge` and `bet_rating` still consume the
uncorrected probability.

**Verdict on the hypothesis: SUPPORTED.** Edge is computed from a probability that carries
the model's known miscalibration; the 4% blend is far too blunt to remove it.

## 1b. Is the market side de-vigged — RULED OUT: de-vig is already correct

The suspected missing de-vig **does not exist as a bug**. `updates/schedule_fetcher.py`
normalizes both implied probabilities by their sum:

```python
# updates/schedule_fetcher.py:545-549
def _american_to_raw(ml):
    """Convert American moneyline to raw (pre-vig) implied probability."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

# updates/schedule_fetcher.py:598-600, 617-618
away_raw = _american_to_raw(away_ml)
home_raw = _american_to_raw(home_ml)
total    = away_raw + home_raw  # >1 due to vig
...
"away_implied": round(away_raw / total, 4),
"home_implied": round(home_raw / total, 4),
```

This is the standard multiplicative de-vig. Empirical confirmation on the live log — every
stored pair sums to exactly 1.0, so no overround survives into the edge calculation:

```
entries with both implied: 44
distinct sums: [(1.0, 44)]
min/max: 1.0 1.0
```

One methodological note (not a bug, but worth recording): odds are averaged across
bookmakers *first*, then de-vigged **once** on that cross-book average — there is no
per-book de-vig before averaging. Averaging American moneylines is also not identical to
averaging probabilities. Both effects are small relative to the miscalibration in 1a.

**Verdict: de-vig hypothesis RULED OUT. No fix needed here.** Do not re-investigate.

## Edge reconstruction validated (enables Phase 3)

`betting_log.json` stores `away_implied`/`home_implied` on only 44 of 222 usable rows, but
`away_ml`/`home_ml` on all 222. Recomputing the de-vigged market probability from the
moneylines with the formula above and re-deriving edge reproduces the stored `model_edge`
**exactly on all 222 rows**:

```
rows: 222
max abs diff vs stored model_edge: 0.0
rows matching within 0.0002: 222 / 222
```

So Phase 3 can recompute edge for the full sample without being limited to the 44
implied-bearing rows, and any change it reports is attributable purely to the
probability substitution.

## Data completeness — a 3-day odds blackout (2026-07-29 → 07-31)

Git-history audit of the data logs: `betting_log.json`'s history on `origin/main` is 100%
`Auto-backup` commits plus the original reorg commit (`2b719f1`) — no manual edits,
resets, or force-overwrites. Provenance is clean. But there is a real gap:

| date | predictions_log entries | with odds (`away_ml`) | with `bet_rating` | resolved |
|---|---|---|---|---|
| 2026-07-28 | 15 | 14 | 14 | 15 |
| 2026-07-29 | 16 | **0** | **0** | 16 |
| 2026-07-30 | 10 | **0** | **0** | 10 |
| 2026-07-31 | 15 | **0** | **0** | 15 |
| 2026-08-01 | 15 | 14 | 14 | 15 |

`betting_log.json` skips from the `2026-07-28` key straight to `2026-08-01`. Predictions
were generated and games resolved normally on all three days; only **odds attachment**
failed, so no row could be rated.

**Likely cause** (well-corroborated, not proven): The Odds API free tier allows 500
credits/month. Render's free tier spins down after ~15 min idle, and until commit
`bef2049` every cold-boot page view re-fetched odds already present in the log — a credit
leak. The blackout starting 07-29 and ending exactly on 08-01, the calendar month
boundary when the quota resets, matches quota exhaustion precisely. Commit `bef2049`
(this branch's parent) adds `get_last_odds_quota()` so this becomes visible in
`/api/status` instead of silent, plus a gate that reuses already-captured odds.

**Effect on Phases 2-3:** the ~41 affected games are simply *absent* from the sample, not
miscategorized into a bucket. Win rates are therefore unbiased by the gap. The n=222
figure should be read as "current data minus a known 3-day blackout," not a full census
of 2026-07-16 → 08-06.

## Sample scope (for Phases 2-3)

All 222 rated+resolved rows are `game_type = 'R'`, `post_game_created` unset, and carry a
single `model_version` — `b9133b95d2ec`, the currently deployed one. No model-version
scoping or mixed-model contamination to correct for.

## Status

- 1a hypothesis (edge built on uncalibrated probability): **CONFIRMED as mechanism** — a
  fixed 4%-toward-0.53 blend is the only correction applied; no fitted calibrator exists.
- 1b hypothesis (missing de-vig): **RULED OUT** — de-vig present and empirically exact.
- Whether the miscalibration actually *explains* the Value Bet / Toss-Up inversion is
  quantified in Phase 2, and tested by substitution in Phase 3.
