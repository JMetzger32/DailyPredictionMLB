# Phase 7 evidence — past-date fast path + response cache (2026-07-12)

## What was added (Main/app.py)
1. **Inference skip** — `_synth_results_from_log(game_ctx, log_by_pk)`: for past
   dates, when EVERY game has a stored pre-game prediction, per-game results are
   synthesized from the log and `predict_games_batch` is skipped. If any game lacks
   one, returns None and the route falls back to real inference. Cover probs /
   est_components aren't logged so they synthesize as None — the UI null-guards
   both (index.html) and hides those sections. Grep-able: `[pred] <date>: all N
   predictions from log — inference skipped`.
2. **Response cache** — `_past_pred_cache` (OrderedDict, max 30, lock). Checked
   right after date parsing; written only for dates older than yesterday where every
   non-skipped game has `actual_winner` (fully resolved — dates with a permanently
   unresolved/postponed game are deliberately never cached). Invalidation: any
   `_save_log()` write clears the whole cache.
3. GitHub push remains gated on `log_changed` (unchanged); cache write happens
   after the persist block so cached payloads reflect post-save state.

## Verification (local boot, port 5099, 2026-07-12)
- `?date=2026-06-25` (9 games, 1 permanently unresolved): 0.81s first hit with
  "inference skipped" in the log, 0.03s second hit — full route, memoized schedule,
  correctly NOT response-cached (unresolved game). from_log=9/9.
- `?date=2026-06-23` (15 games, all resolved): served from response cache —
  byte-identical payloads including last_updated across two hits, 0.01s.
- Today (no date param): 15 games, fresh probabilities, zero from_log flags,
  fast-path log line did not fire — live path untouched.
- Unit test: test_synth_results_from_log (full synthesis / missing-game fallback /
  None-prob fallback / empty slate). 11/11 pass.
- DB side effects from the boot (live resolves) reverted via git checkout before
  commit; app.py/tests are the only tracked changes.

## Production note
Locally inference is sub-second; on Render's 0.1 vCPU the batch is the dominant
cost of past-date requests (was the 30s-timeout culprit pre-batching), so skipping
it + response-caching resolved dates is the difference between multi-second and
millisecond history browsing there.
