# Phase 1 — Root-cause findings (fix/betting-accuracy-recovery, 2026-07-09)

## 1a. Betting page empty — CONFIRMED, mechanism fully established

**Symptom:** live `/api/betting` returns 200 with all-zero buckets (`value_bets.games: 0`,
`tracking_start: null`). Reproduced locally against the repo DB — not Render-specific.

**Deterministic cause:** the endpoint filter (Main/app.py:2003-2008) requires
`bet_rating IS NOT NULL AND correct IS NOT NULL`. In the repo DB, all 1,211 betting_log
rows have `bet_rating = NULL` / `predicted_team_ml = NULL` (query evidence below), so the
intersection is empty. The filter itself is *correct* (P/L needs odds); the data is the bug.

```
betting_log: total=1211  with_bet_rating=0  resolved=1211  both=0  with_predicted_ml=0
predictions_log.json: 1222 entries, 0 with bet_rating, 0 with away_ml (2026-03-27..06-28)
betting_log.json: literally "{}" (2 bytes)
closing_odds_log.json: does not exist locally
```

**Why odds and resolution never coexist durably:**
1. Live `/api/debug/odds` (2026-07-09): `odds_api_key_set: true`, 10 games fetched,
   `predictions_with_odds: 10` → odds DO attach to *today's* entries on Render, daily.
2. Games resolve the next morning → rows would then qualify… if state survived.
3. Render free-tier restarts reset the disk to the deploy snapshot; recovery depends on
   `_restore_file_from_github`. But **the GitHub auto-backup has never committed once**:
   `git log --grep="Auto-backup"` shows only the commits that *implemented* the feature
   (924e907, 3a865da). Last real commit touching predictions_log.json: 9812e99 (June 29).
   So every restart wipes the odds-bearing day back to the June-28, odds-less snapshot.
4. Net effect: `bet_rating` rows exist only for "today", `correct` rows only for the past
   → intersection permanently empty except brief same-awake-period windows.

**Root cause of the backup failure: on Render, `GITHUB_TOKEN` is missing or invalid.**
`_push_file_to_github` prints `[github] GITHUB_TOKEN not set — skipping backup` (or a
4xx line) and continues. Cannot be verified further from here — **USER ACTION: check
Render → Environment for `GITHUB_TOKEN`; if present, paste any `[github]` log lines from
a recent day.** (IMPROVEMENTS.md's claim that backups are "now functional" was wrong —
the import fix removed one blocker; the token was never there to begin with.)

**Ruled out:** `ON CONFLICT(game_pk) DO UPDATE` upsert — `game_pk INTEGER PRIMARY KEY`
exists (updates/init_betting_log.py:28); local SQLite 3.45.1 ≥ 3.24 requirement; upsert
executes cleanly (regression test added in Phase 2). Scheduler duplicate jobs — single
gunicorn worker default; no evidence of doubles. TRIGGER_SECRET — unrelated to this path.

## 1b. Accuracy page "frozen" — CONFIRMED (three stacking causes)

1. Server-side response cache `_accuracy_cache`, TTL 60s (app.py:1763; docstring claims
   5 min — stale). Same Response object replayed within the window.
2. `_auto_heal_log` runs in a background daemon thread when >3 days are missing
   (app.py:1789-1795) and the response is computed from the PRE-heal log — new days
   appear only on a later request after the thread finishes and the cache expires.
3. No Cache-Control headers anywhere in the app → browser heuristic caching + stale tabs.
   (CDN ruled out: `cf-cache-status: DYNAMIC` observed.)
Also: `rs.overall.games` counts only RESOLVED entries — today's games never appear in the
count until resolved, which reads as "stuck" day-to-day. Live API served 1,360 games while
the user's tab showed 1,212 → user-side staleness dominated, amplified by (1)-(3).

## 1c. Accuracy "drop since leak fix" — NOT A BUG, and premise partly moot

- **The leak-fixed model has never run in production** (live `model_version: null`; the
  deployed pkl predates the fix). The leak inflated *offline holdout* metrics only
  (0.6083 → 0.5762 AUC, scripts/PHASE4_RESULTS.md); live accuracy never received leak
  benefits. Expected LIVE drop from deploying the leak-fixed retrain: **≈ 0**.
- Observed live accuracy 53.9% (733/1360) vs always-home baseline 52.1% — consistent with
  the honest ~0.576-AUC model family, and with the retrain's 2026 validation acc 0.544.
- **expected drop: ~0 pp (live), observed drop: n/a (new model not yet deployed),
  unexplained gap: none.**
- Feature-construction parity (new artifact): pkl `feature_cols` == code `FEATURE_COLS`
  exact order (18); scaler `n_features_in_`=18; LR coef shape (1,18); end-to-end
  `x_scaled_features` length 18; LR prob recomputed from the stored scaled vector is
  finite/in-range. No 1.4× boost in code. SP-ERA dominance corrected (LR coef
  −0.3693 → −0.0649) as expected post-leak-fix.
