# OPERATIONS — what healthy looks like, and what to check first

One page. If something breaks, start at the top and work down.

## Healthy looks like

1. `GET /api/status` shows, for every job, a `last_success` within its cadence:
   - `daily_update` — daily, ~8:00 AM ET
   - `resolve_games` — every 30 min (while games are finishing)
   - `closing_odds` — daily, ~6:45 PM ET
   - `odds_refresh` — every 3 h
   - `github_backup` — several times a day (after log writes)
   - `retrain` — weekly (manual/cron trigger via /api/retrain-model)
   plus `github_token_set: true`, `odds_api_key_set: true`, and a non-null
   `model_version`.
2. `GET /api/betting` → `diagnostics.rows_qualifying` grows day over day.
3. The GitHub repo shows fresh `Auto-backup …` commits (this is the ONLY durable
   persistence across Render redeploys — everything else resets to the deploy snapshot).
4. Render logs contain `[job:<name>] OK` lines and `[github] PUSH OK` lines.

## If the betting page is empty

- `GET /api/betting` → read `diagnostics`. If `rows_total > 0` but
  `rows_qualifying == 0`, odds and results are never coexisting on a row. The
  three causes found in July 2026, in the order to check them:
  1. **Restore silently broken** — GitHub's contents API returns EMPTY content for
     files > 1 MB unless requested with `Accept: application/vnd.github.raw`
     (fixed 2026-07-14). Symptom: `[github] Skipping restore ... empty remote` in
     logs while the repo shows recent Auto-backup commits; each restart then boots
     from the stale deploy snapshot and its next push CLOBBERS the good backup.
  2. **Odds job never fires** — `odds_refresh` must be a cron at fixed ET times
     (10:15/13:15/16:15/18:15), never an `interval`: intervals count from process
     boot and a spin-down host rarely lives that long (fixed 2026-07-14).
     Check `/api/status → jobs.odds_refresh.last_success`.
  3. **Backups not landing at all** — `/api/status → github_token_set` +
     `[github] PUSH FAIL` lines; repair `GITHUB_TOKEN` (repo contents write;
     rotate before its expiry — current one dies 2026-11-11).
- Odds also patch on any /predictions page view (request-path safety net), so a
  single visit while books have lines up is enough for that day.
- If `rows_total == 0`: the betting_log table didn't initialize — look for
  `[startup] betting_log table init failed` in logs.

## If accuracy looks stale or frozen

- The API response carries `last_updated` and `heal_in_progress`. If
  `heal_in_progress: true`, missing days are being backfilled in the background —
  refresh in a minute or two. Server cache TTL is 60s; API responses are sent with
  `Cache-Control: no-store`, so a hard browser refresh always shows server truth.
- `rs.overall.games` counts only RESOLVED games — today's slate never appears in the
  total until games finish.

## If predictions fail to load

- `GET /health` fast + `GET /api/predictions` slow/500 → CPU contention on the free
  tier (a retrain running, or cold start). Wait, or check `/api/status → jobs.retrain`.
- `model_version: null` in `/api/status` → the artifact was never retrained/restored;
  trigger `/api/retrain-model?key=<TRIGGER_SECRET>` and watch `[retrain]` log lines.

## Environment variables (Render → Environment)

- `ODDS_API_KEY` — The Odds API key. Missing → no odds, betting page can never fill.
- `GITHUB_TOKEN` — token with repo contents write. Missing → NOTHING survives a
  redeploy (root cause of the July 2026 empty-betting-page incident; see
  scripts/results/phase1_root_causes.md).
- `TRIGGER_SECRET` — gates /api/refresh, /api/retrain-model, /api/trigger-*.

## Where state lives

- `Databases_and_logs/mlb_allseasons.db` — games, pitcher_stats, betting_log table.
  Committed to the repo; runtime changes lost on redeploy unless backed up.
- `Databases_and_logs/*.json` — predictions/betting/picks/closing-odds logs. Backed up
  to GitHub via the contents API (files must stay < 1 MB for restore to work).
- `Databases_and_logs/job_status.json` — /api/status state, gitignored, per-instance.
- `updates/mlb_model_artifacts.pkl` — model. Too big for contents-API restore; ships
  with the deploy; regenerate via retrain.
