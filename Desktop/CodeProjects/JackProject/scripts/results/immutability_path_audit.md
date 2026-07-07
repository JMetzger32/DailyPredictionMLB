# A2 — Entry-creation / immutability path audit (integrate/full-fix @ 55710ce)

Question: after merging both branches, does every code path that creates a log entry
after the fact flag it `post_game_created`, and does any path rewrite a stored pick?

## Creation paths (all go through `_build_prediction_entry`, app.py:472 — which stamps
`prediction_timestamp` (write-once) and `model_version` on every entry)

| # | Path | Where | Timing | `post_game_created` | Status |
|---|------|-------|--------|---------------------|--------|
| 1 | Daily job / seed — `_log_predictions_for_date` for **today/future** (8 AM job, startup seed, `_refresh_today_odds` seed) | app.py:635 | pre-game | not set | ✅ correct (genuine pre-game pick) |
| 2 | Auto-heal backfill — `_log_predictions_for_date` for **past dates** | app.py:635 → flag at 648 | post-game | **True** | ✅ flagged |
| 3 | Mid-request create — `/api/predictions` when a finished game was never logged | app.py:1541 → flag at 1547 | post-game | **True** | ✅ flagged |

No other callers of `_build_prediction_entry` exist (grep: definition + 2 call sites).
Startup re-seed (app.py ~941) replaces *today's* entries pre-game when `x_scaled_features`
is missing — acceptable by design (both versions pre-game), noted as a caveat, not a flag case.

## Mutation paths (may they rewrite `predicted_winner` / `home_win_prob`? — NO)

| Path | Where | What it writes | Scores stored pick? |
|------|-------|----------------|---------------------|
| `update_yesterday_results` | app.py:549 | scores, actual_winner, correct, brier/ll | ✅ `entry["predicted_winner"] == actual` |
| `_log_predictions_for_date` immediate resolve (backfill) | app.py:663 | same | ✅ (entry is itself the flagged backfill) |
| `_resolve_unresolved_for_date` (30-min job, auto-heal) | app.py:720 | same + correct_rl | ✅ |
| `/api/predictions` live-resolve, stored entry exists | app.py:1519-1536 | same | ✅ `_s_correct = (stored_pick == actual_winner)`; brier/ll from **stored** prob; card shows `_s_correct` |
| `/api/predictions` odds write-back + `_refresh_today_odds` + `_store_closing_odds` | various | odds fields only (`away_ml`, `closing_*`, `clv`, `bet_rating`) | n/a — never touch pick/prob |

Grep proof: **zero** assignment sites for `entry["predicted_winner"] =` or
`entry["home_win_prob"] =` outside `_build_prediction_entry`. Line 1510 computes a
fresh-pick `correct`, but it only survives into the *new-entry* branch (where the entry IS
the fresh prediction); for stored entries it is overridden by `_s_correct` at 1536.

## Verdict
Both branches' logic merged consistently: **every post-hoc creation path is flagged, no
path rewrites a stored pick or probability, and all resolution paths score the stored pick.**
