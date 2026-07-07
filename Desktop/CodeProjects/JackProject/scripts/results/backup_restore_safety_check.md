# B1 — GitHub backup/restore clobber-safety check (integrate/full-fix)

## Question
Now that the missing `import requests` is fixed and backups will actually run: does
`_restore_file_from_github` (called at startup for picks/predictions/betting/closing-odds
logs + the artifacts pkl) unconditionally overwrite good local data with a stale GitHub
backup on the next deploy?

## What the restore logic actually did (pre-hardening)
**Not unconditional.** It had a size guard: overwrite only `if len(remote_bytes) > local_size`
(app.py:824). On a fresh deploy, local files = the repo-committed copies; the GitHub contents
API returns those same committed bytes, so restore was a same-size no-op. Two residual issues:

1. **Stale-but-larger remote over newer-but-smaller local.** Logs occasionally shrink
   (`_resolve_unresolved_for_date` prunes stale UTC-duplicate entries), so a bigger old
   backup could clobber a newer smaller local log. Low probability, real.
2. **The artifacts pkl restore has never been able to work at all**: GitHub's contents API
   returns base64 content only for files ≤ 1 MB. For the 13 MB pkl it returns empty content
   → decoded to 0 bytes → size guard skips. Safe (no clobber) but also no recovery — and a
   naive "fix" would have introduced the worst clobber: a stale backup pkl overwriting a
   freshly retrained artifact.
3. Bonus: `r.json()["content"]` would KeyError on responses without a content field;
   now `.get("content") or ""`.

## Hardening applied (small, behavior-conservative)
New `_should_restore(filepath, remote_bytes) -> (bool, reason)` decision function
(app.py, replaces the inline guard), rules most-specific first:
1. Empty remote → never restore (covers missing backups AND the >1MB contents-API case).
2. `.pkl` → restore **only when local file is missing/empty**; a present local artifact is
   never overwritten by a backup.
3. Date-keyed JSON logs (handles both `{date: …}` and picks' `{email: {date: …}}` shapes via
   `_latest_date_key`) → compare max date keys: local newer → skip; **remote newer → restore
   even if smaller**.
4. Fallback → original size guard (remote larger ⇒ restore).
Every decision logs its reason (`[github] Skipping restore of … — <reason>`).

## Dry-run decision matrix (real local files, synthetic remote blobs — all PASS)
| case | expected | got | reason |
|---|---|---|---|
| stale-but-larger remote log | skip | skip | local newer by date (2026-06-28 > 2026-06-23) |
| newer-but-smaller remote log | restore | restore | remote newer by date (2027-01-01 > 2026-06-28) |
| empty remote (>1MB API case) | skip | skip | remote empty |
| pkl, local present, huge remote | skip | skip | never overwrite artifacts with a backup |
| pkl, local missing | restore | restore | local pkl missing |
| betting_log.json local `{}` (2B), real remote backup | restore | restore | remote larger (32 > 2 bytes) |
| identical bytes | skip | skip | local up-to-date |

## Verdict
**Safe to deploy.** The current 1,211-row betting_log (SQLite — not touched by restore at
all; it rebuilds from predictions_log at startup) and the predictions_log cannot be
clobbered by a stale backup: date-aware guard blocks stale restores, the pkl can never be
overwritten while present, and legitimate recovery (bigger/newer backup after a redeploy
wiped local state) still works.

Note: pkl *backup* (push) may also be subject to GitHub API file-size behavior for 13 MB
payloads; the push path base64-encodes and PUTs (limit 100 MB) so it should succeed, but the
restore path intentionally no longer depends on it.
