# Phase 9 evidence — full verification (2026-07-13)

## Test suites
- Unit tests: **11/11 pass** (`tests/test_units.py`)
- Smoke test: **OK** — pkl loads, xgb ensemble on, model_version=f35f679086c3,
  sane prediction (BOS 0.530 vs CHA)
- E2E pipeline sim: **9/9 stages pass** (`scripts/e2e_local_sim.py`, new) —
  one fake game walked through the REAL extracted code units with an in-memory DB:
  1. prediction entry built  2. odds captured (bet_rating=good)  3. resolution
  scored vs stored pick  4. betting_log upsert via the real SQL **including the
  restart/NULL-re-upsert pattern that used to wipe odds — row survives**
  5. qualifying filter admits it  6. P/L $7.14 + quarter-Kelly $2.20/$1.57
  7. ISO-week bucket 2026-W27  8. accuracy scores it  9. past-date fast path
  synthesizes it from the log.

## Live endpoint sweep (local boot, port 5099)
| Endpoint | Status | Time |
|---|---|---|
| /api/predictions (today) | 200 | 0.75s |
| /api/predictions?date=2026-06-23 (1st) | 200 | 0.98s |
| /api/predictions?date=2026-06-23 (2nd, cached) | 200 | **0.002s** |
| /api/betting | 200 | 0.011s |
| /api/betting/weekly | 200 | 0.011s |
| /api/accuracy | 200 | 0.78s |
| /api/status | 200 | 0.005s |
| /api/calibration | 200 | 0.016s |
| /health | 200 | 0.002s |

## Hygiene
- App process killed by name (`pkill -f`), 0 survivors verified — the orphan-process
  trap from 2026-07-12 (wrapper killed, python child kept scheduling jobs overnight)
  is documented in CLAUDE.md.
- DB side effects from the boot reverted via git checkout before commit.

## Known-empty-by-design
/api/betting and /weekly return zero bets locally: the local betting_log has no
odds+result pairs (production persistence gap, pending GITHUB_TOKEN diagnosis and
this branch's deploy). The e2e sim proves the full pipeline works the moment real
rows exist.
