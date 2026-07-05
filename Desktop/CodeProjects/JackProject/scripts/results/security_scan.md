# Step 3 — Security scan: Odds API key in git history

## Question
Does the previously-pasted Odds API key `8c0af7bc746aa4ab3b570ec464677d63` (redacted prefix
`8c0af7bc…`) appear anywhere in git history on this branch (or the whole repo), not just the
working tree?

## Method (read-only)
Repo root is the user's HOME dir (`/Users/jackmetzger`); the project is a subdirectory.
Commands run:

```
git rev-list --count main..HEAD          # 14 commits on the feature branch
git rev-list --all --count               # 122 commits total in the repo
git log -S"<key>" --oneline main..HEAD   # pickaxe, branch
git log -S"<key>" --oneline --all        # pickaxe, full history
git log -G"<key>" --oneline --all        # regex, full history (catches any occurrence)
git grep -n "<key>" $(git rev-list main..HEAD)   # grep every branch commit tree
git grep -n "<key>"                      # working tree / index
grep -l "<key>" <plan file>              # untracked plan file
git grep -nE "ODDS_API_KEY *= *[\"'][0-9a-f]{16,}"  # hardcoded assignment
```

## Result: CLEAN — 0 matches everywhere
| scope | matches |
|---|---|
| feature branch commits (`main..HEAD`, 14) | 0 |
| full history (`--all`, 122 commits) — pickaxe `-S` | 0 |
| full history (`--all`) — regex `-G` | 0 |
| `git grep` across all branch commit trees | 0 |
| tracked working tree / index | 0 |
| untracked plan file | 0 |
| hardcoded `ODDS_API_KEY = "<hex>"` assignment | 0 |

**No affected commit hashes.** The key was pasted only in chat; it was never written to any
tracked or committed file. `get_mlb_odds(ODDS_API_KEY)` reads the key from the environment
(`os.environ["ODDS_API_KEY"]`), which is the correct pattern — nothing to purge from history.

## Forward guard (this commit)
Per the user's decision, **`.gitignore` entries only** (no pre-commit hook, no changes to
`~/.git/hooks`). Added to `.gitignore`:

```
.env
.env.*
*.key
*.pem
secrets*.json
credentials*.json
```

## Recommendation (user action, not a code change)
Rotate the Odds API key, since it was pasted in plaintext into the chat transcript. It is not
in the repo, but the transcript is outside git's control.
