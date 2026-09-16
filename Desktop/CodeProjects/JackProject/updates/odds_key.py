"""
Resolve The Odds API key without putting it in the repo or a command line.

Order:
  1. ODDS_API_KEY environment variable (how production/Render supplies it)
  2. ~/.odds_api_key            (local dev; chmod 600, outside the repo)
  3. ODDS_API_KEY_FILE env var pointing at any other file

Kept separate so long-running backfill jobs and the Flask app resolve the key the
same way, and so a key never has to be pasted into a shell command (where it would
land in shell history and terminal scrollback).
"""
import os

DEFAULT_KEY_FILE = os.path.expanduser("~/.odds_api_key")


def load_odds_api_key(verbose=False):
    """Return the API key, or '' if none is available. Never prints the key."""
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if key:
        if verbose:
            print(f"[key] using ODDS_API_KEY env var (len={len(key)})")
        return key
    path = (os.environ.get("ODDS_API_KEY_FILE") or DEFAULT_KEY_FILE).strip()
    if path and os.path.exists(path):
        try:
            key = open(path).read().strip()
        except Exception as e:
            if verbose:
                print(f"[key] could not read {path}: {e}")
            return ""
        if key and verbose:
            print(f"[key] using key file {path} (len={len(key)})")
        return key
    if verbose:
        print(f"[key] no key found (env ODDS_API_KEY unset, {path} absent)")
    return ""


def key_fingerprint(key):
    """Short non-reversible fingerprint, safe to log for 'which key is this?'."""
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:8] if key else "none"
