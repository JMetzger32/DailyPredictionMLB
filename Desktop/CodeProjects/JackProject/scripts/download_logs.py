"""
download_logs.py
----------------
Pulls the latest betting_log.json and predictions_log.json from GitHub
(where Render backs them up after each daily update) to your local machine.

Run this before any analysis scripts:
    export GITHUB_TOKEN=your_personal_access_token
    python3 scripts/download_logs.py

Your GITHUB_TOKEN needs read access to the JMetzger32/DailyPredictionMLB repo.
Find it in: GitHub → Settings → Developer settings → Personal access tokens.
"""

import os
import sys
import json
import base64
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO  = "JMetzger32/DailyPredictionMLB"

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_SCRIPTS_DIR)

FILES = [
    (
        "Databases_and_logs/betting_log.json",
        os.path.join(_ROOT, "Databases_and_logs", "betting_log.json"),
    ),
    (
        "Databases_and_logs/predictions_log.json",
        os.path.join(_ROOT, "Databases_and_logs", "predictions_log.json"),
    ),
]


def download_file(github_path, local_path, token, repo):
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/contents/{github_path}"
    r   = requests.get(url, headers=headers, timeout=15)

    if r.status_code == 404:
        print(f"  {github_path}: not found on GitHub (404) — skipping")
        return False
    if r.status_code != 200:
        print(f"  {github_path}: GitHub returned {r.status_code} — skipping")
        return False

    remote_content = base64.b64decode(r.json()["content"])
    remote_size    = len(remote_content)
    local_size     = os.path.getsize(local_path) if os.path.exists(local_path) else 0

    with open(local_path, "wb") as f:
        f.write(remote_content)

    try:
        data    = json.loads(remote_content)
        entries = sum(len(v) for v in data.values()) if isinstance(data, dict) else len(data)
        dates   = sorted(data.keys()) if isinstance(data, dict) else []
        date_range = f"{dates[0]} → {dates[-1]}" if dates else "no dates"
        print(f"  {github_path}: {entries} entries across {len(dates)} dates ({date_range})  [{remote_size:,} bytes]")
    except Exception:
        print(f"  {github_path}: downloaded {remote_size:,} bytes (could not parse JSON for summary)")

    return True


def main():
    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN is not set.")
        print("Run:  export GITHUB_TOKEN=your_token_here")
        sys.exit(1)

    print(f"Downloading logs from GitHub ({GITHUB_REPO})...\n")

    success = 0
    for github_path, local_path in FILES:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if download_file(github_path, local_path, GITHUB_TOKEN, GITHUB_REPO):
            success += 1

    print(f"\nDone — {success}/{len(FILES)} files downloaded to Databases_and_logs/")
    if success == len(FILES):
        print("You can now run backtest_threshold.py and backtest_kelly.py.")


if __name__ == "__main__":
    main()
