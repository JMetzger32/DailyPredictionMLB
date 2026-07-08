# Retrain Execution Log — mlb_model_artifacts.pkl

Append-only history of artifact-regeneration attempts. (File created 2026-07-07 during
Phase E2 — no prior log existed; earlier attempts are reconstructed from session records.)

---

## 2026-07-03 — local attempt #1 (branch fix/eval-and-leak-phased) — ❌ FAILED
- `pip install xgboost` into `.venv` succeeded, but import failed:
  `libxgboost.dylib ... Library not loaded: @rpath/libomp.dylib` — macOS OpenMP runtime
  missing and **Homebrew is not installed** on this machine, so `brew install libomp` was
  unavailable. xgboost was uninstalled again (a half-broken install would break
  MLBModel's `from xgboost import XGBClassifier` guard, which catches ImportError but not
  XGBoostError).
- Consequence: no local retrain — running one anyway would silently drop the 50-model
  bootstrap ensemble from the artifact.

## 2026-07-07 — local attempt #2 (branch integrate/full-fix, Phase E3) — ❌ FAILED
- `pip install xgboost==3.3.0` succeeded; import failed identically to attempt #1:
  `Library not loaded: @rpath/libomp.dylib` (searched /opt/homebrew/opt/libomp/... —
  absent). **No Homebrew and no Docker on this machine**, so neither fallback B nor C is
  currently possible locally.
- xgboost uninstalled again immediately: a half-broken install raises `XGBoostError`
  (not `ImportError`) at import time, which would crash `MLBModel`'s optional-import
  guard and take the whole app down. Verified post-uninstall: `MLBModel import OK,
  HAS_XGBOOST = False`; smoke test green.
- **Conclusion: retrain must happen via fallback A (Render).** Exact commands below.

---

## Fallback paths (when local xgboost is unavailable)

**A. Render (recommended — xgboost works there):**
1. Merge `integrate/full-fix` → `main`, let Render deploy.
2. Weekly-retrain path (updates models in the existing pkl, ~5–10 min):
   `curl "https://dailypredictionmlb.onrender.com/api/retrain-model?key=<TRIGGER_SECRET>"`
3. Or full offline rebuild (models + baselines + plots) in a Render shell:
   `cd /opt/render/project/src/Desktop/CodeProjects/JackProject && python Main/MLBModel.py`
4. Verify per `scripts/RETRAIN_CHECKLIST.md` §4 (model_version stamped, 50 bootstrap
   models, no 1.4×, holdout ≈ 0.576 AUC) then let the 8 AM job's artifact backup push it,
   or commit the regenerated pkl.

**B. Docker (local, no brew needed):**
```bash
cd ~/Desktop/CodeProjects/JackProject
docker run --rm -v "$PWD":/w -w /w python:3.12-slim bash -c \
  "pip install -r Main/requirements.txt && python Main/MLBModel.py"
```
(Linux wheels bundle OpenMP; the pkl lands in updates/ on the host via the volume mount.)

**C. macOS with Homebrew (if brew gets installed):**
```bash
brew install libomp && .venv/bin/pip install xgboost==3.3.0 \
  && .venv/bin/python Main/MLBModel.py
```
