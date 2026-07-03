#!/usr/bin/env python3
"""
Minimal smoke test: load the model artifacts and generate one prediction.

Purpose: catch pkl / feature-alignment / prediction-path breakage after each edit.

Note on xgboost: the artifacts pkl contains XGBoost bootstrap models. When a working
xgboost (with libomp) is unavailable locally, we load the pkl with a stub-unpickler that
replaces xgboost classes with placeholders, and call predict_game WITHOUT the xgb ensemble
(xgb_bootstrap_models=None). This validates the LR + GB + scaler + feature-vector path,
which is what the Phase 1 edits touch. The full xgb ensemble is validated in production.

Exit 0 on success, 1 on failure.
"""
import os
import sys
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "Main"))

ARTIFACTS_PATH = os.path.join(_ROOT, "updates", "mlb_model_artifacts.pkl")


class _Stub:
    """Placeholder for classes we can't import locally (e.g. xgboost)."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("xgboost"):
            return _Stub
        return super().find_class(module, name)


def main():
    try:
        import xgboost  # noqa: F401
        with open(ARTIFACTS_PATH, "rb") as f:
            artifacts = pickle.load(f)
        xgb_ok = True
    except Exception:
        # xgboost unavailable/broken locally -> stub it out and skip the xgb ensemble
        with open(ARTIFACTS_PATH, "rb") as f:
            artifacts = _SafeUnpickler(f).load()
        xgb_ok = False

    from MLBModel import predict_game

    team_baselines = artifacts["team_baselines"]
    sp_baselines = artifacts["sp_baselines"]
    lr_model = artifacts["lr_model"]
    scaler = artifacts.get("scaler")
    gb_model = artifacts.get("gb_model")

    teams = list(team_baselines.keys())
    assert len(teams) >= 2, "need at least two teams in baselines"
    home_ts = team_baselines[teams[0]]
    away_ts = team_baselines[teams[1]]

    sp_ids = list(sp_baselines.keys())
    assert len(sp_ids) >= 2, "need at least two SPs in baselines"
    home_sp = sp_baselines[sp_ids[0]]
    away_sp = sp_baselines[sp_ids[1]]

    result = predict_game(
        home_ts, away_ts, home_sp, away_sp, lr_model,
        scaler=scaler, gb_model=gb_model,
        xgb_bootstrap_models=(artifacts.get("xgb_bootstrap_models") if xgb_ok else None),
    )

    p = result["home_win_prob"]
    assert 0.0 <= p <= 1.0, f"home_win_prob out of range: {p}"
    print(f"[smoke] OK  xgb_ensemble={'on' if xgb_ok else 'off (stubbed)'}  "
          f"model_version={artifacts.get('model_version')}  "
          f"home_win_prob={p:.3f} ({teams[0]} vs {teams[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
