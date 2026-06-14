"""
test_ensemble_weights.py
------------------------
Tests ensemble weighting strategies for LR + GBM + XGBoost on the
2025 holdout. Generates "Ensemble Weighting Graph" saved as PNG.

Run:  python3 scripts/test_ensemble_weights.py

Two strategies compared:
  Option A — Accuracy-proportional: weights ∝ each model's 2025 holdout accuracy
  Option B — Grid search: all (w_lr, w_gbm, w_xgb) combos in 0.05 steps

After reviewing the graph, send a follow-up plan to implement the
best weights in Main/MLBModel.py and Main/app.py.
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering for PNG output
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, os.path.join(_ROOT, "Main"))
sys.path.insert(0, os.path.join(_ROOT, "updates"))

from MLBModel import (
    load_data, build_team_game_log, compute_rolling_team_features,
    merge_sp_stats, merge_bullpen_era, assemble_features,
    FEATURE_COLS, DB_PATH, RANDOM_STATE,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed — running LR + GBM only")


def main():
    print("=" * 60)
    print("Ensemble Weighting Analysis — 2025 Holdout")
    print("=" * 60)

    # ── Build feature matrix ──────────────────────────────────────
    print("\n[1/4] Building feature matrix from DB...")
    df, pitcher_stats, bullpen_stats = load_data(DB_PATH)
    tgl = build_team_game_log(df)
    tgl = compute_rolling_team_features(tgl)
    tgl = merge_sp_stats(tgl, pitcher_stats)
    tgl = merge_bullpen_era(tgl, bullpen_stats)
    model_df = assemble_features(df, tgl)
    valid    = model_df.dropna(subset=FEATURE_COLS)
    print(f"  {len(valid)} complete-feature games")

    # ── Train / test split ────────────────────────────────────────
    train = valid[valid["season"].between(2021, 2024)]
    test  = valid[valid["season"] == 2025]
    if len(test) == 0:
        print("ERROR: No 2025 data found in DB — cannot run holdout evaluation.")
        sys.exit(1)

    X_train, y_train = train[FEATURE_COLS].values, train["home_win"].values
    X_test,  y_test  = test[FEATURE_COLS].values,  test["home_win"].values

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"  Train: {len(train)} games (2021-2024)  |  Test: {len(test)} games (2025)")

    # ── Train individual models ───────────────────────────────────
    print("\n[2/4] Training individual models...")

    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_sc, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_sc))
    lr_p   = lr.predict_proba(X_test_sc)[:, 1]
    print(f"  LR  accuracy: {lr_acc:.3f}")

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    gbm.fit(X_train, y_train)
    gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
    gbm_p   = gbm.predict_proba(X_test)[:, 1]
    print(f"  GBM accuracy: {gbm_acc:.3f}")

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.8, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
        xgb_p   = xgb.predict_proba(X_test)[:, 1]
        print(f"  XGB accuracy: {xgb_acc:.3f}")
    else:
        xgb_acc = gbm_acc  # fallback: treat as GBM copy
        xgb_p   = gbm_p

    # Current equal-weight ensemble
    equal_p   = (lr_p + gbm_p + xgb_p) / 3
    equal_acc = accuracy_score(y_test, (equal_p >= 0.5).astype(int))
    print(f"  Equal-weight ensemble: {equal_acc:.3f}")

    # ── Option A: accuracy-proportional weights ───────────────────
    print("\n[3/4] Option A — accuracy-proportional weights...")
    total = lr_acc + gbm_acc + xgb_acc
    w_lr_A, w_gbm_A, w_xgb_A = lr_acc / total, gbm_acc / total, xgb_acc / total
    optA_p   = w_lr_A * lr_p + w_gbm_A * gbm_p + w_xgb_A * xgb_p
    optA_acc = accuracy_score(y_test, (optA_p >= 0.5).astype(int))
    print(f"  Weights: LR={w_lr_A:.3f}  GBM={w_gbm_A:.3f}  XGB={w_xgb_A:.3f}")
    print(f"  Option A accuracy: {optA_acc:.3f}")

    # ── Option B: grid search ─────────────────────────────────────
    print("\n[4/4] Option B — grid search over weight combinations...")
    grid_results = []
    steps = np.arange(0.0, 1.01, 0.05)
    for w_lr in steps:
        for w_gbm in steps:
            w_xgb = round(1.0 - w_lr - w_gbm, 4)
            if w_xgb < -0.001:
                continue
            w_xgb = max(w_xgb, 0.0)
            combo_p = w_lr * lr_p + w_gbm * gbm_p + w_xgb * xgb_p
            acc     = accuracy_score(y_test, (combo_p >= 0.5).astype(int))
            grid_results.append((round(w_lr, 2), round(w_gbm, 2), round(w_xgb, 2), acc))

    best = max(grid_results, key=lambda x: x[3])
    print(f"  Searched {len(grid_results)} weight combinations")
    print(f"\n  Top 10 combos:")
    print(f"  {'w_LR':>6} {'w_GBM':>6} {'w_XGB':>6} {'Accuracy':>10}")
    for row in sorted(grid_results, key=lambda x: -x[3])[:10]:
        marker = " ← best" if row == best else ""
        print(f"  {row[0]:>6.2f} {row[1]:>6.2f} {row[2]:>6.2f} {row[3]:>10.3f}{marker}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_strategies = [
        ("LR (solo)",           lr_acc),
        ("GBM (solo)",          gbm_acc),
        ("XGB (solo)",          xgb_acc),
        ("Equal 1/3 each",      equal_acc),
        ("Option A (acc-prop)", optA_acc),
        (f"Option B best ({best[0]:.2f}/{best[1]:.2f}/{best[2]:.2f})", best[3]),
    ]
    for name, acc in sorted(all_strategies, key=lambda x: -x[1]):
        marker = " ← current production" if name == "Equal 1/3 each" else ""
        print(f"  {name:<38} {acc:.3f}{marker}")

    gain = best[3] - equal_acc
    print(f"\n  Best gain over equal-weight: +{gain:.3f} ({gain*100:.1f} pp)")

    # ── Plot ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ensemble Weighting Graph", fontsize=14, fontweight="bold")

    # Left: bar chart
    bar_labels = ["LR", "GBM", "XGB", "Equal\n(1/3 each)", "Option A\n(acc-prop)",
                  f"Option B best\n({best[0]:.2f}/{best[1]:.2f}/{best[2]:.2f})"]
    bar_accs   = [lr_acc, gbm_acc, xgb_acc, equal_acc, optA_acc, best[3]]
    bar_colors = ["steelblue", "steelblue", "steelblue", "orange", "seagreen", "crimson"]
    bars = ax1.bar(range(len(bar_labels)), bar_accs, color=bar_colors, alpha=0.85, edgecolor="white")
    ax1.set_xticks(range(len(bar_labels)))
    ax1.set_xticklabels(bar_labels, fontsize=8)
    ax1.set_ylabel("2025 Holdout Accuracy")
    ymin = min(bar_accs) - 0.005
    ymax = max(bar_accs) + 0.005
    ax1.set_ylim(ymin, ymax)
    ax1.axhline(equal_acc, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Current production")
    for bar, acc in zip(bars, bar_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_title("Strategy Accuracy Comparison (2025 Holdout)")
    ax1.legend(fontsize=8)

    # Right: heatmap — w_lr vs w_gbm at the best w_xgb
    best_xgb = best[2]
    subset   = [(r[0], r[1], r[3]) for r in grid_results if abs(r[2] - best_xgb) < 0.026]
    lr_vals  = sorted(set(r[0] for r in subset))
    gbm_vals = sorted(set(r[1] for r in subset))
    lookup   = {(r[0], r[1]): r[2] for r in subset}
    heat_arr = np.array([[lookup.get((lr_v, gbm_v), np.nan)
                          for gbm_v in gbm_vals] for lr_v in lr_vals])
    im = ax2.imshow(heat_arr, aspect="auto", origin="lower", cmap="RdYlGn",
                    vmin=np.nanmin(heat_arr), vmax=np.nanmax(heat_arr))
    ax2.set_xticks(range(len(gbm_vals)))
    ax2.set_xticklabels([f"{v:.2f}" for v in gbm_vals], fontsize=7, rotation=45)
    ax2.set_yticks(range(len(lr_vals)))
    ax2.set_yticklabels([f"{v:.2f}" for v in lr_vals], fontsize=7)
    ax2.set_xlabel(f"w_GBM   (w_XGB fixed ≈ {best_xgb:.2f})")
    ax2.set_ylabel("w_LR")
    ax2.set_title("Option B Grid Search — Accuracy Heatmap")
    plt.colorbar(im, ax=ax2, label="Accuracy", fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(_SCRIPTS_DIR, "ensemble_weights_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    print("Open the PNG to review, then send a follow-up plan to implement best weights.")


if __name__ == "__main__":
    main()
