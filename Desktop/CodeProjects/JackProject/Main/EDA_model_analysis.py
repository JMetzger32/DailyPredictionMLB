"""
EDA_model_analysis.py — Three targeted EDA analyses for model improvement.

  1. Era Mismatch  — does older training data hurt? Compare training windows on 2025.
  2. Confidence Calibration — does high confidence actually predict higher accuracy?
  3. Feature Reduction + SP ERA Sensitivity — remove noisy features, tune ERA weight.

Run:  python3 EDA_model_analysis.py
Output: plots saved to EDA_2/  +  printed summary tables
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

from MLBModel import (
    load_data, build_team_game_log, compute_rolling_team_features,
    merge_sp_stats, merge_bullpen_era, assemble_features, FEATURE_COLS
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DB_PATH        = "mlb_allseasons.db"
LOG_PATH       = "predictions_log.json"
ARTIFACTS_PATH = "mlb_model_artifacts.pkl"
OUT_DIR        = "EDA_2"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared data loader
# ---------------------------------------------------------------------------
def load_model_df():
    print("Building feature matrix from DB (this takes ~60s)...")
    df, pitcher_stats, bullpen_stats = load_data(DB_PATH)
    tgl = build_team_game_log(df)
    tgl = compute_rolling_team_features(tgl)
    tgl = merge_sp_stats(tgl, pitcher_stats)
    tgl = merge_bullpen_era(tgl, bullpen_stats)
    model_df = assemble_features(df, tgl)
    model_df = model_df.dropna(subset=FEATURE_COLS)
    print(f"  {len(model_df)} games with complete features "
          f"({model_df['season'].min():.0f}–{model_df['season'].max():.0f})")
    return model_df


def train_eval(X_train, y_train, X_test, y_test,
               feats=None, sample_weight=None, C=0.5):
    """Train LR+GBM ensemble, return metrics dict."""
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_train)
    Xte_sc = sc.transform(X_test)

    lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
    lr.fit(Xtr_sc, y_train, sample_weight=sample_weight)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gb.fit(X_train, y_train, sample_weight=sample_weight)

    lr_prob  = lr.predict_proba(Xte_sc)[:, 1]
    gb_prob  = gb.predict_proba(X_test)[:, 1]
    ens_prob = (lr_prob + gb_prob) / 2
    preds    = (ens_prob > 0.5).astype(int)

    return {
        "lr":      lr,
        "gb":      gb,
        "scaler":  sc,
        "lr_prob":  lr_prob,
        "gb_prob":  gb_prob,
        "ens_prob": ens_prob,
        "accuracy": accuracy_score(y_test, preds),
        "auc":      roc_auc_score(y_test, ens_prob),
        "brier":    brier_score_loss(y_test, ens_prob),
        "logloss":  log_loss(y_test, ens_prob),
    }


# ===========================================================================
# SECTION 1: ERA MISMATCH
# ===========================================================================
def section1_era_mismatch(model_df):
    print("\n" + "="*60)
    print("SECTION 1: Training Data Era Mismatch")
    print("="*60)

    feats = FEATURE_COLS
    test_df  = model_df[model_df["season"] == 2025]
    X_test   = test_df[feats].values
    y_test   = test_df["home_win"].values
    print(f"  Test set (2025): {len(test_df)} games")

    # ── 1A. Feature distribution drift by season ─────────────────────────
    print("\n[1A] Feature drift across seasons...")
    seasonal_means = model_df.groupby("season")[feats].mean()

    # Z-score each feature's means across seasons so scales are comparable
    feat_std = seasonal_means.std(axis=0).replace(0, 1)
    z_scores = (seasonal_means - seasonal_means.mean()) / feat_std

    short_labels = [f.replace("diff_", "").replace("_", " ")[:22] for f in feats]

    fig, ax = plt.subplots(figsize=(12, max(8, len(feats) * 0.32)))
    sns.heatmap(
        z_scores.T,
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-2, vmax=2,
        xticklabels=[str(int(s)) for s in z_scores.index],
        yticklabels=short_labels,
        ax=ax, linewidths=0.3,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Mean Z-Score by Season\n(red = above avg, blue = below avg)", fontsize=12)
    ax.set_xlabel("Season")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/era_drift_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/era_drift_heatmap.png")

    # Flag drifting features (|2021 mean - 2024 mean| > 0.3 of std)
    drifting = []
    if 2021 in seasonal_means.index and 2024 in seasonal_means.index:
        for f in feats:
            raw_std = model_df[f].std()
            drift   = abs(seasonal_means.loc[2021, f] - seasonal_means.loc[2024, f])
            if raw_std > 0 and drift / raw_std > 0.30:
                drifting.append((f, round(drift / raw_std, 3)))
    if drifting:
        print(f"\n  Features that drifted most 2021→2024 (|Δ|/σ > 0.30):")
        for f, d in sorted(drifting, key=lambda x: -x[1]):
            print(f"    {f:<40s}  drift/σ = {d}")
    else:
        print("  No strongly drifting features found (all < 0.30 σ).")

    # ── 1B. Accuracy by training window ──────────────────────────────────
    print("\n[1B] Accuracy by training window on 2025 holdout...")

    windows = [
        ("2021–2024 (current)",  [2021, 2022, 2023, 2024], None),
        ("2022–2024",            [2022, 2023, 2024],         None),
        ("2023–2024",            [2023, 2024],               None),
        ("2024 only",            [2024],                     None),
        ("2021–2024 recency-wtd", [2021, 2022, 2023, 2024],  {2021: 1.0, 2022: 1.1, 2023: 1.3, 2024: 1.6}),
    ]

    results = []
    for label, seasons, wt_map in windows:
        tr = model_df[model_df["season"].isin(seasons)]
        X_tr = tr[feats].values
        y_tr = tr["home_win"].values
        sw   = tr["season"].map(wt_map).values if wt_map else None
        m    = train_eval(X_tr, y_tr, X_test, y_test, sample_weight=sw)
        results.append({
            "Training window": label,
            "n_train": len(tr),
            "Accuracy": round(m["accuracy"] * 100, 2),
            "AUC":      round(m["auc"], 4),
            "Brier":    round(m["brier"], 4),
            "LogLoss":  round(m["logloss"], 4),
        })
        print(f"  {label:<30s}  n={len(tr):5d}  acc={m['accuracy']:.4f}  "
              f"auc={m['auc']:.4f}  brier={m['brier']:.4f}")

    res_df = pd.DataFrame(results)
    print(f"\n{res_df.to_string(index=False)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results))]
    labels = [r["Training window"] for r in results]

    axes[0].bar(range(len(results)), [r["Accuracy"] for r in results], color=colors)
    axes[0].set_xticks(range(len(results)))
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel("Accuracy %")
    axes[0].set_title("Accuracy on 2025 by Training Window")
    baseline = 100 * y_test.mean()
    axes[0].axhline(max(baseline, 100 - baseline), color="red", linestyle="--",
                    lw=1, label=f"Always-home baseline ({max(baseline, 100-baseline):.1f}%)")
    axes[0].legend(fontsize=7)
    mn = min(r["Accuracy"] for r in results)
    axes[0].set_ylim(mn - 1.5, max(r["Accuracy"] for r in results) + 1.5)

    axes[1].bar(range(len(results)), [r["AUC"] for r in results], color=colors)
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC on 2025 by Training Window")
    axes[1].set_ylim(0.50, 0.65)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/era_accuracy_by_window.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/era_accuracy_by_window.png")

    # ── 1C. Home win rate trend ────────────────────────────────────────
    print("\n[1C] Home win rate by season...")
    seasonal = model_df.groupby("season").agg(
        games=("home_win", "count"),
        home_win_rate=("home_win", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(seasonal["season"], seasonal["games"], color="#3498db")
    axes[0].set_title("Games per Season (feature-complete games)")
    axes[0].set_ylabel("# Games")
    for _, row in seasonal.iterrows():
        axes[0].text(row["season"], row["games"] + 5, str(int(row["games"])),
                     ha="center", fontsize=9)

    axes[1].plot(seasonal["season"], seasonal["home_win_rate"] * 100,
                 "o-", color="#e74c3c", ms=8, lw=2)
    axes[1].axhline(50, color="gray", linestyle="--", lw=1, label="50%")
    axes[1].set_title("Home Win Rate by Season")
    axes[1].set_ylabel("Home Win %")
    axes[1].set_ylim(46, 58)
    for _, row in seasonal.iterrows():
        axes[1].text(row["season"], row["home_win_rate"] * 100 + 0.3,
                     f"{row['home_win_rate']*100:.1f}%", ha="center", fontsize=9)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/home_win_rate_by_season.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/home_win_rate_by_season.png")

    return results  # carry forward best window info


# ===========================================================================
# SECTION 2: CONFIDENCE CALIBRATION
# ===========================================================================
def section2_confidence_calibration(model_df):
    print("\n" + "="*60)
    print("SECTION 2: Accuracy by Confidence Bucket")
    print("="*60)

    feats   = FEATURE_COLS
    test_df = model_df[model_df["season"] == 2025]
    X_test  = test_df[feats].values
    y_test  = test_df["home_win"].values

    # Use full 2021-2024 model (baseline window)
    train_df = model_df[model_df["season"].isin([2021, 2022, 2023, 2024])]
    m        = train_eval(train_df[feats].values, train_df["home_win"].values, X_test, y_test)

    # ── 2A. Calibration on 2025 holdout ──────────────────────────────────
    print("\n[2A] Calibration on 2025 holdout...")
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    bin_labels = ["50–55%", "55–60%", "60–65%", "65–70%", "70%+"]

    holdout_rows = []
    for (lo, hi), label in zip(bins, bin_labels):
        # Use probability of predicted winner (i.e. max(p, 1-p))
        conf  = np.maximum(m["ens_prob"], 1 - m["ens_prob"])
        mask  = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            holdout_rows.append({"Bucket": label, "n": 0, "mean_pred": None, "actual_acc": None})
            continue
        pred_winners = (m["ens_prob"][mask] > 0.5).astype(int)
        actual_acc   = (pred_winners == y_test[mask]).mean()
        mean_pred    = conf[mask].mean()
        holdout_rows.append({
            "Bucket":     label,
            "n":          int(mask.sum()),
            "mean_pred":  round(mean_pred * 100, 1),
            "actual_acc": round(actual_acc * 100, 1),
        })
        print(f"  {label}: n={mask.sum():4d}  mean_confidence={mean_pred*100:.1f}%  "
              f"actual_accuracy={actual_acc*100:.1f}%")

    ho_df = pd.DataFrame(holdout_rows)

    # ── 2B. Calibration on live predictions_log.json ─────────────────────
    print("\n[2B] Calibration on live predictions_log.json...")
    live_rows = []
    try:
        with open(LOG_PATH) as f:
            log = json.load(f)
        entries = [
            e for day in log.values() for e in day
            if e.get("game_type") != "S"
            and e.get("correct") is not None
            and e.get("home_win_prob") is not None
        ]
        print(f"  Found {len(entries)} resolved regular-season predictions")

        for (lo, hi), label in zip(bins, bin_labels):
            bucket = []
            for e in entries:
                conf = max(e.get("home_win_prob", 0.5), e.get("away_win_prob", 0.5))
                if lo <= conf < hi:
                    bucket.append(e.get("correct"))
            if not bucket:
                live_rows.append({"Bucket": label, "n": 0, "actual_acc": None})
                continue
            acc = sum(bucket) / len(bucket)
            live_rows.append({
                "Bucket":     label,
                "n":          len(bucket),
                "actual_acc": round(acc * 100, 1),
            })
            print(f"  {label}: n={len(bucket):4d}  actual_accuracy={acc*100:.1f}%")
    except FileNotFoundError:
        print(f"  {LOG_PATH} not found — skipping live calibration")
        live_rows = [{"Bucket": lb, "n": 0, "actual_acc": None} for lb in bin_labels]
    live_df = pd.DataFrame(live_rows)

    # ── 2C. Home/away prediction bias ────────────────────────────────────
    print("\n[2C] Home vs Away pick accuracy on live data...")
    try:
        home_picks = [e for e in entries if e.get("predicted_winner") == "Home" and e.get("correct") is not None]
        away_picks = [e for e in entries if e.get("predicted_winner") == "Away" and e.get("correct") is not None]
        home_acc   = sum(e["correct"] for e in home_picks) / len(home_picks) if home_picks else None
        away_acc   = sum(e["correct"] for e in away_picks) / len(away_picks) if away_picks else None
        print(f"  Picks on Home: n={len(home_picks)}  acc={home_acc*100:.1f}%" if home_acc else "  No Home picks")
        print(f"  Picks on Away: n={len(away_picks)}  acc={away_acc*100:.1f}%" if away_acc else "  No Away picks")
    except Exception:
        home_acc = away_acc = None

    # ── Plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    valid_ho = ho_df.dropna(subset=["mean_pred", "actual_acc"])
    valid_li = live_df.dropna(subset=["actual_acc"])
    ax1.plot([50, 72], [50, 72], "--", color="gray", lw=1.5, label="Perfect calibration", zorder=1)
    if not valid_ho.empty:
        ax1.plot(valid_ho["mean_pred"], valid_ho["actual_acc"],
                 "o-", color="#3498db", ms=8, lw=2, label=f"2025 holdout (n={len(test_df)})")
        for _, row in valid_ho.iterrows():
            ax1.annotate(str(row["n"]), (row["mean_pred"], row["actual_acc"]),
                         textcoords="offset points", xytext=(4, 3), fontsize=7, color="#3498db")
    if not valid_li.empty and len(valid_li) >= 2:
        buckets_mid = {"50–55%": 52.5, "55–60%": 57.5, "60–65%": 62.5, "65–70%": 67.5, "70%+": 72.5}
        valid_li["mid"] = valid_li["Bucket"].map(buckets_mid)
        ax1.plot(valid_li["mid"], valid_li["actual_acc"],
                 "s-", color="#e74c3c", ms=8, lw=2, label=f"Live log (n={sum(live_df['n'])})")
        for _, row in valid_li.iterrows():
            ax1.annotate(str(row["n"]), (row["mid"], row["actual_acc"]),
                         textcoords="offset points", xytext=(4, -9), fontsize=7, color="#e74c3c")
    ax1.set_xlabel("Model confidence (%)")
    ax1.set_ylabel("Actual accuracy (%)")
    ax1.set_title("Reliability Diagram")
    ax1.set_xlim(49, 75)
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(gs[1])
    x      = np.arange(len(bin_labels))
    width  = 0.35
    ho_acc = [r["actual_acc"] or 0 for r in holdout_rows]
    li_acc = [r["actual_acc"] or 0 for r in live_rows]
    b1 = ax2.bar(x - width/2, ho_acc, width, label="2025 holdout", color="#3498db", alpha=0.8)
    b2 = ax2.bar(x + width/2, li_acc, width, label="Live log",     color="#e74c3c", alpha=0.8)
    ax2.axhline(50, color="gray", linestyle="--", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=20, fontsize=8)
    ax2.set_ylabel("Actual accuracy (%)")
    ax2.set_title("Accuracy by Confidence Bucket")
    ax2.set_ylim(35, 75)
    ax2.legend(fontsize=8)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

    ax3 = fig.add_subplot(gs[2])
    if home_acc is not None and away_acc is not None:
        bars = ax3.bar(["Picks Home", "Picks Away"],
                       [home_acc * 100, away_acc * 100],
                       color=["#2ecc71", "#e67e22"])
        ax3.axhline(50, color="gray", linestyle="--", lw=1)
        ax3.set_ylabel("Accuracy %")
        ax3.set_title(f"Home vs Away Pick Accuracy\n(live log)")
        ax3.set_ylim(35, 70)
        for bar, n in zip(bars, [len(home_picks), len(away_picks)]):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No live data", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=12)
        ax3.set_title("Home vs Away Pick Accuracy")

    plt.suptitle("Section 2: Confidence Calibration", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confidence_calibration.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUT_DIR}/confidence_calibration.png")


# ===========================================================================
# SECTION 3: FEATURE REDUCTION + SP ERA SENSITIVITY
# ===========================================================================
def section3_feature_analysis(model_df):
    print("\n" + "="*60)
    print("SECTION 3: Feature Reduction + SP ERA Sensitivity")
    print("="*60)

    feats     = FEATURE_COLS
    test_df   = model_df[model_df["season"] == 2025]
    y_test    = test_df["home_win"].values
    train_df  = model_df[model_df["season"].isin([2021, 2022, 2023, 2024])]
    y_train   = train_df["home_win"].values

    # ── 3A. Baseline feature importances ─────────────────────────────────
    print("\n[3A] Baseline feature importances on 2025 holdout...")
    m_base = train_eval(train_df[feats].values, y_train, test_df[feats].values, y_test)
    gb_imp = pd.Series(m_base["gb"].feature_importances_, index=feats).sort_values(ascending=False)

    print(f"\n  Baseline accuracy: {m_base['accuracy']*100:.2f}%  AUC: {m_base['auc']:.4f}")
    print("\n  Feature importances:")
    for feat, imp_val in gb_imp.items():
        flag = " ← LOW (<2%)" if imp_val < 0.02 else ""
        print(f"    {feat:<42s}  {imp_val*100:5.2f}%{flag}")

    low_feats = gb_imp[gb_imp < 0.01].index.tolist()
    med_feats = gb_imp[gb_imp < 0.02].index.tolist()
    top10     = gb_imp.index[:10].tolist()
    top15     = gb_imp.index[:15].tolist()
    top20     = gb_imp.index[:20].tolist()

    fig, ax = plt.subplots(figsize=(10, max(6, len(feats) * 0.38)))
    colors = ["#e74c3c" if v < 0.02 else "#3498db" for v in gb_imp.values]
    short  = [f.replace("diff_", "").replace("_", " ")[:30] for f in gb_imp.index]
    ax.barh(short, gb_imp.values * 100, color=colors)
    ax.axvline(2, color="red", linestyle="--", lw=1, label="2% threshold")
    ax.set_xlabel("GBM Importance (%)")
    ax.set_title("Feature Importance — red bars (<2%) are low-signal candidates")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance_baseline.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/feature_importance_baseline.png")

    # ── 3B. Feature subset experiments ───────────────────────────────────
    print("\n[3B] Feature subset experiments...")
    subsets = [
        (f"All 28 (baseline)",             feats),
        (f"Drop <1% importance ({len(feats)-len(low_feats)} feats)",  [f for f in feats if f not in low_feats]),
        (f"Drop <2% importance ({len(feats)-len(med_feats)} feats)",  [f for f in feats if f not in med_feats]),
        (f"Top 20",                         top20),
        (f"Top 15",                         top15),
        (f"Top 10",                         top10),
    ]

    subset_results = []
    for label, sub_feats in subsets:
        if len(sub_feats) == 0:
            continue
        m = train_eval(
            train_df[sub_feats].values, y_train,
            test_df[sub_feats].values, y_test
        )
        subset_results.append({
            "Feature set":   label,
            "n_features":    len(sub_feats),
            "Accuracy %":    round(m["accuracy"] * 100, 2),
            "AUC":           round(m["auc"], 4),
            "Brier":         round(m["brier"], 4),
        })
        print(f"  {label:<38s}  n={len(sub_feats):2d}  "
              f"acc={m['accuracy']*100:.2f}%  auc={m['auc']:.4f}")

    sub_df = pd.DataFrame(subset_results)
    print(f"\n{sub_df.to_string(index=False)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(subset_results))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(subset_results))]

    axes[0].bar(x, sub_df["Accuracy %"], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sub_df["Feature set"], rotation=20, ha="right", fontsize=7)
    axes[0].set_ylabel("Accuracy %")
    axes[0].set_title("Accuracy by Feature Subset")
    mn = sub_df["Accuracy %"].min()
    axes[0].set_ylim(mn - 1, sub_df["Accuracy %"].max() + 1)
    for bar, val in zip(axes[0].patches, sub_df["Accuracy %"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}%", ha="center", fontsize=7)

    axes[1].bar(x, sub_df["AUC"], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub_df["Feature set"], rotation=20, ha="right", fontsize=7)
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC by Feature Subset")
    axes[1].set_ylim(sub_df["AUC"].min() - 0.005, sub_df["AUC"].max() + 0.005)

    plt.suptitle("Section 3B: Feature Subset Experiments", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_subset_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/feature_subset_comparison.png")

    # ── 3C. SP ERA multiplier sensitivity ────────────────────────────────
    print("\n[3C] SP ERA weight sensitivity (LR only)...")
    era_idx = feats.index("diff_sp_era") if "diff_sp_era" in feats else None
    if era_idx is None:
        print("  diff_sp_era not in FEATURE_COLS — skipping")
        return

    # Train the LR model once on full feature set
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(train_df[feats].values)
    X_te_sc = sc.transform(test_df[feats].values)
    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_train)

    multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    era_results = []
    base_coef   = lr.coef_[0][era_idx]
    print(f"  diff_sp_era base LR coef: {base_coef:.4f}")

    for mult in multipliers:
        lr_adj = LogisticRegression.__new__(LogisticRegression)
        lr_adj.__dict__.update(lr.__dict__)
        import copy
        lr_adj.coef_ = copy.deepcopy(lr.coef_)
        lr_adj.coef_[0][era_idx] = base_coef * mult

        probs = lr_adj.predict_proba(X_te_sc)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc   = accuracy_score(y_test, preds)
        auc   = roc_auc_score(y_test, probs)
        era_results.append({
            "multiplier": mult,
            "accuracy":   round(acc * 100, 2),
            "auc":        round(auc, 4),
        })
        flag = "  ← baseline" if mult == 1.0 else ""
        print(f"  SP ERA ×{mult:.2f}:  acc={acc*100:.2f}%  auc={auc:.4f}{flag}")

    best = max(era_results, key=lambda x: x["accuracy"])
    print(f"\n  Best accuracy at SP ERA multiplier = {best['multiplier']}× ({best['accuracy']}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mults = [r["multiplier"] for r in era_results]

    axes[0].plot(mults, [r["accuracy"] for r in era_results], "o-", color="#3498db", ms=8, lw=2)
    axes[0].axvline(1.0, color="red", linestyle="--", lw=1, label="Current weight")
    axes[0].axvline(best["multiplier"], color="green", linestyle=":", lw=1.5,
                    label=f"Best ({best['multiplier']}×)")
    axes[0].set_xlabel("SP ERA multiplier")
    axes[0].set_ylabel("Accuracy %")
    axes[0].set_title("LR Accuracy vs SP ERA Weight")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(mults)

    axes[1].plot(mults, [r["auc"] for r in era_results], "o-", color="#e74c3c", ms=8, lw=2)
    axes[1].axvline(1.0, color="red", linestyle="--", lw=1, label="Current weight")
    axes[1].set_xlabel("SP ERA multiplier")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("LR AUC vs SP ERA Weight")
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(mults)

    plt.suptitle("Section 3C: SP ERA Sensitivity (LR model, 2025 holdout)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sp_era_multiplier_curve.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/sp_era_multiplier_curve.png")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MLB Model — EDA Analysis (Era Mismatch, Calibration, Features)")
    print(f"Output directory: {OUT_DIR}/")
    print("=" * 60)

    model_df = load_model_df()

    section1_era_mismatch(model_df)
    section2_confidence_calibration(model_df)
    section3_feature_analysis(model_df)

    print("\n" + "=" * 60)
    print("DONE. Plots saved to EDA_2/:")
    plots = [
        "era_drift_heatmap.png        — feature means by season (drift check)",
        "era_accuracy_by_window.png   — accuracy/AUC by training window",
        "home_win_rate_by_season.png  — home win % trend 2021-2025",
        "confidence_calibration.png   — reliability diagram + bucket accuracy + home/away bias",
        "feature_importance_baseline.png — GBM importances, flags <2% features",
        "feature_subset_comparison.png   — accuracy when dropping low-signal features",
        "sp_era_multiplier_curve.png     — accuracy vs SP ERA weight multiplier",
    ]
    for p in plots:
        print(f"  {p}")
    print("=" * 60)
