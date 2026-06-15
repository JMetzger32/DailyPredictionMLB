import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, log_loss, brier_score_loss,
                             roc_auc_score, classification_report, confusion_matrix)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MLBMODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # Main/
_PROJECT_ROOT = os.path.dirname(_MLBMODEL_DIR)              # JackProject/
DB_PATH = os.path.join(_PROJECT_ROOT, "Databases_and_logs", "mlb_allseasons.db")
_ARTIFACTS_SAVE_PATH = os.path.join(_PROJECT_ROOT, "updates", "mlb_model_artifacts.pkl")
RANDOM_STATE = 42
ROLLING_WINDOW = 30
ROLLING_WINDOW_SHORT = 10
SP_MIN_STARTS = 3
PYTH_EXPONENT = 1.83

FEATURE_COLS = [
    # Home park environment (not differenced — it's a property of the venue)
    "home_park_factor",
    # Team quality
    "diff_pyth_win_pct",
    "diff_season_win_pct",
    # Offensive stats — OBP captures contact+walks; ISO captures pure extra-base power (SLG dropped: ISO = SLG-AVG → collinear)
    "diff_roll30_obp",            # on-base percentage (30-game rolling avg) — most important batting stat
    "diff_roll30_iso",            # isolated power (SLG - AVG) — pure extra-base hit ability; SLG excluded (VIF)
    "diff_roll10_runs_scored",    # recent offensive form (10-game)
    "diff_roll10_homeruns",       # recent power surge (10-game)
    # Defensive stats — normalized rate stats replace raw counts (PA-volume distortion removed)
    "diff_roll30_opp_whip",       # (opp_walks + opp_hits) / IP allowed — normalized contact+walk rate
    "diff_roll30_opp_hr_per9",    # (opp_homeruns / IP) * 9 — normalized HR rate allowed
    "diff_roll30_opp_strikeouts", # full-staff pitching K rate (kept: already a counting stat per game, stable)
    # Short-window recent form
    "diff_roll10_win_pct",
    # Bullpen — exponential decay replaces raw IP sum and pitchers-used count
    "diff_bullpen_era",
    "diff_roll7_bullpen_fatigue", # exp-decayed bullpen IP over 7 days (half-life=3d); recent work penalized more
    # Schedule / context
    "diff_rest_days",             # home rest - away rest (days since last game; 3+ days is measurable advantage)
    # SP handedness — separate features (not differenced) so model learns home/away LHP effects independently
    "home_sp_is_lhp",             # 1 if home starter is LHP, 0 if RHP
    "away_sp_is_lhp",             # 1 if away starter is LHP, 0 if RHP
    # Starting pitcher stats — 5 features selected by EDA (2021-2025, 9,510 games)
    # Ranked by GBM importance: ERA 25.3%, IP/GS 8.0%, WHIP 7.7%, K/BB 6.7%, xFIP 5.4%
    "diff_sp_era",     # season ERA — strongest single signal; r=-0.186 vs home win
    "diff_sp_ip_gs",   # innings per start — durability/quality; non-linear signal (low Pearson, high GBM)
    "diff_sp_whip",    # WHIP — contact+walk rate, complements ERA
    "diff_sp_k_bb",    # K/BB ratio — command signal, independent of ERA/WHIP
    "diff_sp_xfip",    # xFIP — defense-independent ERA estimator; regresses HR luck
]

# Park run factors per home ballpark (Retrosheet team code -> multi-year average, 1.0 = league avg)
# Source: Baseball Reference park factors (2021-2024 average)
PARK_FACTORS = {
    "COL": 1.23,  # Coors Field
    "CIN": 1.08,  # Great American Ball Park
    "BOS": 1.06,  # Fenway Park
    "NYA": 1.06,  # Yankee Stadium
    "CHN": 1.05,  # Wrigley Field
    "PHI": 1.05,  # Citizens Bank Park
    "TEX": 1.04,  # Globe Life Field
    "ATL": 1.03,  # Truist Park
    "ARI": 1.03,  # Chase Field
    "BAL": 1.02,  # Camden Yards
    "MIL": 1.02,  # American Family Field
    "TOR": 1.01,  # Rogers Centre
    "MIN": 1.01,  # Target Field
    "WAS": 1.00,  # Nationals Park
    "DET": 1.00,  # Comerica Park
    "SLN": 0.99,  # Busch Stadium
    "PIT": 0.99,  # PNC Park
    "KCA": 0.99,  # Kauffman Stadium
    "CHA": 0.99,  # Guaranteed Rate Field
    "CLE": 0.98,  # Progressive Field
    "NYN": 0.98,  # Citi Field
    "HOU": 0.98,  # Minute Maid Park
    "ANA": 0.97,  # Angel Stadium
    "LAN": 0.97,  # Dodger Stadium
    "SEA": 0.97,  # T-Mobile Park
    "TBA": 0.96,  # Tropicana Field
    "SFN": 0.96,  # Oracle Park
    "MIA": 0.95,  # loanDepot Park
    "ATH": 0.95,  # Oakland Coliseum / Sacramento
    "SDN": 0.93,  # Petco Park
}


# ===========================================================================
# Section 1: Data Loading
# ===========================================================================
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM games WHERE season >= 2021", conn)
    pitcher_stats = pd.read_sql_query("SELECT * FROM pitcher_stats", conn)
    try:
        bullpen_stats = pd.read_sql_query("SELECT * FROM team_bullpen_stats", conn)
    except Exception:
        bullpen_stats = pd.DataFrame()  # table doesn't exist yet — will use fallback
    conn.close()
    df["home_win"] = (df["home_score"] > df["visitor_score"]).astype(int)
    df = df.sort_values(["date", "doubleheader", "game_id"]).reset_index(drop=True)
    return df, pitcher_stats, bullpen_stats


# ===========================================================================
# Section 2: Team Game Log (long format — one row per team per game)
# ===========================================================================
def build_team_game_log(df):
    home = pd.DataFrame({
        "game_id": df["game_id"],
        "date": df["date"],
        "season": df["season"],
        "team": df["home_team"],
        "opponent": df["visiting_team"],
        "is_home": 1,
        "win": (df["home_score"] > df["visitor_score"]).astype(int),
        "runs_scored": df["home_score"],
        "runs_allowed": df["visitor_score"],
        "hits": df["home_hits"],
        "walks": df["home_walks"],
        "errors": df["home_errors"],
        "homeruns": df["home_homeruns"],
        "opp_hits": df["visitor_hits"],
        "opp_walks": df["visitor_walks"],
        "opp_homeruns": df["visitor_homeruns"],
        "opp_strikeouts": df["visitor_strikeouts"],
        "earned_runs_allowed": df["home_team_earned_runs"],
        "pitchers_used": df["home_pitchers_used"],
        "starting_pitcher_id": df["home_starting_pitcher_id"],
        # Batting detail for rate stats
        "at_bats":     df["home_at_bats"],
        "doubles":     df["home_doubles"],
        "triples":     df["home_triples"],
        "hit_by_pitch":df["home_hit_by_pitch"],
        "sac_flies":   df["home_sac_flies"],
        # Estimated total IP for this team's pitching staff (length_outs / 2 / 3)
        "total_ip":    df.get("length_outs", pd.Series(54, index=df.index)) / 6,
    })
    away = pd.DataFrame({
        "game_id": df["game_id"],
        "date": df["date"],
        "season": df["season"],
        "team": df["visiting_team"],
        "opponent": df["home_team"],
        "is_home": 0,
        "win": (df["visitor_score"] > df["home_score"]).astype(int),
        "runs_scored": df["visitor_score"],
        "runs_allowed": df["home_score"],
        "hits": df["visitor_hits"],
        "walks": df["visitor_walks"],
        "errors": df["visitor_errors"],
        "homeruns": df["visitor_homeruns"],
        "opp_hits": df["home_hits"],
        "opp_walks": df["home_walks"],
        "opp_homeruns": df["home_homeruns"],
        "opp_strikeouts": df["home_strikeouts"],
        "earned_runs_allowed": df["visitor_team_earned_runs"],
        "pitchers_used": df["visitor_pitchers_used"],
        "starting_pitcher_id": df["visitor_starting_pitcher_id"],
        # Batting detail for rate stats
        "at_bats":     df["visitor_at_bats"],
        "doubles":     df["visitor_doubles"],
        "triples":     df["visitor_triples"],
        "hit_by_pitch":df["visitor_hit_by_pitch"],
        "sac_flies":   df["visitor_sac_flies"],
        # Estimated total IP for this team's pitching staff
        "total_ip":    df.get("length_outs", pd.Series(54, index=df.index)) / 6,
    })
    tgl = pd.concat([home, away], ignore_index=True)
    tgl = tgl.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    return tgl


# ===========================================================================
# Section 3: Rolling Team Features
# ===========================================================================
def _exp_decay_bullpen(group, half_life_days=3.0):
    """Exponentially decayed bullpen IP sum for each game row (pre-game signal).
    Games within 7 days of the current game are included, weighted by recency.
    Lambda = ln(2) / half_life so IP 3 days ago weighs half as much as yesterday."""
    lam   = np.log(2) / half_life_days
    dates = group["date_dt"].values
    ips   = group["bullpen_ip_game"].fillna(0.0).values
    result = np.zeros(len(dates))
    for i in range(1, len(dates)):
        for j in range(i - 1, -1, -1):
            days_ago = (dates[i] - dates[j]) / np.timedelta64(1, "D")
            if days_ago > 7:
                break
            result[i] += np.exp(-lam * days_ago) * ips[j]
    return pd.Series(result, index=group.index)


def compute_rolling_team_features(tgl):
    # ---- Date column needed early for rest days and bullpen decay ----
    tgl["date_dt"] = pd.to_datetime(tgl["date"], format="%Y%m%d")

    # ---- Season-to-date cumulative (shifted so pre-game) ----
    cum_cols = ["runs_scored", "runs_allowed", "win", "hits", "opp_hits",
                "walks", "opp_walks", "errors", "homeruns", "opp_homeruns"]
    for col in cum_cols:
        tgl[f"cum_{col}"] = tgl.groupby(["team", "season"])[col].transform(
            lambda x: x.expanding().sum().shift(1)
        )
    tgl["cum_games"] = tgl.groupby(["team", "season"])["win"].transform(
        lambda x: x.expanding().count().shift(1)
    )

    # Pythagorean win % (season-to-date, pre-game)
    rs = tgl["cum_runs_scored"].clip(lower=1)
    ra = tgl["cum_runs_allowed"].clip(lower=1)
    tgl["pyth_win_pct"] = rs ** PYTH_EXPONENT / (rs ** PYTH_EXPONENT + ra ** PYTH_EXPONENT)
    tgl.loc[tgl["cum_games"].isna(), "pyth_win_pct"] = np.nan

    # Season win %
    tgl["season_win_pct"] = tgl["cum_win"] / tgl["cum_games"]

    # Fill cold-start (first game of season) with 0.500 / league avg
    tgl["pyth_win_pct"] = tgl["pyth_win_pct"].fillna(0.500)
    tgl["season_win_pct"] = tgl["season_win_pct"].fillna(0.500)

    # ---- Per-game offensive rate stats (OBP, SLG, ISO) ----
    ab  = tgl["at_bats"].fillna(0).clip(lower=1)
    hbp = tgl["hit_by_pitch"].fillna(0)
    sf  = tgl["sac_flies"].fillna(0)
    d   = tgl["doubles"].fillna(0)
    t   = tgl["triples"].fillna(0)
    h   = tgl["hits"]
    bb  = tgl["walks"]
    hr  = tgl["homeruns"]
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    tgl["obp_game"] = (h + bb + hbp) / (ab + bb + hbp + sf).clip(lower=1)
    # SLG retained for completeness but diff_roll30_slg excluded from FEATURE_COLS (VIF with ISO)
    tgl["slg_game"] = (h + d + 2 * t + 3 * hr) / ab
    # ISO = (D + 2T + 3HR) / AB — pure extra-base ability excluding singles
    tgl["iso_game"] = (d + 2 * t + 3 * hr) / ab

    # ---- Per-game defensive rate stats (normalized by IP to remove PA-volume distortion) ----
    safe_ip = tgl["total_ip"].fillna(9.0).clip(lower=1.0)
    # Opponent WHIP proxy: (opp_walks + opp_hits) / IP — lower is better pitching
    tgl["opp_whip_game"] = (tgl["opp_walks"].fillna(0) + tgl["opp_hits"].fillna(0)) / safe_ip
    # Opponent HR/9: normalized HR rate allowed
    tgl["opp_hr9_game"]  = (tgl["opp_homeruns"].fillna(0) / safe_ip) * 9.0

    # ---- 30-game rolling (crosses season boundaries) ----
    roll_cols = ["win", "runs_scored", "runs_allowed", "hits", "opp_hits",
                 "walks", "opp_walks", "errors", "homeruns", "opp_homeruns",
                 "opp_strikeouts", "obp_game", "slg_game", "iso_game",
                 "opp_whip_game", "opp_hr9_game"]
    for col in roll_cols:
        tgl[f"roll30_{col}"] = tgl.groupby("team")[col].transform(
            lambda x: x.rolling(ROLLING_WINDOW, min_periods=10).mean().shift(1)
        )
    # Rename rolling columns to clean names
    tgl.rename(columns={
        "roll30_obp_game":      "roll30_obp",
        "roll30_slg_game":      "roll30_slg",
        "roll30_iso_game":      "roll30_iso",
        "roll30_opp_whip_game": "roll30_opp_whip",
        "roll30_opp_hr9_game":  "roll30_opp_hr_per9",
    }, inplace=True)

    # ---- 10-game rolling for recent form ----
    tgl["roll10_win_pct"] = tgl.groupby("team")["win"].transform(
        lambda x: x.rolling(ROLLING_WINDOW_SHORT, min_periods=5).mean().shift(1)
    )
    tgl["roll10_homeruns"] = tgl.groupby("team")["homeruns"].transform(
        lambda x: x.rolling(ROLLING_WINDOW_SHORT, min_periods=5).mean().shift(1)
    )
    tgl["roll10_runs_scored"] = tgl.groupby("team")["runs_scored"].transform(
        lambda x: x.rolling(ROLLING_WINDOW_SHORT, min_periods=5).mean().shift(1)
    )

    # ---- Exponentially decayed bullpen fatigue (replaces raw IP sum + pitchers-used) ----
    # Bullpen IP per game = total IP minus estimated starter IP (~5.8 innings)
    STARTER_IP_EST = 5.8
    tgl["bullpen_ip_game"] = (tgl["total_ip"].fillna(9.0) - STARTER_IP_EST).clip(lower=0.0)
    # Apply exponential decay per team (sorted chronologically)
    tgl = tgl.sort_values(["team", "date_dt"]).copy()
    tgl["roll7_bullpen_fatigue"] = (
        tgl.groupby("team", group_keys=False)
           .apply(_exp_decay_bullpen)
    )

    # ---- Rest days (days since last game, pre-game) ----
    tgl["prev_game_date"] = tgl.groupby("team")["date_dt"].shift(1)
    tgl["rest_days"] = (tgl["date_dt"] - tgl["prev_game_date"]).dt.days.fillna(1).clip(lower=1, upper=7)

    # Fill any remaining NaNs in rolling features with league averages
    roll_feature_prefixes = ("roll30_", "roll10_", "roll7_")
    for col in tgl.columns:
        if any(col.startswith(p) for p in roll_feature_prefixes):
            tgl[col] = tgl[col].fillna(tgl[col].mean())

    # Sanity-clip rate stats to realistic ranges
    tgl["roll30_obp"]       = tgl["roll30_obp"].clip(0.200, 0.450)
    tgl["roll30_slg"]       = tgl["roll30_slg"].clip(0.200, 0.700)
    tgl["roll30_opp_whip"]  = tgl["roll30_opp_whip"].clip(0.50, 2.50)
    tgl["roll30_opp_hr_per9"] = tgl["roll30_opp_hr_per9"].clip(0.0, 3.0)

    return tgl


# ===========================================================================
# Section 4: Starting Pitcher Features (from Baseball Reference stats)
# ===========================================================================
def merge_sp_stats(tgl, pitcher_stats):
    """Merge real pitcher season stats (ERA, WHIP, xFIP, IP/GS, K/BB) into tgl."""
    has_xfip  = "xfip"  in pitcher_stats.columns
    has_siera = "siera" in pitcher_stats.columns

    stat_cols = ["era", "whip", "so9", "bb9", "hr9", "innings_pitched", "games_started"]
    if has_xfip:  stat_cols.append("xfip")
    if has_siera: stat_cols.append("siera")

    # Build a lookup: (retro_pitcher_id, season) -> stats
    ps_deduped = pitcher_stats.sort_values("games_started", ascending=False).drop_duplicates(
        subset=["retro_pitcher_id", "season"], keep="first"
    )
    sp_lookup = ps_deduped.set_index(["retro_pitcher_id", "season"])[stat_cols].to_dict("index")

    league_avg = pitcher_stats.groupby("season")[stat_cols].mean().to_dict("index")
    overall_avg = pitcher_stats[stat_cols].mean().to_dict()

    sp_era = []; sp_whip = []; sp_so9 = []; sp_bb9 = []; sp_hr9 = []
    sp_xfip = []; sp_siera = []; sp_ip_gs = []; sp_k_bb = []

    for _, row in tgl.iterrows():
        pid = row["starting_pitcher_id"]
        season = row["season"]
        key = (pid, season)

        if key in sp_lookup:
            stats = sp_lookup[key]
        else:
            prev_key = (pid, season - 1)
            if prev_key in sp_lookup:
                stats = sp_lookup[prev_key]
            else:
                stats = league_avg.get(season, overall_avg)

        sp_era.append(stats.get("era", overall_avg["era"]))
        sp_whip.append(stats.get("whip", overall_avg["whip"]))
        sp_so9.append(stats.get("so9", overall_avg["so9"]))
        sp_bb9.append(stats.get("bb9", overall_avg["bb9"]))
        sp_hr9.append(stats.get("hr9", overall_avg["hr9"]))
        sp_xfip.append(stats.get("xfip", stats.get("era", overall_avg["era"])))
        sp_siera.append(stats.get("siera", stats.get("era", overall_avg["era"])))

        # IP/Start: innings per start (durability signal)
        ip = stats.get("innings_pitched", overall_avg.get("innings_pitched", 0))
        gs = stats.get("games_started",   overall_avg.get("games_started",   1))
        sp_ip_gs.append(ip / gs if gs > 0 else overall_avg.get("innings_pitched", 0) / max(overall_avg.get("games_started", 1), 1))

        # K/BB ratio: strikeouts-per-9 / walks-per-9 (command quality)
        bb9 = stats.get("bb9", overall_avg["bb9"])
        so9 = stats.get("so9", overall_avg["so9"])
        sp_k_bb.append(so9 / bb9 if bb9 > 0.5 else so9 / 0.5)

    tgl["sp_era"]   = sp_era
    tgl["sp_whip"]  = sp_whip
    tgl["sp_xfip"]  = sp_xfip
    tgl["sp_siera"] = sp_siera
    tgl["sp_so9"]   = sp_so9
    tgl["sp_bb9"]   = sp_bb9
    tgl["sp_hr9"]   = sp_hr9
    tgl["sp_ip_gs"] = sp_ip_gs
    tgl["sp_k_bb"]  = sp_k_bb

    for col in ["sp_era", "sp_whip", "sp_xfip", "sp_siera", "sp_so9",
                "sp_bb9", "sp_hr9", "sp_ip_gs", "sp_k_bb"]:
        tgl[col] = pd.to_numeric(tgl[col], errors="coerce")
        tgl[col] = tgl[col].fillna(tgl[col].mean())

    return tgl


# ===========================================================================
# Section 4b: Bullpen ERA (team-season level, from team_bullpen_stats table)
# ===========================================================================
def merge_bullpen_era(tgl, bullpen_stats):
    """Merge team-season bullpen ERA into tgl. Falls back to 4.20 if data unavailable."""
    if bullpen_stats is None or len(bullpen_stats) == 0:
        tgl["bullpen_era"] = 4.20
        return tgl

    bp_lookup = bullpen_stats.set_index(["retro_team", "season"])["bullpen_era"].to_dict()
    tgl["bullpen_era"] = tgl.apply(
        lambda r: bp_lookup.get((r["team"], r["season"]),
                  bp_lookup.get((r["team"], r["season"] - 1), 4.20)),
        axis=1
    )
    return tgl


# ===========================================================================
# Section 5: Feature Assembly (differentials)
# ===========================================================================
def assemble_features(df, tgl):
    feature_map = {
        "pyth_win_pct":            "pyth_win_pct",
        "season_win_pct":          "season_win_pct",
        # Offensive rate stats (SLG excluded — collinear with ISO; VIF verified)
        "roll30_obp":              "roll30_obp",
        "roll30_iso":              "roll30_iso",
        "roll10_runs_scored":      "roll10_runs_scored",
        "roll10_homeruns":         "roll10_homeruns",
        # Defensive rate stats — normalized by IP (replaces raw opp_hits/opp_homeruns/errors)
        "roll30_opp_whip":         "roll30_opp_whip",
        "roll30_opp_hr_per9":      "roll30_opp_hr_per9",
        "roll30_opp_strikeouts":   "roll30_opp_strikeouts",
        # Recent form
        "roll10_win_pct":          "roll10_win_pct",
        # Bullpen — exp-decayed fatigue replaces raw IP sum and pitchers-used count
        "bullpen_era":             "bullpen_era",
        "roll7_bullpen_fatigue":   "roll7_bullpen_fatigue",
        "rest_days":               "rest_days",
        "sp_era":                  "sp_era",
        "sp_whip":                 "sp_whip",
        "sp_xfip":                 "sp_xfip",
        "sp_siera":                "sp_siera",
        "sp_so9":                  "sp_so9",
        "sp_bb9":                  "sp_bb9",
        "sp_hr9":                  "sp_hr9",
        "sp_ip_gs":                "sp_ip_gs",
        "sp_k_bb":                 "sp_k_bb",
    }

    tgl_cols = ["game_id", "is_home"] + list(feature_map.values())

    home_feats = tgl[tgl["is_home"] == 1][tgl_cols].copy()
    home_feats.columns = ["game_id", "is_home"] + [f"home_{k}" for k in feature_map.keys()]
    home_feats = home_feats.drop(columns=["is_home"])

    vis_feats = tgl[tgl["is_home"] == 0][tgl_cols].copy()
    vis_feats.columns = ["game_id", "is_home"] + [f"vis_{k}" for k in feature_map.keys()]
    vis_feats = vis_feats.drop(columns=["is_home"])

    model_df = df[["game_id", "season", "date", "home_team", "visiting_team",
                   "home_win", "home_score", "visitor_score"]].copy()
    # Run line target: did the home team cover -1.5?
    model_df["home_covers"] = ((model_df["home_score"] - model_df["visitor_score"]) > 1.5).astype("Int64")
    model_df = model_df.merge(home_feats, on="game_id", how="left")
    model_df = model_df.merge(vis_feats, on="game_id", how="left")

    # Compute differentials (home - visitor)
    for feat in feature_map.keys():
        model_df[f"diff_{feat}"] = model_df[f"home_{feat}"] - model_df[f"vis_{feat}"]

    # Park factor is NOT differenced — it's a property of the home ballpark
    model_df["home_park_factor"] = model_df["home_team"].map(PARK_FACTORS).fillna(1.0)

    # SP handedness — not available in historical Retrosheet data; default 0.
    # Feature is present so model accepts it at prediction time; real L/R values
    # kick in once live data has handedness populated.
    model_df["home_sp_is_lhp"] = 0
    model_df["away_sp_is_lhp"] = 0

    return model_df


# ===========================================================================
# Section 6: Cross-Validation (Leave-One-Season-Out)
# ===========================================================================
def cross_validate_loso(model_df, feature_cols):
    train_df = model_df[model_df["season"].between(2021, 2024)].dropna(subset=feature_cols)
    seasons = [2021, 2022, 2023, 2024]
    results = []

    for hold_out in seasons:
        train_mask = train_df["season"] != hold_out
        val_mask = train_df["season"] == hold_out

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "home_win"]
        X_val = train_df.loc[val_mask, feature_cols]
        y_val = train_df.loc[val_mask, "home_win"]

        # Logistic Regression
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_va_sc = scaler.transform(X_val)

        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_tr_sc, y_train)
        lr_probs = lr.predict_proba(X_va_sc)[:, 1]
        lr_preds = lr.predict(X_va_sc)

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        )
        gb.fit(X_train, y_train)
        gb_probs = gb.predict_proba(X_val)[:, 1]
        gb_preds = gb.predict(X_val)

        baseline_acc = max(y_val.mean(), 1 - y_val.mean())

        fold = {
            "season": hold_out,
            "n_val": len(y_val),
            "baseline_acc": baseline_acc,
            "lr_acc": accuracy_score(y_val, lr_preds),
            "lr_logloss": log_loss(y_val, lr_probs),
            "lr_auc": roc_auc_score(y_val, lr_probs),
            "gb_acc": accuracy_score(y_val, gb_preds),
            "gb_logloss": log_loss(y_val, gb_probs),
            "gb_auc": roc_auc_score(y_val, gb_probs),
        }
        results.append(fold)

        print(f"\n  Fold: hold out {hold_out}  (train={len(y_train)}, val={len(y_val)})")
        print(f"    Baseline (always home):   {fold['baseline_acc']:.3f}")
        print(f"    Logistic Regression:      Acc={fold['lr_acc']:.3f}  LogLoss={fold['lr_logloss']:.3f}  AUC={fold['lr_auc']:.3f}")
        print(f"    Gradient Boosting:        Acc={fold['gb_acc']:.3f}  LogLoss={fold['gb_logloss']:.3f}  AUC={fold['gb_auc']:.3f}")

    results_df = pd.DataFrame(results)
    print("\n  --- Average across folds ---")
    print(f"    Baseline:            {results_df['baseline_acc'].mean():.3f}")
    print(f"    Logistic Regression: Acc={results_df['lr_acc'].mean():.3f}  AUC={results_df['lr_auc'].mean():.3f}")
    print(f"    Gradient Boosting:   Acc={results_df['gb_acc'].mean():.3f}  AUC={results_df['gb_auc'].mean():.3f}")

    return results_df


# ===========================================================================
# Section 7: 2025 Holdout Evaluation
# ===========================================================================
def evaluate_holdout(model_df, feature_cols):
    train_df = model_df[model_df["season"].between(2021, 2024)].dropna(subset=feature_cols)
    test_df = model_df[model_df["season"] == 2025].dropna(subset=feature_cols)

    X_train = train_df[feature_cols]
    y_train = train_df["home_win"]
    X_test = test_df[feature_cols]
    y_test = test_df["home_win"]

    # Logistic Regression (try C=0.5 for stronger regularization)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_tr_sc, y_train)

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)

    # XGBoost (if installed)
    xgb = None
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
            verbosity=0
        )
        xgb.fit(X_train, y_train)

    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"\n  2025 holdout: {len(y_test)} games")
    print(f"  Home win rate: {y_test.mean():.3f}")
    print(f"  Baseline (always home): {baseline_acc:.3f}")

    candidates = [("Logistic Regression (C=0.5)", lr, X_te_sc),
                  ("Gradient Boosting", gb, X_test)]
    if xgb is not None:
        candidates.append(("XGBoost", xgb, X_test))

    best_acc = 0
    best_model = lr
    for name, model, X in candidates:
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_model = model
        print(f"\n  {name}:")
        print(f"    Accuracy:  {acc:.3f}")
        print(f"    Log Loss:  {log_loss(y_test, probs):.3f}")
        print(f"    Brier:     {brier_score_loss(y_test, probs):.3f}")
        print(f"    AUC:       {roc_auc_score(y_test, probs):.3f}")

    # Ensemble: average LR + best tree model probabilities
    tree_model = xgb if xgb is not None else gb
    tree_X = X_test
    lr_probs  = lr.predict_proba(X_te_sc)[:, 1]
    tree_probs = tree_model.predict_proba(tree_X)[:, 1]
    ens_probs = (lr_probs + tree_probs) / 2
    ens_preds = (ens_probs > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_preds)
    print(f"\n  LR + {'XGBoost' if xgb else 'GB'} Ensemble:")
    print(f"    Accuracy:  {ens_acc:.3f}")
    print(f"    Log Loss:  {log_loss(y_test, ens_probs):.3f}")
    print(f"    Brier:     {brier_score_loss(y_test, ens_probs):.3f}")
    print(f"    AUC:       {roc_auc_score(y_test, ens_probs):.3f}")

    return lr, gb, scaler, test_df, y_test, xgb


# ===========================================================================
# Section 8: Visualization
# ===========================================================================
def plot_feature_importance(gb, feature_cols):
    importances = pd.Series(gb.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    importances.plot.barh(ax=ax, color="#3498db")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Gradient Boosting — Feature Importance")
    plt.tight_layout()
    plt.savefig("model_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_feature_importance.png")


def plot_calibration(gb, X_test, y_test):
    probs = gb.predict_proba(X_test)[:, 1]
    bins = pd.cut(probs, bins=10)
    cal = pd.DataFrame({"prob": probs, "actual": y_test, "bin": bins})
    grouped = cal.groupby("bin", observed=True).agg(
        mean_pred=("prob", "mean"),
        mean_actual=("actual", "mean"),
        count=("actual", "count")
    ).dropna()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.scatter(grouped["mean_pred"], grouped["mean_actual"],
               s=grouped["count"] * 2, alpha=0.7, zorder=5)
    for _, row in grouped.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row["mean_pred"], row["mean_actual"]),
                     fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Predicted Probability (Home Win)")
    ax.set_ylabel("Actual Home Win Rate")
    ax.set_title("Calibration Plot — 2025 Holdout")
    ax.legend()
    plt.tight_layout()
    plt.savefig("model_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_calibration.png")


def plot_confusion(gb, X_test, y_test):
    preds = gb.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Away Win", "Home Win"],
                yticklabels=["Away Win", "Home Win"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — 2025 Holdout (Gradient Boosting)")
    plt.tight_layout()
    plt.savefig("model_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_confusion_matrix.png")


# ===========================================================================
# Section 9: 2026 Prediction Interface
# ===========================================================================
def build_2025_baselines(df, tgl):
    """Extract end-of-2025 team and SP stats for easy 2026 prediction."""
    # Team baselines: last computed features for each team in 2025
    tgl_2025 = tgl[tgl["season"] == 2025]
    team_baselines = {}
    for team in tgl_2025["team"].unique():
        team_data = tgl_2025[tgl_2025["team"] == team].iloc[-1]
        team_baselines[team] = {
            "pyth_win_pct":          float(team_data["pyth_win_pct"]),
            "win_pct":               float(team_data["season_win_pct"]),
            # Rate stats (new)
            "runs_per_game":         float(team_data["roll30_runs_scored"]),
            "runs_allowed_per_game": float(team_data["roll30_runs_allowed"]),
            "recent_runs_per_game":  float(team_data["roll10_runs_scored"]),
            "obp":                   float(team_data["roll30_obp"]),
            "slg":                   float(team_data.get("roll30_slg", 0.400)),
            "iso":                   float(team_data["roll30_iso"]),
            # Volume rolling (kept for display / O/U formula; not all are model features)
            "hits_per_game":         float(team_data["roll30_hits"]),
            "opp_hits_per_game":     float(team_data.get("roll30_opp_hits", 8.5)),
            "walks_per_game":        float(team_data["roll30_walks"]),
            "opp_walks_per_game":    float(team_data["roll30_opp_walks"]),
            "hr_per_game":           float(team_data["roll30_homeruns"]),
            "opp_hr_per_game":       float(team_data.get("roll30_opp_homeruns", 1.1)),
            "recent_win_pct":        float(team_data["roll10_win_pct"]),
            "recent_hr_per_game":    float(team_data["roll10_homeruns"]),
            "opp_k_per_game":        float(team_data["roll30_opp_strikeouts"]),
            # Normalized defensive rate stats (model features)
            "opp_whip":              float(team_data.get("roll30_opp_whip", 1.30)),
            "opp_hr_per9":           float(team_data.get("roll30_opp_hr_per9", 1.10)),
            # Bullpen fatigue (model feature)
            "roll7_bullpen_fatigue": float(team_data.get("roll7_bullpen_fatigue", 8.0)),
        }

    # SP baselines: real pitcher stats from Baseball Reference
    sp_baselines = {}
    sp_names = {}
    for _, row in df[df["season"] == 2025].iterrows():
        sp_names[row["home_starting_pitcher_id"]] = row["home_starting_pitcher_name"]
        sp_names[row["visitor_starting_pitcher_id"]] = row["visitor_starting_pitcher_name"]

    for team in tgl_2025["team"].unique():
        team_data = tgl_2025[tgl_2025["team"] == team].iloc[-1]
        team_baselines[team]["bullpen_era"] = float(team_data.get("bullpen_era", 4.20))
        team_baselines[team]["park_factor"] = PARK_FACTORS.get(team, 1.0)

    for pid in tgl_2025["starting_pitcher_id"].unique():
        sp_data = tgl_2025[tgl_2025["starting_pitcher_id"] == pid].iloc[-1]
        sp_baselines[pid] = {
            "name":  sp_names.get(pid, pid),
            "era":   float(sp_data.get("sp_era",   4.20)),
            "whip":  float(sp_data.get("sp_whip",  1.30)),
            "xfip":  float(sp_data.get("sp_xfip",  4.20)),
            "siera": float(sp_data.get("sp_siera", 4.20)),
            "so9":   float(sp_data.get("sp_so9",   8.0)),
            "bb9":   float(sp_data.get("sp_bb9",   3.0)),
            "hr9":   float(sp_data.get("sp_hr9",   1.2)),
            "ip_gs": float(sp_data.get("sp_ip_gs", 5.8)),
            "k_bb":  float(sp_data.get("sp_k_bb",  2.5)),
        }

    return team_baselines, sp_baselines


def estimate_game_total(home_ts, away_ts, home_sp, away_sp):
    """Formula-based estimated total runs (for O/U display). Not ML — uses run rates + SP ERA.
    Returns dict with 'home', 'away', and 'total' keys."""
    park   = home_ts.get("park_factor", 1.0)
    h_off  = home_ts.get("recent_runs_per_game", 4.5)
    a_off  = away_ts.get("recent_runs_per_game", 4.5)
    # SP ERA relative to 4.20 league avg → scales opponent run expectation
    # home team scores: their offense vs the AWAY SP
    # away team scores: their offense vs the HOME SP
    a_sp_adj = away_sp.get("era", 4.20) / 4.20
    h_sp_adj = home_sp.get("era", 4.20) / 4.20
    home_team_runs = h_off * a_sp_adj * park
    away_team_runs = a_off * h_sp_adj * park
    return {
        "home":  round(home_team_runs, 1),
        "away":  round(away_team_runs, 1),
        "total": round(home_team_runs + away_team_runs, 1),
        "components": {
            "away_off":   round(a_off, 2),
            "home_off":   round(h_off, 2),
            "home_sp_adj": round(h_sp_adj, 2),
            "away_sp_adj": round(a_sp_adj, 2),
            "park":        round(park, 2),
        },
    }


def predict_game(home_team_stats, away_team_stats, home_sp_stats, away_sp_stats,
                 model, scaler=None, feature_cols=None, runline_models=None,
                 gb_model=None, xgb_model=None):
    """
    Predict probability of home team winning.

    home_team_stats / away_team_stats: dict with keys
        pyth_win_pct, win_pct, runs_per_game, runs_allowed_per_game,
        recent_runs_per_game, obp, iso, hits_per_game,
        walks_per_game, hr_per_game, recent_win_pct, recent_hr_per_game,
        opp_k_per_game, opp_whip, opp_hr_per9,
        roll7_bullpen_fatigue, bullpen_era, park_factor

    home_sp_stats / away_sp_stats: dict with keys
        era, whip, xfip, siera, so9, bb9, hr9
    """
    features = {
        "home_park_factor":            home_team_stats.get("park_factor", 1.0),
        "diff_pyth_win_pct":           home_team_stats["pyth_win_pct"]           - away_team_stats["pyth_win_pct"],
        "diff_season_win_pct":         home_team_stats["win_pct"]                - away_team_stats["win_pct"],
        # Rate stats
        "diff_roll30_runs_scored":     home_team_stats["runs_per_game"]          - away_team_stats["runs_per_game"],
        "diff_roll30_runs_allowed":    home_team_stats["runs_allowed_per_game"]  - away_team_stats["runs_allowed_per_game"],
        "diff_roll10_runs_scored":     home_team_stats["recent_runs_per_game"]   - away_team_stats["recent_runs_per_game"],
        "diff_roll30_obp":             home_team_stats["obp"]                         - away_team_stats["obp"],
        "diff_roll30_iso":             home_team_stats.get("iso", 0.150)              - away_team_stats.get("iso", 0.150),
        # Normalized defensive rate stats (replace raw opp_hits / opp_homeruns / errors)
        "diff_roll30_opp_whip":        home_team_stats.get("opp_whip", 1.30)          - away_team_stats.get("opp_whip", 1.30),
        "diff_roll30_opp_hr_per9":     home_team_stats.get("opp_hr_per9", 1.10)       - away_team_stats.get("opp_hr_per9", 1.10),
        "diff_roll30_opp_strikeouts":  home_team_stats["opp_k_per_game"]              - away_team_stats["opp_k_per_game"],
        "diff_roll10_win_pct":         home_team_stats["recent_win_pct"]              - away_team_stats["recent_win_pct"],
        "diff_roll10_runs_scored":     home_team_stats["recent_runs_per_game"]        - away_team_stats["recent_runs_per_game"],
        "diff_roll10_homeruns":        home_team_stats["recent_hr_per_game"]          - away_team_stats["recent_hr_per_game"],
        # Bullpen — exp-decayed fatigue replaces raw IP sum and pitchers-used count
        "diff_bullpen_era":            home_team_stats.get("bullpen_era", 4.20)       - away_team_stats.get("bullpen_era", 4.20),
        "diff_roll7_bullpen_fatigue":  home_team_stats.get("roll7_bullpen_fatigue", 8.0) - away_team_stats.get("roll7_bullpen_fatigue", 8.0),
        # Extra display stats (not in FEATURE_COLS — kept for logging/UI)
        "diff_roll30_hits":            home_team_stats.get("hits_per_game", 8.5)      - away_team_stats.get("hits_per_game", 8.5),
        "diff_roll30_homeruns":        home_team_stats.get("hr_per_game", 1.1)        - away_team_stats.get("hr_per_game", 1.1),
        "diff_rest_days":              max(-1, min(1, home_team_stats.get("rest_days", 1) - away_team_stats.get("rest_days", 1))),
        "home_sp_is_lhp":              1 if home_sp_stats.get("pitch_hand", "R") == "L" else 0,
        "away_sp_is_lhp":              1 if away_sp_stats.get("pitch_hand", "R") == "L" else 0,
        "diff_sp_era":                 home_sp_stats["era"]   - away_sp_stats["era"],
        "diff_sp_whip":                home_sp_stats["whip"]  - away_sp_stats["whip"],
        "diff_sp_xfip":                home_sp_stats["xfip"]  - away_sp_stats["xfip"],
        "diff_sp_ip_gs":               home_sp_stats.get("ip_gs", 5.8) - away_sp_stats.get("ip_gs", 5.8),
        "diff_sp_k_bb":                home_sp_stats.get("k_bb", 2.5)  - away_sp_stats.get("k_bb", 2.5),
        # Extra stats kept in dict for display/logging, not used as features
        "diff_sp_siera":               home_sp_stats.get("siera", 4.0) - away_sp_stats.get("siera", 4.0),
        "diff_sp_so9":                 home_sp_stats.get("so9", 8.0)   - away_sp_stats.get("so9", 8.0),
        "diff_sp_bb9":                 home_sp_stats.get("bb9", 3.0)   - away_sp_stats.get("bb9", 3.0),
        "diff_sp_hr9":                 home_sp_stats.get("hr9", 1.2)   - away_sp_stats.get("hr9", 1.2),
    }

    if feature_cols is None:
        feature_cols = FEATURE_COLS
    X_raw = pd.DataFrame([features])[feature_cols]

    X_scaled = scaler.transform(X_raw) if scaler is not None else X_raw

    # Ensemble: LR + GB + XGB (use whichever models are available)
    probs = [float(model.predict_proba(X_scaled)[0, 1])]  # LR (scaled)
    if gb_model is not None:
        probs.append(float(gb_model.predict_proba(X_raw)[0, 1]))   # GB (raw)
    if xgb_model is not None:
        probs.append(float(xgb_model.predict_proba(X_raw)[0, 1]))  # XGB (raw)
    prob = sum(probs) / len(probs)

    # Soft recalibration: nudge 4% toward MLB home win prior (53%)
    # Reduces overconfident away picks without a hard cutoff
    _HOME_PRIOR  = 0.53
    _RECAL_BLEND = 0.04
    prob = (1 - _RECAL_BLEND) * prob + _RECAL_BLEND * _HOME_PRIOR

    result = {
        "home_win_prob": round(prob, 3),
        "away_win_prob": round(1 - prob, 3),
        "predicted_winner": "Home" if prob > 0.5 else "Away",
        "confidence": round(abs(prob - 0.5) * 2, 3),
        "x_scaled_features": (X_scaled.values[0] if hasattr(X_scaled, "values") else X_scaled[0]).tolist(),
    }

    # Run line prediction (-1.5): ensemble of LR + GBM if models provided
    if runline_models is not None:
        try:
            rl_lr, rl_gb, rl_scaler = runline_models
            X_rl = X_raw  # same features, raw values
            X_rl_sc = rl_scaler.transform(X_rl) if rl_scaler is not None else X_rl
            rl_lr_prob  = float(rl_lr.predict_proba(X_rl_sc)[0, 1])
            rl_gb_prob  = float(rl_gb.predict_proba(X_rl)[0, 1])
            rl_prob = round((rl_lr_prob + rl_gb_prob) / 2, 3)
            result["home_cover_prob"] = rl_prob
            result["away_cover_prob"] = round(1 - rl_prob, 3)
        except Exception:
            result["home_cover_prob"] = None
            result["away_cover_prob"] = None
    else:
        result["home_cover_prob"] = None
        result["away_cover_prob"] = None

    # Estimated game total (formula-based O/U)
    _est = estimate_game_total(
        home_team_stats, away_team_stats, home_sp_stats, away_sp_stats
    )
    result["predicted_total"]    = _est["total"]
    result["home_est_score"]     = _est["home"]
    result["away_est_score"]     = _est["away"]
    result["est_components"]     = _est["components"]

    return result


def predict_by_name(home_team, away_team, home_sp_id, away_sp_id,
                    team_baselines, sp_baselines, model, scaler=None):
    """Predict using team abbreviations and pitcher IDs from the 2025 baselines."""
    home_ts = team_baselines[home_team]
    away_ts = team_baselines[away_team]
    home_sp = sp_baselines.get(home_sp_id, _default_sp_stats())
    away_sp = sp_baselines.get(away_sp_id, _default_sp_stats())
    return predict_game(home_ts, away_ts, home_sp, away_sp, model, scaler)


def _default_sp_stats():
    return {
        "era":          4.20,
        "ip_gs":        5.8,
        "k_bb":         2.5,
        "whip":         1.30,
        "xfip":         4.0,
        "siera":        4.0,
        "so9":          8.0,
        "bb9":          3.0,
        "hr9":          1.2,
        # Raw display fields — shown on card as league-average placeholder
        "era_raw":      4.20,
        "whip_raw":     1.30,
        "fip_raw":      4.00,
        "gs":           0,
        "is_blended":   False,
        "is_league_avg": True,
    }


# ===========================================================================
# Section 10: VIF helper
# ===========================================================================
def compute_vif(model_df, feature_cols):
    """Compute Variance Inflation Factor for each feature.
    VIF > 10 indicates severe multicollinearity; > 5 is moderate."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("  [VIF] statsmodels not installed — skipping VIF calculation")
        return None
    X = model_df[feature_cols].dropna()
    vif_df = pd.DataFrame({
        "feature": feature_cols,
        "VIF":     [variance_inflation_factor(X.values, i) for i in range(len(feature_cols))],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


# ===========================================================================
# Section 11: Pitcher Handedness Lookup (for future platoon splits)
# ===========================================================================
def build_handedness_lookup(db_path):
    """Fetch pitch_hand (L/R) for all pitchers via MLB Stats API.
    Creates pitcher_handedness table in DB: (player_name TEXT, pitch_hand TEXT).
    Safe to re-run — uses INSERT OR REPLACE."""
    import requests as _req
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_handedness (
            player_name TEXT PRIMARY KEY,
            pitch_hand  TEXT,
            updated     TEXT
        )
    """)
    conn.commit()

    print("  [handedness] Fetching pitcher list from MLB Stats API...", flush=True)
    try:
        resp = _req.get(
            "https://statsapi.mlb.com/api/v1/stats",
            params={"stats": "career", "group": "pitching", "sportId": 1,
                    "playerPool": "All", "limit": 5000},
            timeout=30,
        )
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        print(f"  [handedness] MLB API failed: {e}")
        conn.close()
        return

    player_ids = [s["player"]["id"] for s in splits if s.get("player")]
    print(f"  [handedness] {len(player_ids)} pitcher IDs found — fetching handedness...", flush=True)

    handedness_map = {}
    chunk_size = 100
    for i in range(0, len(player_ids), chunk_size):
        chunk = player_ids[i : i + chunk_size]
        try:
            pr = _req.get(
                "https://statsapi.mlb.com/api/v1/people",
                params={"personIds": ",".join(str(x) for x in chunk),
                        "fields":    "people,id,fullName,pitchHand,code"},
                timeout=15,
            )
            for person in pr.json().get("people", []):
                name = person.get("fullName", "")
                hand = person.get("pitchHand", {}).get("code", "R")
                if name:
                    handedness_map[name] = hand
        except Exception:
            pass

    now = pd.Timestamp.now().isoformat()
    for name, hand in handedness_map.items():
        conn.execute(
            "INSERT OR REPLACE INTO pitcher_handedness (player_name, pitch_hand, updated) VALUES (?, ?, ?)",
            (name, hand, now),
        )
    conn.commit()
    conn.close()
    lhp = sum(1 for h in handedness_map.values() if h == "L")
    print(f"  [handedness] Stored {len(handedness_map)} pitchers ({lhp} LHP, {len(handedness_map)-lhp} RHP)")


# ===========================================================================
# Section 12: Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MLB Game Prediction Model")
    print("=" * 60)

    # Step 1: Load
    print("\n[1/7] Loading data...")
    df, pitcher_stats, bullpen_stats = load_data(DB_PATH)
    print(f"  Loaded {len(df)} games (2021-2026)")
    print(f"  Loaded {len(pitcher_stats)} pitcher-season records")
    has_xfip = "xfip" in pitcher_stats.columns
    has_siera = "siera" in pitcher_stats.columns
    print(f"  Advanced SP metrics: xFIP={'yes' if has_xfip else 'no (run fetch_advanced_pitching.py)'}  "
          f"SIERA={'yes' if has_siera else 'no'}")
    print(f"  Bullpen ERA table: {'yes (' + str(len(bullpen_stats)) + ' records)' if len(bullpen_stats) > 0 else 'no (run fetch_advanced_pitching.py)'}")

    # Step 2: Build team game log
    print("\n[2/7] Building team game log...")
    tgl = build_team_game_log(df)
    print(f"  Created {len(tgl)} team-game records")

    # Step 3: Team features
    print("\n[3/7] Engineering rolling team features...")
    tgl = compute_rolling_team_features(tgl)

    # Step 4: SP features (real stats from Baseball Reference / FanGraphs)
    print("\n[4/7] Merging starting pitcher stats...")
    tgl = merge_sp_stats(tgl, pitcher_stats)
    tgl = merge_bullpen_era(tgl, bullpen_stats)

    # Step 5: Assemble
    print("\n[5/7] Assembling feature matrix...")
    model_df = assemble_features(df, tgl)
    valid = model_df.dropna(subset=FEATURE_COLS)
    print(f"  Feature matrix: {model_df.shape[0]} games ({valid.shape[0]} with complete features)")

    # Step 5b: VIF analysis — verify multicollinearity reduction after SLG drop
    print("\n[5b] VIF analysis on current feature set...")
    vif_result = compute_vif(valid, FEATURE_COLS)
    if vif_result is not None:
        print(vif_result.to_string(index=False))
        high_vif = vif_result[vif_result["VIF"] > 5]
        if len(high_vif) > 0:
            print(f"  ⚠ {len(high_vif)} features with VIF > 5: {high_vif['feature'].tolist()}")
        else:
            print("  ✓ No features with VIF > 5 — multicollinearity is acceptable")

    # Step 6: Cross-validate
    print("\n[6/7] Cross-validation (Leave-One-Season-Out)...")
    cv_results = cross_validate_loso(model_df, FEATURE_COLS)

    # Step 7: 2025 holdout (reference evaluation — train 2021-2024, test 2025)
    print("\n[7/7] Evaluating on 2025 holdout...")
    lr, gb, scaler, X_test_df, y_test, xgb = evaluate_holdout(model_df, FEATURE_COLS)

    # Step 7b: Hyperparameter grid search (train 2021-2024, evaluate 2025 holdout)
    print("\n[7b] Hyperparameter grid search (12 combos, 2025 holdout)...")
    _gs_train = model_df[model_df["season"].between(2021, 2024)].dropna(subset=FEATURE_COLS)
    _gs_test  = model_df[model_df["season"] == 2025].dropna(subset=FEATURE_COLS)
    _X_gstr   = _gs_train[FEATURE_COLS]
    _y_gstr   = _gs_train["home_win"]
    _X_gste   = _gs_test[FEATURE_COLS]
    _y_gste   = _gs_test["home_win"]
    _best_params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8}
    _best_acc = 0.0
    for _n_est in [200, 300]:
        for _max_d in [3, 4, 5]:
            for _lr_val in [0.03, 0.05]:
                _p = {"n_estimators": _n_est, "max_depth": _max_d,
                      "learning_rate": _lr_val, "subsample": 0.8}
                _gb_cv = GradientBoostingClassifier(**_p, random_state=RANDOM_STATE)
                _gb_cv.fit(_X_gstr, _y_gstr)
                _acc = accuracy_score(_y_gste, _gb_cv.predict(_X_gste))
                if _acc > _best_acc:
                    _best_acc, _best_params = _acc, dict(_p)
    print(f"  Best GBM params: {_best_params}  holdout_acc={_best_acc:.3f}")

    # Step 7c: Final model — retrain on ALL data 2021-2026 with recency weights
    print("\n[7c] Retraining final model on 2021-2026 with recency weights...")
    YEAR_WEIGHTS = {2021: 1.0, 2022: 1.1, 2023: 1.3, 2024: 1.5, 2025: 1.8, 2026: 1.8}
    _final_df = model_df.dropna(subset=FEATURE_COLS)
    _sw = _final_df["season"].map(YEAR_WEIGHTS).fillna(1.0)
    X_final = _final_df[FEATURE_COLS]
    y_final = _final_df["home_win"]
    seasons_in = sorted(_final_df["season"].unique())
    print(f"  Training on seasons: {seasons_in}  ({len(_final_df)} games)")

    final_scaler = StandardScaler()
    X_final_sc = final_scaler.fit_transform(X_final)

    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_final_sc, y_final, sample_weight=_sw)

    gb = GradientBoostingClassifier(**_best_params, random_state=RANDOM_STATE)
    gb.fit(X_final, y_final, sample_weight=_sw)

    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=_best_params["n_estimators"],
            max_depth=_best_params["max_depth"],
            learning_rate=_best_params["learning_rate"],
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
        )
        xgb.fit(X_final, y_final, sample_weight=_sw)

    scaler = final_scaler
    print("  Final model trained.")

    # Step 7c: Run line model (home covers -1.5)
    print("\n[7d] Training run line model (home covers -1.5)...")
    rl_df = model_df.dropna(subset=FEATURE_COLS + ["home_covers"])
    rl_df = rl_df[rl_df["home_covers"].notna()]
    # Evaluate on 2025 holdout, but train final on all seasons
    rl_train = rl_df[rl_df["season"].between(2021, 2024)]
    rl_test  = rl_df[rl_df["season"] == 2025]
    X_rl_train = rl_train[FEATURE_COLS]
    y_rl_train = rl_train["home_covers"].astype(int)
    X_rl_test  = rl_test[FEATURE_COLS]
    y_rl_test  = rl_test["home_covers"].astype(int)

    rl_scaler = StandardScaler()
    X_rl_tr_sc = rl_scaler.fit_transform(X_rl_train)
    X_rl_te_sc = rl_scaler.transform(X_rl_test)

    rl_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE)
    rl_lr.fit(X_rl_tr_sc, y_rl_train)
    rl_gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )
    rl_gb.fit(X_rl_train, y_rl_train)

    rl_lr_probs  = rl_lr.predict_proba(X_rl_te_sc)[:, 1]
    rl_gb_probs  = rl_gb.predict_proba(X_rl_test)[:, 1]
    rl_ens_probs = (rl_lr_probs + rl_gb_probs) / 2
    rl_ens_preds = (rl_ens_probs > 0.5).astype(int)
    rl_acc  = accuracy_score(y_rl_test, rl_ens_preds)
    rl_base = max(y_rl_test.mean(), 1 - y_rl_test.mean())
    print(f"  Run line covers rate (home): {y_rl_test.mean():.3f}")
    print(f"  Baseline: {rl_base:.3f}  |  Ensemble accuracy: {rl_acc:.3f}")

    # Retrain run-line models on all seasons 2021-2026 with recency weights
    _rl_all = rl_df.copy()
    _rl_sw  = _rl_all["season"].map(YEAR_WEIGHTS).fillna(1.0)
    X_rl_all = _rl_all[FEATURE_COLS]
    y_rl_all = _rl_all["home_covers"].astype(int)
    rl_scaler = StandardScaler()
    X_rl_all_sc = rl_scaler.fit_transform(X_rl_all)
    rl_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE)
    rl_lr.fit(X_rl_all_sc, y_rl_all, sample_weight=_rl_sw)
    rl_gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )
    rl_gb.fit(X_rl_all, y_rl_all, sample_weight=_rl_sw)

    # Plots
    print("\n  Generating plots...")
    plot_feature_importance(gb, FEATURE_COLS)
    X_test_vals = X_test_df[FEATURE_COLS]
    plot_calibration(gb, X_test_vals, y_test)
    plot_confusion(gb, X_test_vals, y_test)

    # Feature importance printout
    print("\n  --- Feature Importance (Gradient Boosting) ---")
    importances = pd.Series(gb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"    {feat:<40s}  {imp:.4f}")

    # LR coefficients
    print("\n  --- Logistic Regression Coefficients (C=0.5) ---")
    coefs = pd.Series(lr.coef_[0], index=FEATURE_COLS).sort_values(ascending=False)
    for feat, c in coefs.items():
        print(f"    {feat:<40s}  {c:+.4f}")

    # Post-hoc SP ERA weight boost: multiply LR coefficient by 1.4
    # StandardScaler absorbs any training-time scaling, so this is the only
    # effective way to increase SP ERA's influence in the LR model.
    _era_idx = list(FEATURE_COLS).index("diff_sp_era")
    _old_era_coef = lr.coef_[0][_era_idx]
    lr.coef_[0][_era_idx] *= 1.4
    print(f"\n  SP ERA coefficient: {_old_era_coef:+.4f} → {lr.coef_[0][_era_idx]:+.4f}  (1.4× boost)")

    # Build 2025 baselines for 2026 predictions
    print("\n  Building 2025 baselines for 2026 predictions...")
    team_baselines, sp_baselines = build_2025_baselines(df, tgl)

    # Save model artifacts
    artifacts = {
        "gb_model":       gb,
        "lr_model":       lr,
        "scaler":         scaler,
        "feature_cols":   FEATURE_COLS,
        "team_baselines": team_baselines,
        "sp_baselines":   sp_baselines,
        # Run line models
        "lr_runline":     rl_lr,
        "gb_runline":     rl_gb,
        "scaler_runline": rl_scaler,
    }
    if xgb is not None:
        artifacts["xgb_model"] = xgb
    with open(_ARTIFACTS_SAVE_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Saved model artifacts to {_ARTIFACTS_SAVE_PATH}")

    # Demo prediction
    print("\n" + "=" * 60)
    print("DEMO PREDICTION: LAN (Dodgers) vs NYA (Yankees)")
    print("=" * 60)
    if "LAN" in team_baselines and "NYA" in team_baselines:
        # Find a known SP for each team
        lan_sps = [pid for pid, info in sp_baselines.items()
                   if tgl[(tgl["starting_pitcher_id"] == pid) & (tgl["season"] == 2025) &
                          (tgl["team"] == "LAN")].shape[0] > 5]
        nya_sps = [pid for pid, info in sp_baselines.items()
                   if tgl[(tgl["starting_pitcher_id"] == pid) & (tgl["season"] == 2025) &
                          (tgl["team"] == "NYA")].shape[0] > 5]

        if lan_sps and nya_sps:
            home_sp_id = lan_sps[0]
            away_sp_id = nya_sps[0]
            result = predict_by_name("LAN", "NYA", home_sp_id, away_sp_id,
                                     team_baselines, sp_baselines, gb)
            print(f"  Home SP: {sp_baselines[home_sp_id]['name']}")
            print(f"  Away SP: {sp_baselines[away_sp_id]['name']}")
            print(f"  Home win prob: {result['home_win_prob']:.1%}")
            print(f"  Away win prob: {result['away_win_prob']:.1%}")
            print(f"  Predicted winner: {result['predicted_winner']}")

    # Print available teams for 2026 predictions
    print("\n  Available teams for 2026 predictions:")
    for team in sorted(team_baselines.keys()):
        print(f"    {team}", end="")
    print()

    print("\n  To predict a 2026 game, load mlb_model_artifacts.pkl and use predict_by_name()")
    print("  Done!")
