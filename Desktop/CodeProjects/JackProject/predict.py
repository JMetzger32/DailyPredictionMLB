import pickle
from MLBModel import predict_by_name, predict_game

# Load saved model artifacts
with open("mlb_model_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

lr_model = artifacts["lr_model"]
scaler = artifacts.get("scaler")
team_baselines = artifacts["team_baselines"]
sp_baselines = artifacts["sp_baselines"]

# Team name lookup (Retrosheet abbreviation -> full name)
TEAM_NAMES = {
    "ANA": "Angels", "ARI": "Diamondbacks", "ATH": "Athletics", "ATL": "Braves",
    "BAL": "Orioles", "BOS": "Red Sox", "CHA": "White Sox", "CHN": "Cubs",
    "CIN": "Reds", "CLE": "Guardians", "COL": "Rockies", "DET": "Tigers",
    "HOU": "Astros", "KCA": "Royals", "LAN": "Dodgers", "MIA": "Marlins",
    "MIL": "Brewers", "MIN": "Twins", "NYA": "Yankees", "NYN": "Mets",
    "PHI": "Phillies", "PIT": "Pirates", "SDN": "Padres", "SEA": "Mariners",
    "SFN": "Giants", "SLN": "Cardinals", "TBA": "Rays", "TEX": "Rangers",
    "TOR": "Blue Jays", "WAS": "Nationals",
}


def show_teams():
    print("\n  Available teams:")
    for abbr in sorted(TEAM_NAMES.keys()):
        print(f"    {abbr} = {TEAM_NAMES[abbr]}")


def show_pitchers(team_abbr):
    print(f"\n  Starting pitchers for {team_abbr} ({TEAM_NAMES.get(team_abbr, '')}):")
    matches = []
    for pid, info in sp_baselines.items():
        matches.append((pid, info["name"], info["era"]))
    # Sort by name
    matches.sort(key=lambda x: x[1])
    for pid, name, era in matches:
        print(f"    {pid:<15s}  {name:<25s}  (ERA: {era:.2f})")
    return matches


def find_pitcher(query):
    """Search for a pitcher by partial name match."""
    query_lower = query.lower()
    matches = []
    for pid, info in sp_baselines.items():
        if query_lower in info["name"].lower() or query_lower in pid.lower():
            matches.append((pid, info["name"]))
    return matches


def main():
    print("=" * 60)
    print("  MLB Game Predictor (using 2025 end-of-season baselines)")
    print("=" * 60)

    while True:
        print("\n  Commands:")
        print("    predict  - Predict a game")
        print("    teams    - Show available teams")
        print("    search   - Search for a pitcher by name")
        print("    quit     - Exit")

        cmd = input("\n  > ").strip().lower()

        if cmd == "quit" or cmd == "q":
            print("  Goodbye!")
            break

        elif cmd == "teams":
            show_teams()

        elif cmd == "search":
            query = input("  Pitcher name (partial): ").strip()
            matches = find_pitcher(query)
            if matches:
                for pid, name in matches:
                    stats = sp_baselines[pid]
                    print(f"    {pid:<15s}  {name:<25s}  ERA: {stats['era']:.2f}  WHIP: {stats['whip']:.3f}  xFIP: {stats.get('xfip', stats['era']):.2f}  SIERA: {stats.get('siera', stats['era']):.2f}")
            else:
                print("    No matches found.")

        elif cmd == "predict":
            print()
            show_teams()
            home = input("\n  Home team (abbreviation): ").strip().upper()
            away = input("  Away team (abbreviation): ").strip().upper()

            if home not in team_baselines:
                print(f"  Unknown team: {home}")
                continue
            if away not in team_baselines:
                print(f"  Unknown team: {away}")
                continue

            # Search for home SP
            home_sp_query = input(f"\n  Home starting pitcher (name search): ").strip()
            home_matches = find_pitcher(home_sp_query)
            if not home_matches:
                print("  No pitcher found. Using league average.")
                home_sp_id = None
            elif len(home_matches) == 1:
                home_sp_id = home_matches[0][0]
                print(f"  Found: {home_matches[0][1]}")
            else:
                print("  Multiple matches:")
                for i, (pid, name) in enumerate(home_matches):
                    print(f"    {i+1}. {name} ({pid})")
                choice = input("  Pick number: ").strip()
                try:
                    home_sp_id = home_matches[int(choice) - 1][0]
                except (ValueError, IndexError):
                    print("  Invalid choice. Using league average.")
                    home_sp_id = None

            # Search for away SP
            away_sp_query = input(f"\n  Away starting pitcher (name search): ").strip()
            away_matches = find_pitcher(away_sp_query)
            if not away_matches:
                print("  No pitcher found. Using league average.")
                away_sp_id = None
            elif len(away_matches) == 1:
                away_sp_id = away_matches[0][0]
                print(f"  Found: {away_matches[0][1]}")
            else:
                print("  Multiple matches:")
                for i, (pid, name) in enumerate(away_matches):
                    print(f"    {i+1}. {name} ({pid})")
                choice = input("  Pick number: ").strip()
                try:
                    away_sp_id = away_matches[int(choice) - 1][0]
                except (ValueError, IndexError):
                    print("  Invalid choice. Using league average.")
                    away_sp_id = None

            # Bullpen workload (optional override)
            home_bp = input("\n  Home team bullpen pitchers used last 3 games avg (Enter to use baseline): ").strip()
            away_bp = input("  Away team bullpen pitchers used last 3 games avg (Enter to use baseline): ").strip()

            # Make prediction
            from MLBModel import _default_sp_stats
            home_ts = dict(team_baselines[home])
            away_ts = dict(team_baselines[away])
            home_sp = dict(sp_baselines.get(home_sp_id, _default_sp_stats())) if home_sp_id else _default_sp_stats()
            away_sp = dict(sp_baselines.get(away_sp_id, _default_sp_stats())) if away_sp_id else _default_sp_stats()

            # Override bullpen workload if provided
            if home_bp:
                home_ts["bullpen_used"] = float(home_bp)
            if away_bp:
                away_ts["bullpen_used"] = float(away_bp)

            result = predict_game(home_ts, away_ts, home_sp, away_sp, lr_model, scaler=scaler)

            home_name = TEAM_NAMES.get(home, home)
            away_name = TEAM_NAMES.get(away, away)
            home_sp_name = sp_baselines[home_sp_id]["name"] if home_sp_id and home_sp_id in sp_baselines else "League Average"
            away_sp_name = sp_baselines[away_sp_id]["name"] if away_sp_id and away_sp_id in sp_baselines else "League Average"

            print(f"\n  {'=' * 58}")
            print(f"  {away_name} @ {home_name}")
            print(f"  {'-' * 58}")
            print(f"  {'':22s}  {'AWAY':>12s}  {'HOME':>12s}")
            print(f"  {'Starting Pitcher':22s}  {away_sp_name[:12]:>12s}  {home_sp_name[:12]:>12s}")
            print(f"  {'SP ERA':22s}  {away_sp['era']:>12.2f}  {home_sp['era']:>12.2f}")
            print(f"  {'SP xFIP':22s}  {away_sp.get('xfip', away_sp['era']):>12.2f}  {home_sp.get('xfip', home_sp['era']):>12.2f}")
            print(f"  {'SP SIERA':22s}  {away_sp.get('siera', away_sp['era']):>12.2f}  {home_sp.get('siera', home_sp['era']):>12.2f}")
            print(f"  {'-' * 58}")
            print(f"  {'Team OBP (30g)':22s}  {away_ts.get('obp', 0):>12.3f}  {home_ts.get('obp', 0):>12.3f}")
            print(f"  {'Team SLG (30g)':22s}  {away_ts.get('slg', 0):>12.3f}  {home_ts.get('slg', 0):>12.3f}")
            print(f"  {'Hits/G (30g)':22s}  {away_ts.get('hits_per_game', 0):>12.1f}  {home_ts.get('hits_per_game', 0):>12.1f}")
            print(f"  {'Runs/G (10g recent)':22s}  {away_ts.get('recent_runs_per_game', 0):>12.1f}  {home_ts.get('recent_runs_per_game', 0):>12.1f}")
            print(f"  {'Opp Hits/G (def)':22s}  {away_ts.get('opp_hits_per_game', 0):>12.1f}  {home_ts.get('opp_hits_per_game', 0):>12.1f}")
            print(f"  {'Bullpen ERA':22s}  {away_ts.get('bullpen_era', 4.20):>12.2f}  {home_ts.get('bullpen_era', 4.20):>12.2f}")
            print(f"  {'-' * 58}")
            print(f"  Home win probability:  {result['home_win_prob']:.1%}")
            print(f"  Away win probability:  {result['away_win_prob']:.1%}")
            print(f"  Predicted winner:      {home_name if result['predicted_winner'] == 'Home' else away_name}")
            print(f"  Confidence:            {result['confidence']:.1%}")
            print(f"  {'=' * 58}")

        else:
            print("  Unknown command. Try: predict, teams, search, quit")


if __name__ == "__main__":
    main()
