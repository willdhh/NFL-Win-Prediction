import sys
import subprocess

# try:
#     import pandas as pd
# except ModuleNotFoundError:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
#     import pandas as pd

import pandas as pd
import glob
import os

# Load the game spread data
df = pd.read_csv("data/spreadspoke_scores.csv")
df = df.dropna(subset=['spread_favorite'])  # Drop rows with missing spread_favorite
df = df[df['schedule_season'] >= 1978].reset_index(drop=True)

# Load the team stats for each season
teamstat_csv = glob.glob(os.path.join("data/team stats", "*.csv"))
season_stat = {}

for csv in teamstat_csv:
    year = os.path.splitext(os.path.basename(csv))[0]
    season_stat[int(year)] = pd.read_csv(csv, header=1)

# Function to get previous season stats
def get_previous_season_stats(team_name, current_season):
    previous_season = current_season - 1
    if previous_season in season_stat:
        team_stats = season_stat[previous_season]
        team_stats = team_stats[team_stats['Tm'] == team_name]
        if not team_stats.empty:
            return team_stats[['Tm', 'PF', 'Yds', 'TO', '1stD']]  # Example stats to join
    return None

# Prepare lists to store home and away team stats
home_team_stats = []
away_team_stats = []
valid_rows = []

for i in range(len(df)):
    current_season = df.iloc[i]['schedule_season']
    
    # Get previous season stats for home and away teams
    home_stats = get_previous_season_stats(df.iloc[i]['team_home'], current_season)
    away_stats = get_previous_season_stats(df.iloc[i]['team_away'], current_season)
    
    # Only append data if both teams have stats from the previous season
    if home_stats is not None and away_stats is not None:
        home_team_stats.append(home_stats.values.flatten())
        away_team_stats.append(away_stats.values.flatten())
        valid_rows.append(i)  # Track rows that have valid stats

# If there are valid previous season stats, merge them with the dataframe
if home_team_stats and away_team_stats:
    home_stats_df = pd.DataFrame(home_team_stats, columns=['home_Tm', 'home_PF', 'home_Yds', 'home_TO', 'home_1stD'])
    away_stats_df = pd.DataFrame(away_team_stats, columns=['away_Tm', 'away_PF', 'away_Yds', 'away_TO', 'away_1stD'])
    
    # Merge home and away stats with the main dataframe using only valid rows
    df_valid = df.iloc[valid_rows].reset_index(drop=True)
    df_valid = pd.concat([df_valid, home_stats_df, away_stats_df], axis=1)

df_valid.to_csv("data/nfl_spread_with_team_stats.csv", index=False)

print(df_valid.columns.tolist())
# Print confirmation message
print("Data saved to 'data/merged_team_stats.csv'")