import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


spread_df = pd.read_csv("data/spreadspoke_scores.csv")
spread_df = spread_df[spread_df['schedule_season'] >= 1994].reset_index(drop=True)
spread_df = spread_df[spread_df['schedule_season'] < 2017].reset_index(drop=True)


papf_df = pd.read_csv("data/spreadspoke_scores.csv")
papf_df = papf_df[papf_df['schedule_season'] >= 1978].reset_index(drop=True)
papf_df = papf_df[papf_df['schedule_playoff'] != True].reset_index(drop=True)

points_for = {}
points_against = {}

for index, row in papf_df.iterrows():
    home_team = row['team_home']
    away_team = row['team_away']
    home_score = row['score_home']
    away_score = row['score_away']
    season = row['schedule_season']
    
    if (season, home_team) not in points_for:
        points_for[(season, home_team)] = 0
    points_for[(season, home_team)] += home_score
    
    if (season, away_team) not in points_for:
        points_for[(season, away_team)] = 0
    points_for[(season, away_team)] += away_score

    if (season, home_team) not in points_against:
        points_against[(season, home_team)] = 0
    points_against[(season, home_team)] += away_score

    if (season, away_team) not in points_against:
        points_against[(season, away_team)] = 0
    points_against[(season, away_team)] += home_score

points_for_df = pd.DataFrame(list(points_for.items()), columns=['Season_Team', 'PF2'])
points_against_df = pd.DataFrame(list(points_against.items()), columns=['Season_Team', 'PA'])

points_for_df[['Season', 'Team']] = pd.DataFrame(points_for_df['Season_Team'].tolist(), index=points_for_df.index)
points_against_df[['Season', 'Team']] = pd.DataFrame(points_against_df['Season_Team'].tolist(), index=points_against_df.index)

points_for_df = points_for_df.drop(columns=['Season_Team'])
points_against_df = points_against_df.drop(columns=['Season_Team'])

points_against_df['Season'] = points_against_df['Season'].astype(int)
# Load the team stats for each season
teamstat_csv = glob.glob(os.path.join("data/team stats", "*.csv"))
season_stat = {}
correct_headers = [
    'Rk', 'Tm', 'G', 'PF', 'Tot_Yds', 'Ply', 'Y/P', 'TO', 'FL', '1stD',
    'Cmp', 'Att', 'Passing_Yds', 'Passing_TD', 'Passing_Int', 'NY/A', 'Passing_1stD',
    'Rushing_Att', 'Rushing_Yds', 'Rushing_TD', 'Y/A', 'Rushing_1stD', 'Pen', 
    'Pen_Yds', '1stPy', 'Sc%', 'TO%'
]

points_against_df['Season'] = points_against_df['Season'].astype(int)

for csv in teamstat_csv:
    year = int(os.path.splitext(os.path.basename(csv))[0]) 
    df = pd.read_csv(csv, header=1)

    if len(df.columns) > len(correct_headers):
        df = df.iloc[:, :len(correct_headers)]
    
    df.columns = correct_headers 
    season_pa = points_against_df[points_against_df['Season'] == year].reset_index(drop=True)

    df = df.merge(season_pa, how='left', left_on='Tm', right_on='Team').drop(columns=['Season', 'Team'])
    season_stat[year] = df 

columns_to_drop = ['Rk', 'Tot_Yds', 'Ply', 'Y/P', '1stD', 'Cmp', 'Rushing_Att']

for year, df in season_stat.items():
    season_stat[year] = df.drop(columns=columns_to_drop)


# Aggregate stats for each season
season_totals = []

for year, df in season_stat.items():
    total_games = df['G'].sum()
    season_totals.append({
        'Season': year,
        'Passing_Yards': df['Passing_Yds'].sum()/total_games,
        'Rushing_Yards': df['Rushing_Yds'].sum()/total_games,
        'Total_Yards':df['Passing_Yds'].sum()/total_games + df['Rushing_Yds'].sum()/total_games,
    })

# Convert to DataFrame
season_df = pd.DataFrame(season_totals).sort_values(by='Season')

# Loop through each season's stats DataFrame
for year, df in season_stat.items():

    df = df.copy()
    df[df.columns.difference(['Tm'])] = df[df.columns.difference(['Tm'])].astype(float)

    stats_to_normalize = df.drop(columns=['Tm'])

    league_mean = stats_to_normalize.mean()
    league_std = stats_to_normalize.std()

    normalized_values = (stats_to_normalize - league_mean) / league_std

    df.loc[:, stats_to_normalize.columns] = normalized_values.astype(float)

    season_stat[year] = df


spread_df = spread_df.dropna(subset=['spread_favorite'])

# Create a new column to indicate where the bet should have been placed to win money
def bet_advised(row):
    # Calculate the margin of victory for both teams
    margin_home = row['score_home'] - row['score_away']
    margin_away = row['score_away'] - row['score_home']
    
    # If the favorite is the home team
    if row['team_favorite_id'] == row['team_home']:
        # Favorite is the home team (spread is negative for favorite)
        if margin_home > abs(row['spread_favorite']):
            return 1  # Bet on the home team (favorite)
        # Underdog is the away team
        elif margin_away > 0:  # If the away team wins outright
            return -1  # Bet on the away team (underdog)
        elif margin_away < row['spread_favorite']:  # If the away team loses but covers the spread
            return -1  # Bet on the away team (underdog)
        else:
            return 0  # No bet advised if spread is not covered
    else:
        # If the favorite is the away team
        if margin_away > abs(row['spread_favorite']):
            return 1  # Bet on the away team (favorite)
        # Underdog is the home team
        elif margin_home > 0:  # If the home team wins outright
            return -1  # Bet on the home team (underdog)
        elif margin_home < row['spread_favorite']:  # If the home team loses but covers the spread
            return -1  # Bet on the home team (underdog)
        else:
            return 0  # No bet advised if spread is not covered

# Apply the function to create the 'bet_advised' column
spread_df['bet_advised'], spread_df['win'] = spread_df.apply(bet_advised, axis=1)
columns_to_drop = [
    'schedule_date', 'schedule_playoff', 'stadium_neutral', 'weather_temperature',
    'weather_wind_mph', 'weather_humidity', 'weather_detail', 'over_under_line'
]

team_name_to_abbr = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAC',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Football Team': 'WAS',
    'St. Louis Rams': 'STL',  # now Los Angeles Rams
    'San Diego Chargers': 'SD',  # now Los Angeles Chargers
    'San Diego Chargers': 'SD',  # previously San Diego Chargers, now Los Angeles Chargers
    'St. Louis Cardinals': 'STL',  # now Arizona Cardinals
    'Houston Oilers': 'HOU'  # now Tennessee Titans
}

# First, map the team abbreviations using the team_name_to_abbr dictionary
spread_df['team_home_abbr'] = spread_df['team_home'].map(lambda x: team_name_to_abbr.get(x, None))

# Create a new column 'new_team_favorite_id' to store the updated value
spread_df['team_favorite_id'] = spread_df.apply(
    lambda row: 1 if row['team_favorite_id'] == row['team_home_abbr'] else 0, axis=1
)

spread_df['score_diff'] = spread_df['score_home'] - spread_df['score_away']
# Drop the temporary 'team_home_abbr' column used for matching
spread_df = spread_df.drop(columns=[
    'score_home','score_away','team_home_abbr','weather_detail','weather_humidity','weather_wind_mph',
    'over_under_line','schedule_date','weather_temperature','stadium_neutral', 'schedule_playoff'])

spread_df['schedule_week'] = spread_df['schedule_week'].replace({
    'Wildcard': 19,
    'Division': 20,
    'Conference': 21,
    'Superbowl': 22
})
spread_df['schedule_week'] = spread_df['schedule_week'].astype(str)
season_data = []
for season, stats in season_stat.items():
    stats['season'] = season+1  # Add the season column
    season_data.append(stats)

# Concatenate all season data into a single DataFrame
season_df = pd.concat(season_data)


spread_df = spread_df.merge(season_df, how='left', left_on=['team_home', 'schedule_season'], right_on=['Tm', 'season'], suffixes=('', '_home'))
spread_df = spread_df.drop(columns=['Tm', 'G'])
spread_df = spread_df.merge(season_df, how='left', left_on=['team_away', 'schedule_season'], right_on=['Tm', 'season'], suffixes=('_home', '_away'))
spread_df = spread_df.drop(columns=['Tm', 'G', 'season_home','season_away','team_home','team_away','Sc%_home','TO%_home','Sc%_away','TO%_away','schedule_week','stadium'])
spread_df = spread_df.dropna()

columns_to_remove = [
    'schedule_season', 'team_favorite_id', 'spread_favorite', 'bet_advised',
    '1stPy_home', '1stPy_away', 'NY/A_home', 'NY/A_away', 'Y/A_home', 'Y/A_away',
    'Passing_1stD_home', 'Passing_1stD_away', 'Rushing_1stD_home', 'Rushing_1stD_away',
    'Att_home', 'Att_away', 'FL_home', 'FL_away',
    'Pen_home', 'Pen_Yds_home', 'Pen_away', 'Pen_Yds_away',
    'Passing_Int_home', 'Passing_Int_away'
]


# Drop these columns from the dataframe
df_cleaned = spread_df.drop(columns=columns_to_remove)

# Optionally, print the first few rows to verify the changes
print(df_cleaned.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
# Prepare the features and target
X = spread_df.drop(columns=['bet_advised'])  # Drop the target column
y = spread_df['bet_advised']  # Replace with the name of the target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(y.value_counts())
# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, classification_report

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Define the range of values for the hyperparameters
# min_samples_split_range = range(2, 102, 2)  # 2 to 102, step 2, for about 50 points
# min_samples_leaf_range = range(1, 51, 1)  # 1 to 50, step 1, for about 50 points

# # Initialize lists to store training and testing accuracy
# train_accuracies_split = []
# test_accuracies_split = []

# train_accuracies_leaf = []
# test_accuracies_leaf = []

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Iterate over min_samples_split values
# for min_samples_split in min_samples_split_range:
#     rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_split=min_samples_split, random_state=42)
#     rf_classifier.fit(X_train, y_train)
    
#     # Record training accuracy
#     train_pred = rf_classifier.predict(X_train)
#     train_accuracy = accuracy_score(y_train, train_pred)
#     train_accuracies_split.append(train_accuracy)
    
#     # Record testing accuracy
#     test_pred = rf_classifier.predict(X_test)
#     test_accuracy = accuracy_score(y_test, test_pred)
#     test_accuracies_split.append(test_accuracy)

# # Iterate over min_samples_leaf values
# for min_samples_leaf in min_samples_leaf_range:
#     rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=min_samples_leaf, random_state=42)
#     rf_classifier.fit(X_train, y_train)
    
#     # Record training accuracy
#     train_pred = rf_classifier.predict(X_train)
#     train_accuracy = accuracy_score(y_train, train_pred)
#     train_accuracies_leaf.append(train_accuracy)
    
#     # Record testing accuracy
#     test_pred = rf_classifier.predict(X_test)
#     test_accuracy = accuracy_score(y_test, test_pred)
#     test_accuracies_leaf.append(test_accuracy)

# # Plot for min_samples_split
# plt.figure(figsize=(10, 6))
# plt.plot(min_samples_split_range, train_accuracies_split, label='Training Accuracy', marker='o')
# plt.plot(min_samples_split_range, test_accuracies_split, label='Testing Accuracy', marker='x')
# plt.title("Training and Testing Accuracy for min_samples_split")
# plt.xlabel("min_samples_split")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot for min_samples_leaf
# plt.figure(figsize=(10, 6))
# plt.plot(min_samples_leaf_range, train_accuracies_leaf, label='Training Accuracy', marker='o')
# plt.plot(min_samples_leaf_range, test_accuracies_leaf, label='Testing Accuracy', marker='x')
# plt.title("Training and Testing Accuracy for min_samples_leaf")
# plt.xlabel("min_samples_leaf")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

# Step 1: Scale the data (StandardScaler standardizes the features)
scaler = StandardScaler()
spread_df = scaler.fit_transform(spread_df)

# Step 2: Apply PCA
pca = PCA(n_components=5)  # Choose the number of components (e.g., 5 components)
spread_df = pca.fit_transform(spread_df)

# Step 3: Create a DataFrame with the PCA components
spread_df = pd.DataFrame(spread_df, columns=[f'PC{i+1}' for i in range(spread_df.shape[1])])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
log_reg_classifier = LogisticRegression(max_iter=500, random_state=42, multi_class='ovr')

# Train the model
log_reg_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)



# Compare predictions to the actual values
misclassified_data = X_test[y_pred != y_test]

# Add the true labels and predicted labels for better inspection
misclassified_data['True_Label'] = y_test[y_pred != y_test]
misclassified_data['Predicted_Label'] = y_pred[y_pred != y_test]

# Print out the misclassified data
print(misclassified_data)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Perform cross-validation (using 5 folds in this case)
cv_scores = cross_val_score(log_reg_classifier, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Print the mean and standard deviation of the scores
print("Mean accuracy:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))
