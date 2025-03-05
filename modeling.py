import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the cleaned data
df = pd.read_csv("data/nfl_spread_with_team_stats.csv")

# Select relevant numerical columns for correlation analysis
numerical_cols = ['score_home', 'score_away', 'spread_favorite', 'home_PF', 'home_Yds', 'home_TO', 'home_1stD',
                  'away_PF', 'away_Yds', 'away_TO', 'away_1stD', 'weather_temperature', 'weather_wind_mph', 'weather_humidity']

# Calculate the correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Plotting the heatmap for correlations
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Matrix of Numerical Features')
# plt.show()

# # Optional: Create histograms for some key features
# df[numerical_cols].hist(figsize=(12, 10), bins=20)
# plt.suptitle('Histograms of Numerical Features')
# plt.show()

# # Boxplot of the spread_favorite to see distribution
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='spread_favorite', data=df)
# plt.title('Distribution of Spread Favorite')
# plt.show()

# Create a new column indicating whether the home team won (1) or lost (0)
df['home_win'] = (df['score_home'] > df['score_away']).astype(int)

# Define the features for the y-axis that are unrelated to the score
y_features = ['home_Yds', 'home_TO', 'home_1stD', 'away_Yds']

# Set up a grid of subplots (2 rows x 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Filter data for wins (home_win == 1) and losses (home_win == 0)
df_win = df[df['home_win'] == 1]
df_loss = df[df['home_win'] == 0]

# Loop through each feature and create a scatter plot for wins and losses
# Define the features for the y-axis that are unrelated to the score
y_features = ['home_Yds', 'home_TO', 'home_1stD', 'away_Yds']

# Set up a grid of subplots (2 rows x 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Filter data for wins (home_win == 1) and losses (home_win == 0)
df['home_win'] = (df['score_home'] > df['score_away']).astype(int)

# Define the features for the y-axis that are unrelated to the score
y_features = ['home_Yds', 'home_TO', 'home_1stD', 'away_Yds']


# Filter data for wins (home_win == 1) and losses (home_win == 0)
df_win = df[df['home_win'] == 1]
df_loss = df[df['home_win'] == 0]


import seaborn as sns
import matplotlib.pyplot as plt

# Define numerical features to visualize in the pairplot
pairplot_features = ['spread_favorite', 'home_Yds', 'home_TO', 'home_1stD', 'away_Yds', 'home_win']

# Create the pairplot with hue set to 'home_win' for color distinction
sns.pairplot(df[pairplot_features], hue='home_win', palette={1: 'green', 0: 'red'}, plot_kws={'alpha': 0.6, 's': 10})

# Show the plot
plt.show()