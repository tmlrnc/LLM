import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
import xgboost as xgb
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Load the dataset
train_data = pd.read_csv('/Users/thomaslorenc/Sites/eyes/data-scripts-main/data-pull/src/myenv/cyber/train.csv')
test_data = pd.read_csv('/Users/thomaslorenc/Sites/eyes/data-scripts-main/data-pull/src/myenv/cyber/test-3.csv')

# Convert the 'Date' column to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%y')

# Filter training data to exclude 2017-18 and keep 2009-2016 for training
train_data_filtered = train_data[train_data['Date'].dt.year < 2017]

# Separate the data for 2017-18 for testing
test_data_filtered = train_data[train_data['Date'].dt.year >= 2017]

# Encode categorical features
le_home_team = LabelEncoder()
le_away_team = LabelEncoder()
le_ftr = LabelEncoder()

train_data_filtered['HomeTeam'] = le_home_team.fit_transform(train_data_filtered['HomeTeam'])
train_data_filtered['AwayTeam'] = le_away_team.fit_transform(train_data_filtered['AwayTeam'])
train_data_filtered['FTR'] = le_ftr.fit_transform(train_data_filtered['FTR'])

# Use .loc to avoid SettingWithCopyWarning
test_data_filtered.loc[:, 'HomeTeam'] = le_home_team.transform(test_data_filtered['HomeTeam'])
test_data_filtered.loc[:, 'AwayTeam'] = le_away_team.transform(test_data_filtered['AwayTeam'])
test_data_filtered.loc[:, 'FTR'] = le_ftr.transform(test_data_filtered['FTR'])

# Prepare features and target for both training and testing
X_train = train_data_filtered.drop(columns=['FTR', 'Date', 'league'])
y_train = train_data_filtered['FTR']

X_test = test_data_filtered.drop(columns=['FTR', 'Date', 'league'])
y_test = test_data_filtered['FTR']

# Define imputer
imputer = SimpleImputer(strategy='mean')

# Handle missing values (impute missing data)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=train_data_filtered.drop(columns=['FTR', 'Date', 'league']).columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=test_data_filtered.drop(columns=['FTR', 'Date', 'league']).columns)

# Scale features and retain DataFrame structure
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Helper function to evaluate and display model performance
def evaluate_model(y_test, y_pred, model_name):
    # Ensure both y_test and y_pred are of integer types
    y_test = np.array(y_test, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # Ensure that the labels in y_test and y_pred are consistent
    print("Unique values in y_test:", np.unique(y_test))
    print("Unique values in y_pred:", np.unique(y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, RMSE: {rmse:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_ftr.classes_, yticklabels=le_ftr.classes_)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Train XGBoost using grid search
def grid_search_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Parameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Set up grid search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_

# Function to simulate a match and predict the outcome
def simulate_match(home_team, away_team, stats, model):
    """
    Simulates a match by predicting the outcome of the provided home and away teams.
    
    Parameters:
    - home_team: Name of the home team
    - away_team: Name of the away team
    - stats: Dictionary containing other match statistics (e.g. goals, shots, fouls, corners)
    - model: Trained XGBoost model
    
    Returns:
    - Predicted outcome: Home Win, Draw, or Away Win
    """
    # Create a dataframe for the simulated match with correct feature order
    match_df = pd.DataFrame({
        'HomeTeam': [le_home_team.transform([home_team])[0]],  # Transform the team name to match encoded values
        'AwayTeam': [le_away_team.transform([away_team])[0]],  # Transform the team name to match encoded values
        'HTHG': [stats['HTHG']],  # Half-time home team goals
        'HTAG': [stats['HTAG']],  # Half-time away team goals
        'HS': [stats['HS']],      # Home team shots
        'AS': [stats['AS']],      # Away team shots
        'HST': [stats['HST']],    # Home team shots on target
        'AST': [stats['AST']],    # Away team shots on target
        'HF': [stats['HF']],      # Home team fouls
        'AF': [stats['AF']],      # Away team fouls
        'HY': [stats['HY']],      # Home team yellow cards
        'AY': [stats['AY']],      # Away team yellow cards
        'HR': [stats['HR']],      # Home team red cards
        'AR': [stats['AR']],      # Away team red cards
        'HC': [stats['HC']],      # Home team corners
        'AC': [stats['AC']]       # Away team corners
    })

    # Reorder columns to match the order used during model training
    match_df = match_df[X_train.columns]

    # Handle missing values
    match_df = imputer.transform(match_df)

    # Scale features
    match_df = scaler.transform(match_df)

    # Predict the outcome
    prediction = model.predict(match_df)
    
    # Decode the prediction (0 = Home Win, 1 = Draw, 2 = Away Win)
    outcome = le_ftr.inverse_transform(prediction)[0]
    
    return outcome



# Main function to train and evaluate the XGBoost model
def main():
    best_xgb_model = grid_search_xgboost(X_train, y_train)
    y_pred_xgb = best_xgb_model.predict(X_test)
    evaluate_model(y_test, y_pred_xgb, "XGBoost with Grid Search")
    
    # Simulate a match and predict the outcome
    simulated_stats = {
        'HTHG': 1,  # Half-time home team goals
        'HTAG': 0,  # Half-time away team goals
        'HS': 12,   # Home team shots
        'AS': 8,    # Away team shots
        'HST': 5,   # Home team shots on target
        'AST': 3,   # Away team shots on target
        'HF': 10,   # Home team fouls
        'AF': 15,   # Away team fouls
        'HY': 1,    # Home team yellow cards
        'AY': 2,    # Away team yellow cards
        'HR': 0,    # Home team red cards
        'AR': 0,    # Away team red cards
        'HC': 6,    # Home team corners
        'AC': 4     # Away team corners
    }
    
    # Predict the outcome for a simulated match between Home Team 'Chelsea' and Away Team 'Arsenal'
    outcome = simulate_match('Chelsea', 'Arsenal', simulated_stats, best_xgb_model)
    
    print(f"Predicted outcome for the match: {outcome}")

if __name__ == "__main__":
    main()
