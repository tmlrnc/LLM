import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Load the dataset
train_data = pd.read_csv('/Users/thomaslorenc/Sites/eyes/data-scripts-main/data-pull/src/myenv/cyber/train.csv')
test_data = pd.read_csv('/Users/thomaslorenc/Sites/eyes/data-scripts-main/data-pull/src/myenv/cyber/test-3.csv')

# Convert the 'Date' column to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d/%m/%y')

# Encode categorical features
le_home_team = LabelEncoder()
le_away_team = LabelEncoder()
le_ftr = LabelEncoder()

train_data['HomeTeam'] = le_home_team.fit_transform(train_data['HomeTeam'])
train_data['AwayTeam'] = le_away_team.fit_transform(train_data['AwayTeam'])
train_data['FTR'] = le_ftr.fit_transform(train_data['FTR'])

# Prepare features and target
X = train_data.drop(columns=['FTR', 'Date', 'league'])
y = train_data['FTR']

# Handle missing values (impute missing data)
imputer = SimpleImputer(strategy='mean')  # You can use 'mean', 'median', or 'most_frequent'
X = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function to evaluate and display model performance
def evaluate_model(y_test, y_pred, model_name):
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

# Train RandomForestClassifier
def train_random_forest(X_train, y_train, X_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model.predict(X_test)

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

# Train Logistic Regression
def train_logistic_regression(X_train, y_train, X_test):
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
    return lr_model.predict(X_test)

# Main function to compare models
def main():
    # Random Forest
    y_pred_rf = train_random_forest(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred_rf, "Random Forest")
    
    # XGBoost with Grid Search
    best_xgb_model = grid_search_xgboost(X_train, y_train)
    y_pred_xgb = best_xgb_model.predict(X_test)
    evaluate_model(y_test, y_pred_xgb, "XGBoost with Grid Search")
    
    # Logistic Regression
    y_pred_lr = train_logistic_regression(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred_lr, "Logistic Regression")

if __name__ == "__main__":
    main()
