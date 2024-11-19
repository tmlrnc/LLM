import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
import autosklearn.classification
import optuna
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.cluster import DBSCAN

class PaysafeFraudDetection:
    def __init__(self):
        self.scaler = StandardScaler()
        self.automl = None
        self.isolation_forest = None
        self.lgbm_model = None
        self.feature_importance = None

    def preprocess_data(self, df):
        """Preprocess transaction data."""
        # Extract time-based features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Calculate transaction velocity features
        df['tx_count_1h'] = df.groupby('user_id')['timestamp'].transform(
            lambda x: x.rolling('1H').count())
        df['tx_amount_1h'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling('1H').sum())
        
        # Location-based features
        df['location_velocity'] = df.groupby('user_id')['location'].transform(
            lambda x: x.diff().fillna(0))
            
        # Device fingerprinting features
        df['device_count'] = df.groupby('user_id')['device_id'].transform('nunique')
        
        # Handle missing values
        df = df.fillna(0)
        
        return df

    def detect_anomalies(self, X):
        """Detect anomalies using Isolation Forest."""
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        anomaly_scores = self.isolation_forest.fit_predict(X)
        return anomaly_scores

    def train_automl(self, X_train, y_train, time_budget=3600):
        """Train AutoML model using auto-sklearn."""
        self.automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_budget,
            per_run_time_limit=300,
            ensemble_size=50,
            metric=autosklearn.metrics.roc_auc
        )
        self.automl.fit(X_train, y_train)

    def optimize_lgbm(self, X_train, y_train, X_val, y_val):
        """Optimize LightGBM using Optuna."""
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(params, train_data, valid_sets=[val_data],
                            num_boost_round=1000, early_stopping_rounds=50,
                            verbose_eval=False)
            
            return model.best_score['valid_0']['auc']

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Train final model with best parameters
        best_params = study.best_params
        train_data = lgb.Dataset(X_train, label=y_train)
        self.lgbm_model = lgb.train(best_params, train_data, num_boost_round=1000)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.lgbm_model.feature_importance()
        }).sort_values('importance', ascending=False)

    def predict_fraud(self, X):
        """Combine predictions from all models."""
        # Get predictions from each model
        automl_pred = self.automl.predict_proba(X)[:, 1]
        anomaly_scores = self.isolation_forest.predict(X)
        lgbm_pred = self.lgbm_model.predict(X)
        
        # Combine predictions (weighted average)
        final_pred = (0.4 * automl_pred + 
                     0.3 * (anomaly_scores == -1).astype(int) +
                     0.3 * lgbm_pred)
        
        return final_pred

    def detect_emerging_patterns(self, X, threshold=0.8):
        """Detect emerging fraud patterns using DBSCAN clustering."""
        # Only cluster high-risk transactions
        high_risk = X[self.predict_fraud(X) > threshold]
        
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(
            self.scaler.transform(high_risk))
        
        # Analyze clusters
        clusters = pd.DataFrame({
            'cluster': clustering.labels_,
            'data': high_risk.values.tolist()
        }).groupby('cluster').agg({
            'data': 'count'
        })
        
        return clusters

    def monitor_performance(self, y_true, y_pred, threshold=0.5):
        """Monitor model performance metrics."""
        predictions = (y_pred > threshold).astype(int)
        
        performance = {
            'classification_report': classification_report(y_true, predictions),
            'confusion_matrix': confusion_matrix(y_true, predictions),
            'feature_importance': self.feature_importance
        }
        
        return performance

def main():
    # Example usage
    # Load your transaction data
    df = pd.read_csv('transactions.csv')
    
    # Initialize the fraud detection system
    fraud_detector = PaysafeFraudDetection()
    
    # Preprocess data
    processed_df = fraud_detector.preprocess_data(df)
    
    # Split features and target
    X = processed_df.drop(['fraud_label', 'timestamp'], axis=1)
    y = processed_df['fraud_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    # Train models
    fraud_detector.detect_anomalies(X_train)
    fraud_detector.train_automl(X_train, y_train)
    fraud_detector.optimize_lgbm(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = fraud_detector.predict_fraud(X_test)
    
    # Monitor performance
    performance = fraud_detector.monitor_performance(y_test, predictions)
    print(performance['classification_report'])
    
    # Detect emerging patterns
    patterns = fraud_detector.detect_emerging_patterns(X_test)
    print("\nEmerging fraud patterns:", patterns)

if __name__ == "__main__":
    main()