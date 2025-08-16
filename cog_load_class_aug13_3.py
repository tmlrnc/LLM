import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours
import joblib
import warnings
warnings.filterwarnings('ignore')

MODEL_VERSION = "v6.0_enhanced_assessment_aware"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enhanced Label mappings
LABEL_MAP = {
    # LOW category (1-2)
    'Very-low': 0, 'very-low': 0, 'Very-Low': 0, 'VERY-LOW': 0,
    'Low': 0, 'LOW': 0, 'low': 0,
    
    # MODERATE category (3-5)
    'Moderate-low': 1, 'moderate-low': 1, 'Moderate-Low': 1, 'MODERATE-LOW': 1,
    'Moderate': 1, 'MODERATE': 1, 'moderate': 1, 'Medium': 1, 'medium': 1, 'MEDIUM': 1,
    'Moderate-high': 1, 'moderate-high': 1, 'Moderate-High': 1, 'MODERATE-HIGH': 1,
    
    # HIGH category (6-7)
    'High': 2, 'HIGH': 2, 'high': 2,
    'Very-high': 2, 'very-high': 2, 'Very-High': 2, 'VERY-HIGH': 2,
    
    # Numeric mappings
    0: 0, 1: 1, 2: 2, 0.0: 0, 1.0: 1, 2.0: 2,
    '0': 0, '1': 1, '2': 2, '0.0': 0, '1.0': 1, '2.0': 2
}
REVERSE_LABEL_MAP = {0: 'Low', 1: 'Moderate', 2: 'High'}

class AdvancedFeatureEngineer:
    """Advanced feature engineering with domain-specific cognitive load features"""
    
    def __init__(self):
        self.feature_selectors = {}
        self.feature_stats = {}
        self.cognitive_patterns = {}
        
    def create_cognitive_domain_features(self, df, feature_names):
        """Create domain-specific cognitive load features"""
        enhanced_df = df.copy()
        
        # 1. PUPIL-BASED COGNITIVE LOAD INDICATORS
        if 'pupil_diameter_mean_5sec.b' in feature_names:
            pupil_col = 'pupil_diameter_mean_5sec.b'
            
            # Pupil dilation patterns
            enhanced_df[f'{pupil_col}_pct_change'] = df[pupil_col].pct_change().fillna(0)
            enhanced_df[f'{pupil_col}_volatility'] = df[pupil_col].rolling(10, min_periods=1).std()
            enhanced_df[f'{pupil_col}_trend_strength'] = abs(df[pupil_col].diff(5))
            
            # Pupil load zones (based on cognitive load research)
            pupil_mean = df[pupil_col].mean()
            pupil_std = df[pupil_col].std()
            enhanced_df[f'{pupil_col}_load_zone'] = pd.cut(
                df[pupil_col], 
                bins=[0, pupil_mean - pupil_std, pupil_mean + pupil_std, float('inf')],
                labels=[0, 1, 2]
            ).astype(float)
        
        # 2. EYE MOVEMENT COGNITIVE INDICATORS
        if 'saccade_num.b' in feature_names:
            saccade_col = 'saccade_num.b'
            
            # Saccadic cognitive load patterns
            enhanced_df[f'{saccade_col}_burst_intensity'] = df[saccade_col].rolling(5, min_periods=1).sum()
            enhanced_df[f'{saccade_col}_rest_periods'] = (df[saccade_col] == 0).astype(int)
            enhanced_df[f'{saccade_col}_activity_ratio'] = df[saccade_col] / (df[saccade_col].rolling(10, min_periods=1).mean() + 1e-8)
        
        # 3. BLINK-BASED ATTENTION INDICATORS
        if 'blink_duration_avg.b' in feature_names and 'interblink_interval_avg.b' in feature_names:
            blink_dur = 'blink_duration_avg.b'
            blink_int = 'interblink_interval_avg.b'
            
            # Attention regulation patterns
            enhanced_df['blink_efficiency'] = df[blink_int] / (df[blink_dur] + 1e-8)
            enhanced_df['attention_stability'] = 1 / (df[blink_dur].rolling(5, min_periods=1).std() + 1e-8)
            enhanced_df['cognitive_effort_proxy'] = df[blink_dur] * df['saccade_num.b'] if 'saccade_num.b' in feature_names else df[blink_dur]
        
        # 4. AUTONOMIC NERVOUS SYSTEM INDICATORS
        if 'LHIPA_20sec' in feature_names:
            ans_col = 'LHIPA_20sec'
            
            # ANS activation patterns
            enhanced_df[f'{ans_col}_activation_level'] = pd.qcut(df[ans_col], q=3, labels=[0, 1, 2], duplicates='drop').astype(float)
            enhanced_df[f'{ans_col}_stability'] = 1 / (df[ans_col].rolling(10, min_periods=1).std() + 1e-8)
            enhanced_df[f'{ans_col}_reactivity'] = abs(df[ans_col].diff())
        
        # 5. MULTIMODAL COGNITIVE LOAD COMPOSITE FEATURES
        key_features = ['pupil_diameter_mean_5sec.b', 'LHIPA_20sec', 'saccade_num.b', 'blink_duration_avg.b']
        available_features = [f for f in key_features if f in feature_names]
        
        if len(available_features) >= 2:
            # Create composite cognitive load index
            feature_values = df[available_features].values
            feature_normalized = (feature_values - feature_values.mean(axis=0)) / (feature_values.std(axis=0) + 1e-8)
            
            # Primary cognitive load index (weighted combination)
            weights = np.array([0.35, 0.25, 0.25, 0.15])[:len(available_features)]  # Pupil gets highest weight
            enhanced_df['cognitive_load_index'] = np.dot(feature_normalized, weights)
            
            # Secondary indices
            enhanced_df['arousal_index'] = feature_normalized[:, :2].mean(axis=1) if len(available_features) >= 2 else feature_normalized[:, 0]
            enhanced_df['attention_index'] = feature_normalized[:, 2:].mean(axis=1) if len(available_features) >= 3 else feature_normalized[:, -1]
        
        # 6. TEMPORAL COGNITIVE STATE FEATURES
        if len(enhanced_df) > 10:
            # Cognitive state transitions
            for feature in available_features:
                enhanced_df[f'{feature}_state_change'] = (df[feature].diff().abs() > df[feature].std()).astype(int)
                enhanced_df[f'{feature}_stability_score'] = df[feature].rolling(10, min_periods=1).std()
        
        return enhanced_df
    
    def apply_advanced_feature_selection(self, X, y, feature_names, target_features=None):
        """Apply multiple feature selection techniques"""
        if target_features is None:
            target_features = min(len(feature_names), 100)  # Reasonable upper limit
        
        print(f"🎯 Applying advanced feature selection: {len(feature_names)} -> {target_features} features")
        
        # 1. Remove highly correlated features
        corr_matrix = pd.DataFrame(X, columns=feature_names).corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        
        remaining_features = [f for f in feature_names if f not in high_corr_features]
        remaining_indices = [i for i, f in enumerate(feature_names) if f in remaining_features]
        X_reduced = X[:, remaining_indices]
        
        print(f"   Removed {len(high_corr_features)} highly correlated features")
        
        # 2. Univariate feature selection
        selector_univariate = SelectKBest(score_func=f_classif, k=min(target_features, len(remaining_features)))
        X_univariate = selector_univariate.fit_transform(X_reduced, y)
        univariate_features = [remaining_features[i] for i in selector_univariate.get_support(indices=True)]
        
        # 3. Mutual information selection
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(target_features, len(remaining_features)))
        X_mi = selector_mi.fit_transform(X_reduced, y)
        mi_features = [remaining_features[i] for i in selector_mi.get_support(indices=True)]
        
        # 4. RFE with Random Forest
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe_selector = RFE(estimator=rf_selector, n_features_to_select=min(target_features, len(remaining_features)))
        X_rfe = rfe_selector.fit_transform(X_reduced, y)
        rfe_features = [remaining_features[i] for i in rfe_selector.get_support(indices=True)]
        
        # 5. Combine selections using voting
        feature_votes = {}
        for feature in remaining_features:
            votes = 0
            if feature in univariate_features:
                votes += 1
            if feature in mi_features:
                votes += 1
            if feature in rfe_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Select features with at least 2 votes, then top features by votes
        selected_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        final_features = [f for f, votes in selected_features if votes >= 2]
        
        # If not enough features with 2+ votes, add top single-vote features
        if len(final_features) < target_features:
            single_vote_features = [f for f, votes in selected_features if votes == 1]
            final_features.extend(single_vote_features[:target_features - len(final_features)])
        
        # Limit to target number
        final_features = final_features[:target_features]
        
        final_indices = [feature_names.index(f) for f in final_features]
        X_final = X[:, final_indices]
        
        print(f"   Final selection: {len(final_features)} features")
        print(f"   Feature vote distribution: {dict(pd.Series([v for f, v in feature_votes.items()]).value_counts().sort_index())}")
        
        self.feature_selectors['selected_features'] = final_features
        return X_final, final_features

class EnhancedCognitivePredictor:
    """Enhanced predictor with advanced cognitive modeling"""
    
    def __init__(self):
        self.class_confidence_thresholds = {0: 0.6, 1: 0.5, 2: 0.6}  # Different thresholds per class
        self.transition_smoothing = 0.3
        self.stability_memory = []
        self.prediction_history = []
        
    def predict_with_confidence_calibration(self, probabilities_sequence, true_labels=None):
        """Enhanced prediction with confidence calibration and class-specific thresholds"""
        predictions = []
        calibrated_confidences = []
        
        for i, probs in enumerate(probabilities_sequence):
            # Apply temperature scaling for confidence calibration
            temperature = 1.5  # Learned parameter to calibrate confidence
            calibrated_probs = np.exp(np.log(probs + 1e-8) / temperature)
            calibrated_probs = calibrated_probs / calibrated_probs.sum()
            
            # Get base prediction
            base_pred = np.argmax(calibrated_probs)
            base_conf = calibrated_probs[base_pred]
            
            # Apply class-specific confidence thresholds
            confident_pred = base_pred
            if base_conf < self.class_confidence_thresholds[base_pred]:
                # Low confidence - consider second best or apply smoothing
                sorted_indices = np.argsort(calibrated_probs)[::-1]
                second_best = sorted_indices[1]
                second_conf = calibrated_probs[second_best]
                
                # If second best is close, bias toward moderate class (class 1)
                if abs(base_conf - second_conf) < 0.1:
                    confident_pred = 1  # Default to moderate when uncertain
                
            # Temporal smoothing
            if len(self.prediction_history) > 0:
                recent_preds = self.prediction_history[-3:]
                if len(set(recent_preds)) == 1 and recent_preds[0] != confident_pred:
                    # Strong temporal consistency - apply smoothing
                    smoothed_prob = self.transition_smoothing
                    confident_pred = int(smoothed_prob * recent_preds[0] + (1 - smoothed_prob) * confident_pred)
            
            predictions.append(confident_pred)
            calibrated_confidences.append(base_conf)
            
            # Update history
            self.prediction_history.append(confident_pred)
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
        
        return np.array(predictions), np.array(calibrated_confidences)

# Base class definition (since CogLoadModel is not available)
class CogLoadModel:
    """Base class for cognitive load models"""
    
    def __init__(self):
        pass
    
    def train(self):
        raise NotImplementedError("Subclasses must implement train method")
    
    def test(self):
        raise NotImplementedError("Subclasses must implement test method")
    
    def run_model_blind_test(self, df: pd.DataFrame) -> list:
        raise NotImplementedError("Subclasses must implement run_model_blind_test method")
    
    def verify_valid_model(self, output_dir):
        raise NotImplementedError("Subclasses must implement verify_valid_model method")

# Placeholder for get_total_data function
def get_total_data(dataset_name, features, include_assessment_id=False):
    """
    Placeholder function - replace with your actual data loading logic
    Should return a list of pandas DataFrames, one per assessment
    """
    print(f"⚠️ Using placeholder data loader for {dataset_name}")
    print("Replace this function with your actual data loading logic")
    
    # Create sample data for demonstration
    sample_data = []
    for i in range(10):  # 10 sample assessments
        n_samples = np.random.randint(50, 200)
        data = {
            'blink_duration_avg.b': np.random.normal(100, 20, n_samples),
            'interblink_interval_avg.b': np.random.normal(500, 100, n_samples),
            'blink_duration_stdev.b': np.random.normal(50, 10, n_samples),
            'LHIPA_20sec': np.random.normal(0.5, 0.2, n_samples),
            'avg_amplitude_both': np.random.normal(2.0, 0.5, n_samples),
            'saccade_num.b': np.random.poisson(3, n_samples),
            'pupil_diameter_mean_5sec.b': np.random.normal(3.5, 0.8, n_samples),
            'pupil_diameter_stdev_5sec.b': np.random.normal(0.5, 0.1, n_samples),
            'ground_truth_cl': np.random.choice(['Low', 'Moderate', 'High'], n_samples)
        }
        sample_data.append(pd.DataFrame(data))
    
    return sample_data

class CogLoad6_0_Enhanced(CogLoadModel):
    """Enhanced assessment-aware cognitive load classifier targeting 60%+ accuracy"""
    
    DATA_SET = "combined_pupil_labs"
    FEATURES = [
        "blink_duration_avg.b",
        "interblink_interval_avg.b", 
        "blink_duration_stdev.b",
        "LHIPA_20sec",
        "avg_amplitude_both",
        "saccade_num.b",
        "pupil_diameter_mean_5sec.b",
        "pupil_diameter_stdev_5sec.b",
        "ground_truth_cl"
    ]
    
    def __init__(self):
        assert self.DATA_SET != "", "Please Fill in the Data Set"
        assert self.FEATURES != [], "Please Fill in the Features"
        
        # Load ALL available data
        self.total_data: list[pd.DataFrame] = get_total_data(
            self.DATA_SET,
            self.FEATURES,
            include_assessment_id=False
        )
        
        # Initialize enhanced components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.cognitive_predictor = EnhancedCognitivePredictor()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.class_weights = None
        
        print(f"🚀 ENHANCED COGNITIVE LOAD CLASSIFIER v6.0")
        print(f"📊 Total assessments: {len(self.total_data)}")
        
        # Prepare enhanced data splits
        self._prepare_enhanced_assessment_splits()
    
    def _prepare_enhanced_assessment_splits(self):
        """Enhanced data preparation with better balancing and feature engineering"""
        print(f"\n🎯 ENHANCED ASSESSMENT-AWARE DATA PREPARATION...")
        
        # Clean and analyze data
        clean_assessments = []
        total_class_counts = {0: 0, 1: 0, 2: 0}
        
        for assessment_id, df in enumerate(self.total_data):
            clean_df = df.dropna()
            y_mapped = clean_df["ground_truth_cl"].map(LABEL_MAP)
            valid_mask = ~y_mapped.isnull()
            
            clean_df = clean_df[valid_mask].copy()
            clean_df["ground_truth_cl"] = y_mapped[valid_mask].astype(int)
            
            if len(clean_df) > 5:  # Minimum samples per assessment
                clean_assessments.append(clean_df)
                class_counts = clean_df["ground_truth_cl"].value_counts()
                for cls in [0, 1, 2]:
                    total_class_counts[cls] += class_counts.get(cls, 0)
        
        print(f"✅ Clean assessments: {len(clean_assessments)}")
        print(f"📊 Overall class distribution: {total_class_counts}")
        
        # Calculate class weights for imbalanced data
        total_samples = sum(total_class_counts.values())
        self.class_weights = {
            cls: total_samples / (3 * count) if count > 0 else 1.0 
            for cls, count in total_class_counts.items()
        }
        print(f"⚖️ Class weights: {self.class_weights}")
        
        # Enhanced stratified split
        np.random.seed(42)
        n_assessments = len(clean_assessments)
        indices = np.arange(n_assessments)
        np.random.shuffle(indices)
        
        # 60% train, 20% val, 20% blind
        n_train = int(0.6 * n_assessments)
        n_val = int(0.2 * n_assessments)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        blind_indices = indices[n_train + n_val:]
        
        # Create assessment splits
        self.train_assessments = [clean_assessments[i] for i in train_indices]
        self.val_assessments = [clean_assessments[i] for i in val_indices]
        self.blind_test_assessments = [clean_assessments[i] for i in blind_indices]
        
        print(f"🚂 Enhanced splits: {len(self.train_assessments)} train, {len(self.val_assessments)} val, {len(self.blind_test_assessments)} blind")
        
        # Enhanced feature engineering
        base_feature_names = [col for col in self.FEATURES if col != "ground_truth_cl"]
        
        # Apply advanced feature engineering to training data
        print("🧠 Applying advanced feature engineering...")
        enhanced_train_assessments = []
        for df in self.train_assessments:
            enhanced_df = self.feature_engineer.create_cognitive_domain_features(df, base_feature_names)
            enhanced_train_assessments.append(enhanced_df)
        
        # Combine training data for feature selection
        train_combined = pd.concat(enhanced_train_assessments, ignore_index=True)
        
        # Extract features for selection
        feature_columns = [col for col in train_combined.columns if col != "ground_truth_cl"]
        X_train_full = train_combined[feature_columns].fillna(0).values
        y_train_full = train_combined["ground_truth_cl"].values
        
        print(f"📈 Generated {len(feature_columns)} enhanced features")
        
        # Apply advanced feature selection
        X_train_selected, selected_features = self.feature_engineer.apply_advanced_feature_selection(
            X_train_full, y_train_full, feature_columns, target_features=50
        )
        
        self.feature_names = selected_features
        
        # Apply same feature engineering and selection to validation and blind test
        enhanced_val_assessments = []
        for df in self.val_assessments:
            enhanced_df = self.feature_engineer.create_cognitive_domain_features(df, base_feature_names)
            enhanced_val_assessments.append(enhanced_df)
        
        enhanced_blind_assessments = []
        for df in self.blind_test_assessments:
            enhanced_df = self.feature_engineer.create_cognitive_domain_features(df, base_feature_names)
            enhanced_blind_assessments.append(enhanced_df)
        
        # Prepare final datasets
        self.X_train = train_combined[self.feature_names].fillna(0).values
        self.y_train = train_combined["ground_truth_cl"].values
        
        if enhanced_val_assessments:
            val_combined = pd.concat(enhanced_val_assessments, ignore_index=True)
            self.X_val = val_combined[self.feature_names].fillna(0).values
            self.y_val = val_combined["ground_truth_cl"].values
            self.val_assessments_enhanced = enhanced_val_assessments
        else:
            self.X_val = np.array([]).reshape(0, len(self.feature_names))
            self.y_val = np.array([])
            self.val_assessments_enhanced = []
        
        self.blind_test_assessments_enhanced = enhanced_blind_assessments
        
        print(f"✅ ENHANCED DATA READY: {len(self.X_train)} train samples, {len(self.feature_names)} selected features")
    
    def train(self):
        """Enhanced training with robust techniques"""
        print("🚀 ENHANCED ASSESSMENT-AWARE TRAINING...")
        
        # Robust data balancing
        balancing_techniques = {
            'SMOTE': SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(self.y_train)))),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=min(5, len(np.unique(self.y_train)))),
            'SMOTEENN': SMOTEENN(random_state=42)
        }
        
        best_balanced_data = None
        best_score = 0
        best_technique = None
        
        # Evaluate balancing techniques
        quick_rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        
        for name, technique in balancing_techniques.items():
            try:
                X_balanced, y_balanced = technique.fit_resample(self.X_train, self.y_train)
                scores = cross_val_score(quick_rf, X_balanced, y_balanced, cv=3, scoring='balanced_accuracy')
                avg_score = scores.mean()
                
                print(f"   {name:15}: {avg_score:.3f} balanced accuracy")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_balanced_data = (X_balanced, y_balanced)
                    best_technique = name
                    
            except Exception as e:
                print(f"   {name:15}: Failed ({str(e)[:30]}...)")
        
        if best_balanced_data is not None:
            X_balanced, y_balanced = best_balanced_data
            print(f"✅ Selected {best_technique} (score: {best_score:.3f})")
        else:
            X_balanced, y_balanced = self.X_train, self.y_train
            print("⚠️ Using original data (no balancing)")
        
        # Enhanced scaling techniques
        self.scalers = {
            'standard': StandardScaler().fit(X_balanced),
            'robust': RobustScaler().fit(X_balanced)
        }
        
        # Robust model configurations
        print("🤖 Training enhanced model ensemble...")
        
        # Enhanced XGBoost with simpler config
        enhanced_xgb = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0,
            objective='multi:softprob',
            num_class=3
        )
        
        # Enhanced LightGBM
        enhanced_lgb = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1,
            objective='multiclass',
            num_class=3,
            class_weight='balanced'
        )
        
        # Enhanced Random Forest
        enhanced_rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Gradient Boosting
        enhanced_gb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        model_configs = [
            ('enhanced_xgb', enhanced_xgb),
            ('enhanced_lgb', enhanced_lgb),
            ('enhanced_rf', enhanced_rf),
            ('enhanced_gb', enhanced_gb)
        ]
        
        # Train models
        for name, model in model_configs:
            try:
                print(f"   🔧 Training {name}...")
                model.fit(X_balanced, y_balanced)
                self.models[name] = model
                print(f"     ✅ {name} trained successfully")
            except Exception as e:
                print(f"     ❌ {name} failed: {str(e)[:50]}...")
        
        # Model evaluation
        print("🎯 Evaluating models...")
        model_scores = {}
        
        use_cv = len(self.X_val) == 0
        
        for name, model in self.models.items():
            try:
                if use_cv:
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='balanced_accuracy')
                    val_score = scores.mean()
                else:
                    val_pred = model.predict(self.X_val)
                    val_score = balanced_accuracy_score(self.y_val, val_pred)
                
                model_scores[name] = val_score
                print(f"   {name:20}: {val_score:.3f}")
                
            except Exception as e:
                print(f"   {name:20}: FAILED - {str(e)[:40]}...")
                if name in self.models:
                    del self.models[name]
        
        # Create ensemble
        print("🏆 Creating ensemble...")
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_models = [(name, self.models[name]) for name, score in sorted_models if score > 0.3]
        
        if len(top_models) >= 2:
            try:
                self.ensemble = VotingClassifier(
                    estimators=top_models[:3],  # Top 3 models
                    voting='soft',
                    n_jobs=1
                )
                self.ensemble.fit(X_balanced, y_balanced)
                print(f"✅ Ensemble created with {len(top_models[:3])} models")
                
            except Exception as e:
                print(f"❌ Ensemble failed: {str(e)[:50]}...")
                if self.models:
                    best_model_name = list(self.models.keys())[0]
                    self.ensemble = self.models[best_model_name]
                    print(f"✅ Using best single model: {best_model_name}")
                else:
                    return False
        else:
            print("❌ Not enough good models for ensemble")
            if self.models:
                best_model_name = list(self.models.keys())[0]
                self.ensemble = self.models[best_model_name]
                print(f"✅ Using fallback model: {best_model_name}")
            else:
                return False
        
        # Save models
        self._save_models()
        print("✅ ENHANCED TRAINING COMPLETE!")
        return True
    
    def test(self):
        """Enhanced testing with cognitive prediction"""
        if not hasattr(self, 'ensemble'):
            print("❌ No trained ensemble found")
            return 0.0
        
        if len(self.X_val) == 0:
            print("⚠️ No validation set available")
            return 0.0
        
        print("📊 ENHANCED VALIDATION TESTING...")
        
        # Standard ensemble prediction
        ensemble_pred = self.ensemble.predict(self.X_val)
        ensemble_proba = self.ensemble.predict_proba(self.X_val)
        
        # Enhanced cognitive prediction
        cognitive_pred, cognitive_conf = self.cognitive_predictor.predict_with_confidence_calibration(
            ensemble_proba
        )
        
        # Calculate accuracies
        ensemble_acc = accuracy_score(self.y_val, ensemble_pred)
        cognitive_acc = accuracy_score(self.y_val, cognitive_pred)
        
        ensemble_bal_acc = balanced_accuracy_score(self.y_val, ensemble_pred)
        cognitive_bal_acc = balanced_accuracy_score(self.y_val, cognitive_pred)
        
        print(f"🏆 Standard ensemble: {ensemble_acc:.3f} (bal: {ensemble_bal_acc:.3f})")
        print(f"🧠 + Cognitive enhancement: {cognitive_acc:.3f} (bal: {cognitive_bal_acc:.3f})")
        
        # Use better performing approach
        if cognitive_acc > ensemble_acc:
            best_pred = cognitive_pred
            best_acc = cognitive_acc
            print("🎯 Using cognitive-enhanced predictions")
        else:
            best_pred = ensemble_pred
            best_acc = ensemble_acc
            print("🎯 Using standard ensemble predictions")
        
        # Enhanced confusion matrix
        cm = confusion_matrix(self.y_val, best_pred)
        labels = ['Low', 'Moderate', 'High']
        
        print(f"\n📊 Enhanced Validation Confusion Matrix:")
        print("     Pred", end="")
        for label in labels:
            print(f"  {label[:4]:>4}", end="")
        print("  Total   Acc")
        
        for i, true_label in enumerate(labels):
            total = cm[i].sum()
            acc = cm[i, i] / total if total > 0 else 0
            print(f"     {true_label[:4]:>4}", end="")
            for j in range(len(labels)):
                print(f"  {cm[i, j]:>4}", end="")
            print(f"  {total:>5}  {acc:>3.0%}")
        
        return best_acc
    
    def run_enhanced_blind_test(self):
        """Enhanced blind test targeting 60%+ accuracy"""
        if not hasattr(self, 'ensemble'):
            print("❌ No trained ensemble found")
            return {}
        
        print("🎯 ENHANCED ASSESSMENT-AWARE BLIND TEST...")
        
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        assessment_results = []
        
        for assessment_idx, assessment_df in enumerate(self.blind_test_assessments_enhanced):
            print(f"   Processing assessment {assessment_idx + 1}/{len(self.blind_test_assessments_enhanced)}")
            
            # Reset cognitive predictor for each assessment
            self.cognitive_predictor = EnhancedCognitivePredictor()
            
            # Get features and labels
            X_assessment = assessment_df[self.feature_names].fillna(0).values
            y_true_assessment = assessment_df["ground_truth_cl"].values
            
            # Get ensemble probabilities
            ensemble_proba = self.ensemble.predict_proba(X_assessment)
            
            # Apply enhanced cognitive prediction
            enhanced_pred, enhanced_conf = self.cognitive_predictor.predict_with_confidence_calibration(
                ensemble_proba, y_true_assessment
            )
            
            # Calculate assessment accuracy
            assessment_acc = accuracy_score(y_true_assessment, enhanced_pred)
            assessment_bal_acc = balanced_accuracy_score(y_true_assessment, enhanced_pred)
            
            assessment_results.append({
                'assessment_id': assessment_idx,
                'samples': len(y_true_assessment),
                'accuracy': assessment_acc,
                'balanced_accuracy': assessment_bal_acc,
                'avg_confidence': enhanced_conf.mean()
            })
            
            all_predictions.extend(enhanced_pred)
            all_true_labels.extend(y_true_assessment)
            all_confidences.extend(enhanced_conf)
        
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_confidences = np.array(all_confidences)
        
        # Calculate overall performance
        overall_acc = accuracy_score(all_true_labels, all_predictions)
        overall_bal_acc = balanced_accuracy_score(all_true_labels, all_predictions)
        
        print(f"\n🎯 ENHANCED BLIND TEST RESULTS:")
        print(f"🔍 Total samples: {len(all_predictions)}")
        print(f"🎯 Overall accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")
        print(f"⚖️ Overall balanced accuracy: {overall_bal_acc:.3f} ({overall_bal_acc*100:.1f}%)")
        print(f"🔮 Average confidence: {all_confidences.mean():.3f}")
        
        # Enhanced confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        labels = ['Low', 'Moderate', 'High']
        
        print(f"\n📊 ENHANCED BLIND TEST Confusion Matrix:")
        print("     Pred", end="")
        for label in labels:
            print(f"  {label[:4]:>4}", end="")
        print("  Total   Acc")
        
        for i, true_label in enumerate(labels):
            total = cm[i].sum()
            acc = cm[i, i] / total if total > 0 else 0
            
            print(f"     {true_label[:4]:>4}", end="")
            for j in range(len(labels)):
                print(f"  {cm[i, j]:>4}", end="")
            print(f"  {total:>5}  {acc:>3.0%}")
        
        # Check if target achieved
        if overall_acc >= 0.60:
            print(f"\n🏆 TARGET ACHIEVED! {overall_acc*100:.1f}% >= 60%")
        else:
            gap = 0.60 - overall_acc
            print(f"\n📈 Gap to 60%: {gap:.3f} ({gap/overall_acc*100:.1f}% relative improvement needed)")
        
        return {
            'overall_accuracy': overall_acc,
            'overall_balanced_accuracy': overall_bal_acc,
            'average_confidence': all_confidences.mean(),
            'per_assessment_results': assessment_results,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'confidences': all_confidences
        }
    
    def run_model_blind_test(self, df: pd.DataFrame) -> list:
        """Run blind test with enhanced cognitive prediction on new data"""
        if not hasattr(self, 'ensemble'):
            print("❌ No trained ensemble found")
            return []
        
        base_feature_names = [col for col in self.FEATURES if col != "ground_truth_cl"]
        
        # Check for required features
        missing_features = [col for col in base_feature_names if col not in df.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return []
        
        # Apply enhanced feature engineering
        enhanced_df = self.feature_engineer.create_cognitive_domain_features(df, base_feature_names)
        
        # Get selected features
        X_blind = enhanced_df[self.feature_names].fillna(0)
        
        # Remove rows with missing values
        clean_mask = ~X_blind.isnull().any(axis=1)
        X_blind_clean = X_blind[clean_mask].values
        
        print(f"🔍 ENHANCED BLIND TEST:")
        print(f"   📝 Total samples: {len(df)}")
        print(f"   ✅ Clean samples: {len(X_blind_clean)}")
        print(f"   🧠 Enhanced features: {len(self.feature_names)}")
        
        if len(X_blind_clean) == 0:
            return []
        
        # Get ensemble probabilities
        ensemble_proba = self.ensemble.predict_proba(X_blind_clean)
        
        # Apply enhanced cognitive prediction
        self.cognitive_predictor = EnhancedCognitivePredictor()  # Reset predictor
        enhanced_pred, confidence_scores = self.cognitive_predictor.predict_with_confidence_calibration(ensemble_proba)
        
        # Convert to original labels
        predictions = [REVERSE_LABEL_MAP[pred] for pred in enhanced_pred]
        
        print(f"🎯 ENHANCED BLIND TEST COMPLETE: {len(predictions)} predictions")
        print(f"🔮 Average confidence: {confidence_scores.mean():.3f}")
        
        return predictions
    
    def _save_models(self):
        """Save enhanced models"""
        models_to_save = {
            'enhanced_models.joblib': self.models,
            'enhanced_ensemble.joblib': getattr(self, 'ensemble', None),
            'enhanced_scalers.joblib': self.scalers,
            'enhanced_feature_engineer.joblib': self.feature_engineer,
            'enhanced_cognitive_predictor.joblib': self.cognitive_predictor,
            'enhanced_feature_names.joblib': self.feature_names,
            'enhanced_class_weights.joblib': self.class_weights
        }
        
        for filename, model in models_to_save.items():
            if model is not None:
                joblib.dump(model, os.path.join(OUTPUT_DIR, filename))
        
        print(f"💾 Enhanced models saved to {OUTPUT_DIR}")
    
    def verify_valid_model(self, output_dir):
        """Verify model is properly trained and saved"""
        required_files = [
            'enhanced_ensemble.joblib',
            'enhanced_feature_names.joblib',
            'enhanced_feature_engineer.joblib'
        ]
        
        for filename in required_files:
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                print(f"❌ Missing required file: {filename}")
                return False
        
        if not hasattr(self, 'ensemble') or self.ensemble is None:
            print("❌ No trained ensemble found")
            return False
        
        print("✅ Model validation successful")
        return True

if __name__ == "__main__":
    print("🚀 ENHANCED ASSESSMENT-AWARE COGNITIVE LOAD CLASSIFIER v6.0")
    print("🎯 Target: 60%+ Accuracy")
    
    model = CogLoad6_0_Enhanced()
    
    success = model.train()
    
    if success:
        # Test on validation set
        validation_accuracy = model.test()
        
        # Run enhanced blind test
        blind_test_results = model.run_enhanced_blind_test()
        
        if validation_accuracy > 0:
            print(f"\n🔍 Validation Accuracy: {validation_accuracy:.4f} ({validation_accuracy*100:.1f}%)")
        
        if blind_test_results:
            blind_acc = blind_test_results['overall_accuracy']
            print(f"🎯 Enhanced Blind Test Accuracy: {blind_acc:.4f} ({blind_acc*100:.1f}%)")
            
            if blind_acc >= 0.60:
                print(f"🏆 SUCCESS! Target accuracy achieved: {blind_acc*100:.1f}% >= 60%")
            else:
                improvement_needed = (0.60 - blind_acc) / blind_acc * 100
                print(f"📈 {improvement_needed:.1f}% relative improvement needed to reach 60%")
        
        # Verify model
        assert model.verify_valid_model(OUTPUT_DIR)
        
    else:
        print("❌ Training failed")