# models/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Using alternative models.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not installed. Using alternative models.")

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("⚠️ imbalanced-learn not installed. Using basic sampling.")

class AdvancedHealthRiskPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.feature_importance = {}
        self.model_performance = {}
        self.model_metadata = {}
        
        # Create directories
        self.model_dir = 'models/saved_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def create_medical_features(self, df):
        """Create medically relevant features"""
        print("🔧 Creating medical features...")
        
        # Basic health indicators
        if 'bmi' not in df.columns and all(col in df.columns for col in ['weight', 'height']):
            df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
        
        # Blood pressure features
        if all(col in df.columns for col in ['systolic_bp', 'diastolic_bp']):
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
            df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
            df['bp_category'] = pd.cut(df['systolic_bp'], 
                                      bins=[0, 120, 130, 140, 180, 300],
                                      labels=['normal', 'elevated', 'stage1', 'stage2', 'crisis'])
        
        # Cholesterol ratios
        if all(col in df.columns for col in ['cholesterol', 'hdl', 'ldl']):
            df['total_hdl_ratio'] = df['cholesterol'] / np.maximum(df['hdl'], 1)
            df['ldl_hdl_ratio'] = df['ldl'] / np.maximum(df['hdl'], 1)
            df['non_hdl_cholesterol'] = df['cholesterol'] - df['hdl']
        
        # Age risk groups
        if 'age' in df.columns:
            df['age_risk_group'] = pd.cut(df['age'], 
                                         bins=[0, 40, 50, 60, 70, 100],
                                         labels=['low', 'moderate', 'high', 'very_high', 'extreme'])
        
        # Glucose categories
        if 'glucose' in df.columns:
            df['glucose_category'] = pd.cut(df['glucose'],
                                           bins=[0, 100, 126, 200, 500],
                                           labels=['normal', 'prediabetic', 'diabetic', 'very_high'])
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        encodings = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'smoking_status': {'never': 0, 'former': 1, 'current': 2},
            'alcohol_consumption': {'none': 0, 'occasional': 1, 'moderate': 2, 'heavy': 3},
            'physical_activity': {'sedentary': 0, 'light': 1, 'moderate': 2, 'active': 3},
            'diet_type': {'poor': 0, 'fair': 1, 'good': 2, 'excellent': 3}
        }
        
        for col, mapping in encodings.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].map(mapping).fillna(0)
        
        # One-hot encode remaining categorical features
        categorical_cols = ['bp_category', 'age_risk_group', 'glucose_category']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Create medical features
        df = self.create_medical_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variables and identifiers
        exclude_patterns = ['has_', 'id', 'date', 'name', '_id']
        feature_cols = [col for col in numeric_cols 
                       if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        # Remove original categorical columns
        categorical_originals = ['gender', 'smoking_status', 'alcohol_consumption', 
                               'physical_activity', 'diet_type', 'bp_category', 
                               'age_risk_group', 'glucose_category']
        feature_cols = [col for col in feature_cols if col not in categorical_originals]
        
        self.feature_columns = feature_cols
        return df[feature_cols].fillna(0)  # Fill NaN with 0
    
    def balance_dataset(self, X, y):
        """Balance dataset using SMOTE or simple oversampling"""
        if HAS_IMBALANCED:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        else:
            # Simple oversampling
            from sklearn.utils import resample
            
            # Separate classes
            X_majority = X[y == 0]
            X_minority = X[y == 1]
            y_majority = y[y == 0]
            y_minority = y[y == 1]
            
            # Oversample minority class
            n_samples = len(y_majority)
            X_minority_upsampled = resample(X_minority, 
                                          replace=True, 
                                          n_samples=n_samples, 
                                          random_state=42)
            y_minority_upsampled = resample(y_minority, 
                                          replace=True, 
                                          n_samples=n_samples, 
                                          random_state=42)
            
            # Combine
            X_balanced = np.vstack([X_majority, X_minority_upsampled])
            y_balanced = np.hstack([y_majority, y_minority_upsampled])
        
        return X_balanced, y_balanced
    
    def train_single_model(self, X_train, X_test, y_train, y_test, model_name, model):
        """Train and evaluate a single model"""
        print(f"  Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0])
        else:
            feature_importance = np.zeros(X_train.shape[1])
        
        return model, metrics, feature_importance
    
    def train_disease_model(self, X, y, disease_name):
        """Train ensemble model for a specific disease"""
        print(f"\n🏥 Training {disease_name} Risk Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[disease_name] = scaler
        
        # Balance training data
        X_train_balanced, y_train_balanced = self.balance_dataset(X_train_scaled, y_train)
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        # Add optional models
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        # Train all models
        trained_models = {}
        all_metrics = {}
        all_importances = {}
        
        for name, model in models.items():
            trained_model, metrics, importance = self.train_single_model(
                X_train_balanced, X_test_scaled, 
                y_train_balanced, y_test, 
                name, model
            )
            
            trained_models[name] = trained_model
            all_metrics[name] = metrics
            all_importances[name] = importance
            
            print(f"    ✓ {name}: Accuracy={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")
        
        # Create ensemble with top 3 models
        sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['auc'], reverse=True)
        top_models = [(name, trained_models[name]) for name, _ in sorted_models[:3]]
        
        ensemble = VotingClassifier(estimators=top_models, voting='soft')
        print(f"  Training ensemble model...")
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        y_prob_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]
        
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
            'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
            'f1': f1_score(y_test, y_pred_ensemble, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob_ensemble) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        print(f"  ✅ Ensemble: Accuracy={ensemble_metrics['accuracy']:.3f}, AUC={ensemble_metrics['auc']:.3f}")
        
        # Store results
        self.models[disease_name] = ensemble
        self.model_performance[disease_name] = {
            'individual_models': all_metrics,
            'ensemble': ensemble_metrics
        }
        
        # Calculate average feature importance
        avg_importance = np.mean(list(all_importances.values()), axis=0)
        self.feature_importance[disease_name] = avg_importance
        
        return ensemble
    
    def train_all_models(self, df):
        """Train models for all diseases"""
        print("🚀 Starting model training...")
        
        # Prepare features
        X = self.prepare_features(df.copy())
        
        # Define target variables
        disease_targets = {
            'diabetes': 'has_diabetes',
            'heart_disease': 'has_heart_disease',
            'hypertension': 'has_hypertension'
        }
        
        # Train model for each disease
        for disease, target_col in disease_targets.items():
            if target_col in df.columns:
                y = df[target_col].fillna(0).astype(int)
                self.train_disease_model(X, y, disease)
        
        # Save models
        self.save_models()
        
        print("\n✅ All models trained successfully!")
        
    def save_models(self):
        """Save all trained models and metadata"""
        print("\n💾 Saving models...")
        
        # Save individual models
        for disease, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_dir, f'{disease}_model.pkl'))
            joblib.dump(self.scalers[disease], os.path.join(self.model_dir, f'{disease}_scaler.pkl'))
        
        # Save feature