import pandas as pd
import numpy as np
import joblib
from typing import Dict
import os

class HealthRiskScorer:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        model_path = 'models/saved_models/'
        
        try:
            self.models['diabetes'] = joblib.load(f'{model_path}diabetes_model.pkl')
            self.models['heart_disease'] = joblib.load(f'{model_path}heart_disease_model.pkl')
            self.scaler = joblib.load(f'{model_path}scaler.pkl')
            self.feature_columns = joblib.load(f'{model_path}feature_columns.pkl')
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run train_model.py first!")
    
    def calculate_risk_scores(self, health_data: Dict) -> Dict:
        """Calculate risk scores for various conditions"""
        
        # Prepare data
        df = pd.DataFrame([health_data])
        
        # Calculate BMI
        df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
        
        # Encode categorical variables
        df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
        df['smoking_encoded'] = df['smoking_status'].map({'never': 0, 'former': 1, 'current': 2})
        df['alcohol_encoded'] = df['alcohol_consumption'].map({'none': 0, 'moderate': 1, 'heavy': 2})
        df['exercise_encoded'] = df['physical_activity'].map({'sedentary': 0, 'moderate': 1, 'active': 2})
        
        # Select features
        X = df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        risk_scores = {}
        
        # Diabetes risk
        diabetes_proba = self.models['diabetes'].predict_proba(X_scaled)[0][1]
        risk_scores['diabetes_risk'] = round(diabetes_proba * 100, 2)
        
        # Heart disease risk
        heart_proba = self.models['heart_disease'].predict_proba(X_scaled)[0][1]
        risk_scores['heart_disease_risk'] = round(heart_proba * 100, 2)
        
        # Calculate other risks using rule-based logic
        risk_scores['stroke_risk'] = self._calculate_stroke_risk(health_data)
        risk_scores['hypertension_risk'] = self._calculate_hypertension_risk(health_data)
        
        # Overall risk
        risk_scores['overall_risk'] = round(np.mean(list(risk_scores.values())), 2)
        
        # Risk category
        if risk_scores['overall_risk'] < 30:
            risk_scores['risk_category'] = "Low"
        elif risk_scores['overall_risk'] < 60:
            risk_scores['risk_category'] = "Medium"
        elif risk_scores['overall_risk'] < 80:
            risk_scores['risk_category'] = "High"
        else:
            risk_scores['risk_category'] = "Critical"
        
        return risk_scores
    
    def _calculate_stroke_risk(self, data: Dict) -> float:
        """Calculate stroke risk using rule-based logic"""
        risk = 0
        
        if data.get('systolic_bp', 120) > 140:
            risk += 30
        if data.get('age', 30) > 55:
            risk += 20
        if data.get('smoking_status', 'never') == 'current':
            risk += 25
        if data.get('physical_activity', 'moderate') == 'sedentary':
            risk += 15
            
        return min(risk, 100)
    
    def _calculate_hypertension_risk(self, data: Dict) -> float:
        """Calculate hypertension risk"""
        risk = 0
        
        if data.get('systolic_bp', 120) > 130:
            risk += 35
        if data.get('diastolic_bp', 80) > 85:
            risk += 25
        if data.get('age', 30) > 45:
            risk += 20
        if data.get('stress_level', 5) > 7:
            risk += 20
            
        return min(risk, 100)

# Test the prediction
if __name__ == "__main__":
    # Sample health data
    test_data = {
        'age': 45,
        'gender': 'Male',
        'height': 175,
        'weight': 80,
        'systolic_bp': 135,
        'diastolic_bp': 85,
        'heart_rate': 75,
        'blood_glucose': 110,
        'cholesterol': 210,
        'smoking_status': 'never',
        'alcohol_consumption': 'moderate',
        'physical_activity': 'moderate',
        'sleep_hours': 7,
        'stress_level': 6
    }
    
    scorer = HealthRiskScorer()
    risks = scorer.calculate_risk_scores(test_data)
    
    print("\n🏥 Health Risk Assessment Results:")
    print(f"Diabetes Risk: {risks['diabetes_risk']}%")
    print(f"Heart Disease Risk: {risks['heart_disease_risk']}%")
    print(f"Stroke Risk: {risks['stroke_risk']}%")
    print(f"Hypertension Risk: {risks['hypertension_risk']}%")
    print(f"\nOverall Risk: {risks['overall_risk']}% ({risks['risk_category']})")