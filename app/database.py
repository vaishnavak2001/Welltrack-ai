# app/database.py
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
import hashlib
from .models import SessionLocal, Employee, Company, HealthRecord, RiskAssessment  # Note the dot (.) for relative import

class DatabaseOperations:
    @staticmethod
    def get_db():
        db = SessionLocal()
        try:
            return db
        finally:
            db.close()
    
    @staticmethod
    def create_user(db: Session, user_data: dict):
        """Create a new user/employee"""
        # Hash password
        password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
        
        # Create employee record
        employee = Employee(
            employee_id=user_data['employee_id'],
            first_name=user_data['first_name'],
            last_name=user_data['last_name'],
            email=user_data['email'],
            department=user_data['department'],
            password_hash=password_hash,
            company_id=1  # Default company for now
        )
        
        db.add(employee)
        db.commit()
        db.refresh(employee)
        return employee
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str):
        """Authenticate user login"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        user = db.query(Employee).filter(
            Employee.email == email,
            Employee.password_hash == password_hash
        ).first()
        
        if user:
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            return user
        return None
    
    @staticmethod
    def save_health_record(db: Session, employee_id: int, health_data: dict):
        """Save health assessment data"""
        health_record = HealthRecord(
            employee_id=employee_id,
            # Physical measurements
            height=health_data.get('height'),
            weight=health_data.get('weight'),
            bmi=health_data.get('bmi'),
            
            # Vital signs
            systolic_bp=health_data.get('systolic_bp'),
            diastolic_bp=health_data.get('diastolic_bp'),
            heart_rate=health_data.get('heart_rate'),
            respiratory_rate=health_data.get('respiratory_rate'),
            body_temperature=health_data.get('body_temp'),
            
            # Blood tests
            blood_glucose_fasting=health_data.get('glucose'),
            total_cholesterol=health_data.get('cholesterol'),
            hdl_cholesterol=health_data.get('hdl'),
            ldl_cholesterol=health_data.get('ldl'),
            triglycerides=health_data.get('triglycerides'),
            hemoglobin=health_data.get('hemoglobin'),
            hba1c=health_data.get('hba1c'),
            creatinine=health_data.get('creatinine'),
            sgpt=health_data.get('alt'),
            
            # Lifestyle
            smoking_status=health_data.get('smoking'),
            alcohol_consumption=health_data.get('alcohol'),
            physical_activity=health_data.get('exercise'),
            sleep_hours=health_data.get('sleep_hours'),
            stress_level=health_data.get('stress_level'),
            
            # Medical history
            family_diabetes=health_data.get('diabetes_family', False),
            family_heart_disease=health_data.get('heart_disease_family', False),
            family_hypertension=health_data.get('hypertension_family', False)
        )
        
        db.add(health_record)
        db.commit()
        db.refresh(health_record)
        return health_record
    
    @staticmethod
    def save_risk_assessment(db: Session, employee_id: int, health_record_id: int, risk_data: dict):
        """Save risk assessment results"""
        risk_assessment = RiskAssessment(
            employee_id=employee_id,
            health_record_id=health_record_id,
            diabetes_risk=risk_data.get('diabetes_risk_score', 0),
            heart_disease_risk=risk_data.get('heart_risk_score', 0),
            overall_risk_score=risk_data.get('overall_score', 0),
            risk_category=risk_data.get('risk_category', 'low'),
            recommendations=risk_data.get('recommendations', ''),
            model_version='1.0'
        )
        
        db.add(risk_assessment)
        db.commit()
        return risk_assessment
    
    @staticmethod
    def get_latest_health_record(db: Session, employee_id: int):
        """Get the most recent health record for an employee"""
        return db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()
    
    @staticmethod
    def get_health_history(db: Session, employee_id: int, limit: int = 10):
        """Get health record history for an employee"""
        return db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_risk_assessments(db: Session, employee_id: int):
        """Get all risk assessments for an employee"""
        return db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).all()