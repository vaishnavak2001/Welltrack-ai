from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class Company(Base):
    __tablename__ = 'companies'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    industry = Column(String(50))
    size = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    employees = relationship("Employee", back_populates="company")


class Employee(Base):
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), unique=True, nullable=False)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=True)

    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20))
    date_of_birth = Column(DateTime)
    gender = Column(String(10))
    department = Column(String(50))

    password_hash = Column(String(200))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    company = relationship("Company", back_populates="employees")
    health_records = relationship("HealthRecord", back_populates="employee")
    risk_assessments = relationship("RiskAssessment", back_populates="employee")


class HealthRecord(Base):
    __tablename__ = 'health_records'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))

    # Physical measurements
    height = Column(Float)
    weight = Column(Float)
    bmi = Column(Float)
    waist_circumference = Column(Float)

    # Vital signs
    systolic_bp = Column(Integer)
    diastolic_bp = Column(Integer)
    heart_rate = Column(Integer)
    respiratory_rate = Column(Integer)
    body_temperature = Column(Float)

    # Blood tests
    blood_glucose_fasting = Column(Float)
    blood_glucose_random = Column(Float)
    hba1c = Column(Float)
    total_cholesterol = Column(Float)
    hdl_cholesterol = Column(Float)
    ldl_cholesterol = Column(Float)
    triglycerides = Column(Float)

    # Liver function
    sgot = Column(Float)
    sgpt = Column(Float)
    alkaline_phosphatase = Column(Float)

    # Kidney function
    creatinine = Column(Float)
    urea = Column(Float)
    uric_acid = Column(Float)

    # Complete blood count
    hemoglobin = Column(Float)
    rbc_count = Column(Float)
    wbc_count = Column(Float)
    platelet_count = Column(Float)

    # Lifestyle
    smoking_status = Column(String(20))
    alcohol_consumption = Column(String(20))
    physical_activity = Column(String(20))
    sleep_hours = Column(Float)
    stress_level = Column(Integer)

    # Diet
    diet_type = Column(String(20))
    daily_water_intake = Column(Float)
    fruits_per_day = Column(Integer)
    vegetables_per_day = Column(Integer)

    # Medical history
    diabetes = Column(Boolean, default=False)
    hypertension = Column(Boolean, default=False)
    heart_disease = Column(Boolean, default=False)
    stroke_history = Column(Boolean, default=False)
    cancer_history = Column(Boolean, default=False)

    # Family history
    family_diabetes = Column(Boolean, default=False)
    family_hypertension = Column(Boolean, default=False)
    family_heart_disease = Column(Boolean, default=False)
    family_cancer = Column(Boolean, default=False)

    recorded_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    employee = relationship("Employee", back_populates="health_records")


class RiskAssessment(Base):
    __tablename__ = 'risk_assessments'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    health_record_id = Column(Integer, ForeignKey('health_records.id'))

    # Risk scores (0-1 scale)
    diabetes_risk = Column(Float)
    heart_disease_risk = Column(Float)
    stroke_risk = Column(Float)
    hypertension_risk = Column(Float)
    cancer_risk = Column(Float)
    kidney_disease_risk = Column(Float)
    liver_disease_risk = Column(Float)

    overall_risk_score = Column(Float)
    risk_category = Column(String(20))

    model_version = Column(String(20))
    confidence_score = Column(Float)

    recommendations = Column(Text)
    priority_actions = Column(Text)

    assessed_at = Column(DateTime, default=datetime.utcnow)
    next_assessment_due = Column(DateTime)

    employee = relationship("Employee", back_populates="risk_assessments")


# Database setup - defaults to SQLite for easy local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///welltrack.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables on import
Base.metadata.create_all(bind=engine)


def get_db():
    """Yields a database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()