import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from sqlalchemy.orm import Session
from app.models import engine, Employee, HealthRecord, RiskAssessment, SessionLocal
from app.database import DatabaseOperations
import hashlib

# Page config
st.set_page_config(
    page_title="WellTrackAI - Corporate Health Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keep your existing CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1e3d59;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Sidebar navigation (keep your existing sidebar code)
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1e3d59/ffffff?text=WellTrackAI", width=150)
    st.markdown("---")
    
    if st.session_state.logged_in:
        st.write(f"👤 Welcome, {st.session_state.get('user_name', 'User')}")
        st.markdown("---")
    
    # Navigation menu
    st.subheader("Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
    
    if st.button("📋 Health Assessment", use_container_width=True):
        st.session_state.current_page = 'assessment'
    
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.current_page = 'dashboard'
    
    if st.button("📄 Reports", use_container_width=True):
        st.session_state.current_page = 'reports'
    
    if st.session_state.logged_in and st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.current_page = 'home'

# Main content area
if st.session_state.current_page == 'home':
    st.markdown('<h1 class="main-header">🏥 WellTrackAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">AI-Powered Corporate Health Management Platform</p>', unsafe_allow_html=True)
    
    # Feature cards (keep your existing cards)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Predictive Health Analytics</h3>
        <p>AI-powered risk assessment for diabetes, heart disease, and more</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Real-time Monitoring</h3>
        <p>Track employee health trends and organizational wellness metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🔒 Secure & Compliant</h3>
        <p>HIPAA-compliant platform with enterprise-grade security</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Login/Register section with database integration
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        # LOGIN TAB WITH DATABASE
        with tab1:
            st.subheader("Employee Login")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login", use_container_width=True):
                    if email and password:
                        db = SessionLocal()
                        try:
                            user = DatabaseOperations.authenticate_user(db, email, password)
                            if user:
                                st.session_state.logged_in = True
                                st.session_state.user_name = f"{user.first_name} {user.last_name}"
                                st.session_state.user_id = user.id
                                st.session_state.user_email = user.email
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error("Invalid email or password")
                        except Exception as e:
                            st.error(f"Login error: {str(e)}")
                        finally:
                            db.close()
                    else:
                        st.error("Please enter email and password")
        
        # REGISTER TAB WITH DATABASE
        with tab2:
            st.subheader("New Employee Registration")
            with st.form("register_form"):
                col1, col2 = st.columns(2)
                with col1:
                    first_name = st.text_input("First Name")
                    last_name = st.text_input("Last Name")
                    email = st.text_input("Email Address")
                
                with col2:
                    employee_id = st.text_input("Employee ID")
                    department = st.text_input("Department")
                    password = st.text_input("Create Password", type="password")
                
                if st.form_submit_button("Register", use_container_width=True):
                    if all([first_name, last_name, email, employee_id, password]):
                        db = SessionLocal()
                        try:
                            # Check if user already exists
                            existing = db.query(Employee).filter(
                                (Employee.email == email) | (Employee.employee_id == employee_id)
                            ).first()
                            
                            if existing:
                                st.error("User with this email or employee ID already exists")
                            else:
                                user_data = {
                                    'first_name': first_name,
                                    'last_name': last_name,
                                    'email': email,
                                    'employee_id': employee_id,
                                    'department': department,
                                    'password': password
                                }
                                DatabaseOperations.create_user(db, user_data)
                                st.success("Registration successful! Please login.")
                        except Exception as e:
                            st.error(f"Registration failed: {str(e)}")
                        finally:
                            db.close()
                    else:
                        st.error("Please fill all fields")

elif st.session_state.current_page == 'assessment':
    st.title("📋 Health Assessment")
    
    if not st.session_state.logged_in:
        st.warning("Please login to access health assessment")
    else:
        st.markdown("### Complete Your Quarterly Health Checkup")
        
        with st.form("health_assessment_form"):
            # Personal Information
            st.subheader("1️⃣ Personal Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            with col2:
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
                weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            
            with col3:
                bmi = round(weight / ((height/100) ** 2), 2)
                st.metric("BMI", bmi)
                if bmi < 18.5:
                    st.caption("Underweight")
                elif bmi < 25:
                    st.caption("Normal")
                elif bmi < 30:
                    st.caption("Overweight")
                else:
                    st.caption("Obese")
            
            st.markdown("---")
            
            # Vital Signs
            st.subheader("2️⃣ Vital Signs")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
                diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80)
            
            with col2:
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=72)
                respiratory_rate = st.number_input("Respiratory Rate", min_value=10, max_value=30, value=16)
            
            with col3:
                body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=40.0, value=36.5, step=0.1)
                blood_oxygen = st.number_input("Blood Oxygen (%)", min_value=80, max_value=100, value=98)
            
            st.markdown("---")
            
            # Blood Tests (keep your existing blood test fields)
            st.subheader("3️⃣ Blood Test Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=500, value=100)
                cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=0, max_value=500, value=200)
                hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=0, max_value=200, value=50)
            
            with col2:
                ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=0, max_value=300, value=100)
                triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0, max_value=1000, value=150)
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=30.0, value=14.0, step=0.1)
            
            with col3:
                hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
                creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                alt = st.number_input("ALT (U/L)", min_value=0, max_value=200, value=25)
            
            st.markdown("---")
            
            # Lifestyle Factors (keep your existing lifestyle fields)
            st.subheader("4️⃣ Lifestyle Factors")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
                alcohol = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Moderate", "Heavy"])
            
            with col2:
                exercise = st.selectbox("Exercise Frequency", ["None", "1-2 times/week", "3-4 times/week", "5+ times/week"])
                sleep_hours = st.number_input("Average Sleep Hours", min_value=0, max_value=12, value=7)
            
            with col3:
                stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
                diet_quality = st.selectbox("Diet Quality", ["Poor", "Fair", "Good", "Excellent"])
            
            st.markdown("---")
            
            # Medical History
            st.subheader("5️⃣ Medical History")
            col1, col2 = st.columns(2)
            
            with col1:
                diabetes_family = st.checkbox("Family History of Diabetes")
                heart_disease_family = st.checkbox("Family History of Heart Disease")
                hypertension_family = st.checkbox("Family History of Hypertension")
            
            with col2:
                current_medications = st.text_area("Current Medications (if any)", height=100)
                allergies = st.text_area("Known Allergies", height=100)
            
                        # Submit button with database integration
            submitted = st.form_submit_button("Submit Health Assessment", use_container_width=True, type="primary")
            
            if submitted:
                # Calculate risk scores
                diabetes_risk = "Low"
                if glucose > 126 or hba1c > 6.5:
                    diabetes_risk = "High"
                elif glucose > 100 or hba1c > 5.7:
                    diabetes_risk = "Medium"
                
                heart_risk = "Low"
                if cholesterol > 240 or systolic_bp > 140:
                    heart_risk = "High"
                elif cholesterol > 200 or systolic_bp > 130:
                    heart_risk = "Medium"
                
                # Prepare health data for database
                health_data = {
                    'height': height,
                    'weight': weight,
                    'bmi': bmi,
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'heart_rate': heart_rate,
                    'respiratory_rate': respiratory_rate,
                    'body_temp': body_temp,
                    'glucose': glucose,
                    'cholesterol': cholesterol,
                    'hdl': hdl,
                    'ldl': ldl,
                    'triglycerides': triglycerides,
                    'hemoglobin': hemoglobin,
                    'hba1c': hba1c,
                    'creatinine': creatinine,
                    'alt': alt,
                    'smoking': smoking.lower(),
                    'alcohol': alcohol.lower(),
                    'exercise': exercise,
                    'sleep_hours': sleep_hours,
                    'stress_level': stress_level,
                    'diabetes_family': diabetes_family,
                    'heart_disease_family': heart_disease_family,
                    'hypertension_family': hypertension_family
                }
                
                # Save to database
                db = SessionLocal()
                try:
                    risk_notification_data = {  
                        'diabetes': {
                            'probability': diabetes_risk,
                            'category': diabetes_risk
                        },    
                        'heart_disease': {
                            'probability': heart_risk,
                            'category': heart_risk
                        }
                    }
        
                    # Import and use notification service
                    from app.notifications import send_notification_after_assessment
        
                    # Send notification
                    notification_sent = send_notification_after_assessment(
                        st.session_state.user_id, 
                        risk_notification_data
                    )
        
                    if notification_sent:
                        st.info("📧 Health assessment results have been sent to your email.")
                except Exception as e:
                    # Don't break the app if notifications fail
                    st.warning("Could not send email notification. Please check your dashboard for results.")
                    # Calculate risk scores for database
                    diabetes_risk_score = 0.8 if diabetes_risk == "High" else 0.5 if diabetes_risk == "Medium" else 0.2
                    heart_risk_score = 0.8 if heart_risk == "High" else 0.5 if heart_risk == "Medium" else 0.2
                    overall_score = 85 if diabetes_risk == "Low" and heart_risk == "Low" else 70 if diabetes_risk == "Medium" or heart_risk == "Medium" else 50
                    
                    risk_data = {
                        'diabetes_risk_score': diabetes_risk_score,
                        'heart_risk_score': heart_risk_score,
                        'overall_score': overall_score / 100,  # Convert to 0-1 scale
                        'risk_category': 'high' if diabetes_risk == "High" or heart_risk == "High" else 'medium' if diabetes_risk == "Medium" or heart_risk == "Medium" else 'low',
                        'recommendations': 'Immediate consultation recommended' if diabetes_risk == "High" or heart_risk == "High" else 'Monitor closely' if diabetes_risk == "Medium" or heart_risk == "Medium" else 'Continue healthy lifestyle'
                    }
                    
                    # Save risk assessment
                    DatabaseOperations.save_risk_assessment(
                        db, st.session_state.user_id, HealthRecord.id, risk_data
                    )
                    
                    # Display results
                    st.success("✅ Health Assessment Submitted Successfully!")
                    
                    st.markdown("### 🎯 Risk Assessment Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Diabetes Risk", diabetes_risk)
                        if diabetes_risk == "High":
                            st.markdown('<p class="risk-high">Immediate consultation recommended</p>', unsafe_allow_html=True)
                        elif diabetes_risk == "Medium":
                            st.markdown('<p class="risk-medium">Monitor closely</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="risk-low">Keep up the good work!</p>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Heart Disease Risk", heart_risk)
                        if heart_risk == "High":
                            st.markdown('<p class="risk-high">Immediate consultation recommended</p>', unsafe_allow_html=True)
                        elif heart_risk == "Medium":
                            st.markdown('<p class="risk-medium">Monitor closely</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="risk-low">Keep up the good work!</p>', unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("Overall Health Score", f"{overall_score}/100")
                    
                except Exception as e:
                    st.error(f"Error saving assessment: {str(e)}")
                finally:
                    db.close()

elif st.session_state.current_page == 'dashboard':
    st.title("📊 Health Dashboard")
    
    if not st.session_state.logged_in:
        st.warning("Please login to view your dashboard")
    else:
        db = SessionLocal()
        try:
            # Get latest health record and risk assessments from database
            latest_record = DatabaseOperations.get_latest_health_record(db, st.session_state.user_id)
            risk_assessments = DatabaseOperations.get_risk_assessments(db, st.session_state.user_id)
            
            if latest_record and risk_assessments:
                # Real data from database
                latest_risk = risk_assessments[0]
                
                # Metrics Overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    score = int(latest_risk.overall_risk_score * 100)
                    st.metric("Overall Health Score", f"{score}/100")
                
                with col2:
                    days_ago = (datetime.now() - latest_record.recorded_at).days
                    st.metric("Last Assessment", f"{days_ago} days ago")
                
                with col3:
                    st.metric("Risk Level", latest_risk.risk_category.capitalize())
                
                with col4:
                    st.metric("Next Checkup", "In 88 days")
                
                st.markdown("---")
                
                # Get health history
                health_history = DatabaseOperations.get_health_history(db, st.session_state.user_id)
                
                if len(health_history) > 1:
                    # Create real charts from database data
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Blood pressure trend from real data
                        history_data = []
                        for record in health_history:
                            history_data.append({
                                'Date': record.recorded_at,
                                'Systolic': record.systolic_bp,
                                'Diastolic': record.diastolic_bp
                            })
                        
                        bp_df = pd.DataFrame(history_data)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=bp_df['Date'], y=bp_df['Systolic'], name='Systolic', line=dict(color='red')))
                        fig.add_trace(go.Scatter(x=bp_df['Date'], y=bp_df['Diastolic'], name='Diastolic', line=dict(color='blue')))
                        fig.update_layout(title='Blood Pressure Trend', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Glucose and cholesterol trend from real data
                        glucose_data = []
                        for record in health_history:
                            glucose_data.append({
                                'Date': record.recorded_at,
                                'Glucose': record.blood_glucose_fasting,
                                'Cholesterol': record.total_cholesterol
                            })
                        
                        glucose_df = pd.DataFrame(glucose_data)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=glucose_df['Date'], y=glucose_df['Glucose'], name='Glucose', line=dict(color='green')))
                        fig.add_trace(go.Scatter(x=glucose_df['Date'], y=glucose_df['Cholesterol'], name='Cholesterol', line=dict(color='orange')))
                        fig.update_layout(title='Glucose & Cholesterol Trend', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Recent assessments table with real data
                st.subheader("📋 Recent Health Assessments")
                assessment_data = []
                for i, record in enumerate(health_history[:5]):
                    if i < len(risk_assessments):
                        risk = risk_assessments[i]
                        assessment_data.append({
                            'Date': record.recorded_at.strftime('%Y-%m-%d'),
                            'Health Score': int(risk.overall_risk_score * 100),
                            'Blood Pressure': f"{record.systolic_bp}/{record.diastolic_bp}",
                            'Glucose': record.blood_glucose_fasting,
                            'Risk Level': risk.risk_category.capitalize()
                        })
                
                if assessment_data:
                    st.dataframe(pd.DataFrame(assessment_data), use_container_width=True)
            
            else:
                # No data yet - show placeholder
                st.info("No health assessments found. Please complete your first health assessment.")
                
                # Show empty charts with sample data
                col1, col2 = st.columns(2)
                
                with col1:
                    dates = pd.date_range(end=date.today(), periods=30, freq='D')
                    bp_data = pd.DataFrame({
                        'Date': dates,
                        'Systolic': [120] * 30,
                        'Diastolic': [80] * 30
                    })
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=bp_data['Date'], y=bp_data['Systolic'], name='Systolic', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=bp_data['Date'], y=bp_data['Diastolic'], name='Diastolic', line=dict(color='blue')))
                    fig.update_layout(title='Blood Pressure Trend (No Data)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    risk_data = pd.DataFrame({
                        'Risk Level': ['Low', 'Medium', 'High'],
                        'Count': [0, 0, 0]
                    })
                    
                    fig = px.pie(risk_data, values='Count', names='Risk Level', 
                                title='Risk Distribution (No Data)',
                                color_discrete_map={'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
        finally:
            db.close()

# Keep the rest of your code (reports section and footer) as is
elif st.session_state.current_page == 'reports':
    st.title("📄 Health Reports")
    
    if not st.session_state.logged_in:
        st.warning("Please login to view reports")
    else:
        st.subheader("Generate Health Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox("Report Type", [
                "Individual Health Summary",
                "Department Health Overview",
                "Risk Assessment Report",
                "Trend Analysis Report"
            ])
            
            date_range = st.date_input("Date Range", value=(date.today().replace(day=1), date.today()))
        
        with col2:
            format_type = st.selectbox("Export Format", ["PDF", "Excel", "CSV"])
            
            if st.button("Generate Report", use_container_width=True):
                st.success(f"✅ {report_type} generated successfully!")
                st.download_button(
                    label=f"Download {format_type} Report",
                    data=b"Sample report data",  # Replace with actual report generation
                    file_name=f"health_report_{date.today()}.{format_type.lower()}",
                    mime=f"application/{format_type.lower()}"
                )
        
        st.markdown("---")
        
        # Previous reports
        st.subheader("📂 Previous Reports")
        reports_data = pd.DataFrame({
            'Report Name': ['Q3 Health Summary', 'Risk Assessment Aug 2023', 'Department Overview July 2023'],
            'Date Generated': ['2023-10-01', '2023-08-15', '2023-07-30'],
            'Type': ['Individual', 'Risk Assessment', 'Department'],
            'Status': ['Ready', 'Ready', 'Ready']
        })
        st.dataframe(reports_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>WellTrackAI © 2024 | Powered by AI | HIPAA Compliant</p>
</div>
""", unsafe_allow_html=True)
