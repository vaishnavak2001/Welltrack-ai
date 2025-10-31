# app/visualizations.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from sqlalchemy import func, and_
import io
# Import your models
try:
    from .models import Employee, HealthRecord, RiskAssessment
    from .database import DatabaseOperations
except ImportError:
    from models import Employee, HealthRecord, RiskAssessment
    from database import DatabaseOperations

class AdvancedDashboard:
    def __init__(self, db_session):
        self.db = db_session
        
        # Define color schemes
        self.colors = {
            'primary': '#1e3d59',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db'
        }
        
        self.risk_colors = {
            'Low': self.colors['success'],
            'Medium': self.colors['warning'],
            'High': self.colors['danger']
        }
    
    def create_comprehensive_dashboard(self, employee_id):
        """Create a comprehensive health dashboard"""
        st.title("🏥 Advanced Health Analytics Dashboard")
        
        # Top metrics row
        self._create_metric_cards(employee_id)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📈 Trends", 
            "🏢 Department Analysis",
            "🎯 Risk Analysis",
            "💡 Insights"
        ])
        
        with tab1:
            self.create_overview_dashboard(employee_id)
        
        with tab2:
            self.create_trend_analysis(employee_id)
        
        with tab3:
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            if employee:
                self.create_department_statistics(employee.department)
        
        with tab4:
            self.create_risk_analysis_dashboard(employee_id)
        
        with tab5:
            self.create_insights_dashboard(employee_id)
    
    def _create_metric_cards(self, employee_id):
        """Create top-level metric cards"""
        # Get latest assessment
        latest_assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        if latest_assessment:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                health_score = int((1 - latest_assessment.overall_risk_score) * 100)
                delta = self._calculate_score_change(employee_id)
                st.metric(
                    "Health Score",
                    f"{health_score}/100",
                    f"{delta:+d}" if delta else None,
                    delta_color="normal" if delta >= 0 else "inverse"
                )
            
            with col2:
                risk_category = latest_assessment.risk_category.capitalize()
                color = self.risk_colors.get(risk_category, self.colors['info'])
                st.metric(
                    "Risk Level",
                    risk_category,
                    delta_color="normal"
                )
            
            with col3:
                days_since = (datetime.now() - latest_assessment.assessed_at).days
                st.metric(
                    "Last Assessment",
                    f"{days_since} days ago"
                )
            
            with col4:
                improvement = self._calculate_improvement_rate(employee_id)
                st.metric(
                    "Improvement Rate",
                    f"{improvement:.1f}%",
                    delta_color="normal"
                )
            
            with col5:
                next_due = 90 - days_since  # Assuming 90-day cycle
                st.metric(
                    "Next Assessment",
                    f"In {next_due} days"
                )
    
    def create_overview_dashboard(self, employee_id):
        """Create overview dashboard with key visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Health Score Gauge
            fig = self.create_health_score_gauge(employee_id)
            st.plotly_chart(fig, use_container_width=True)
            
            # Vital Signs Summary
            fig = self.create_vital_signs_chart(employee_id)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk Breakdown
            fig = self.create_risk_breakdown_chart(employee_id)
            st.plotly_chart(fig, use_container_width=True)
            
            # Lifestyle Factors
            fig = self.create_lifestyle_radar_chart(employee_id)
            st.plotly_chart(fig, use_container_width=True)
    
    def create_health_score_gauge(self, employee_id):
        """Create a gauge chart for overall health score"""
        latest_assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        if latest_assessment:
            health_score = int((1 - latest_assessment.overall_risk_score) * 100)
        else:
            health_score = 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Health Score", 'font': {'size': 24}},
            delta={'reference': 75, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self._get_score_color(health_score)},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffebee'},
                    {'range': [40, 70], 'color': '#fff3e0'},
                    {'range': [70, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    def create_vital_signs_chart(self, employee_id):
        """Create a chart showing latest vital signs"""
        latest_record = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()
        
        if not latest_record:
            return go.Figure()
        
        # Define normal ranges
        vital_signs = {
            'Blood Pressure': {
                'value': f"{latest_record.systolic_bp}/{latest_record.diastolic_bp}",
                'numeric': latest_record.systolic_bp,
                'normal_min': 90,
                'normal_max': 120,
                'unit': 'mmHg'
            },
            'Heart Rate': {
                'value': latest_record.heart_rate,
                'numeric': latest_record.heart_rate,
                'normal_min': 60,
                'normal_max': 100,
                'unit': 'bpm'
            },
            'Glucose': {
                'value': latest_record.blood_glucose_fasting,
                'numeric': latest_record.blood_glucose_fasting,
                'normal_min': 70,
                'normal_max': 100,
                'unit': 'mg/dL'
            },
            'BMI': {
                'value': round(latest_record.bmi, 1),
                'numeric': latest_record.bmi,
                'normal_min': 18.5,
                'normal_max': 25,
                'unit': 'kg/m²'
            }
        }
        
        fig = go.Figure()
        
        categories = list(vital_signs.keys())
        values = []
        colors = []
        
        for category, data in vital_signs.items():
            values.append(data['numeric'])
            # Determine color based on normal range
            if data['normal_min'] <= data['numeric'] <= data['normal_max']:
                colors.append(self.colors['success'])
            elif (data['numeric'] < data['normal_min'] * 0.9 or 
                  data['numeric'] > data['normal_max'] * 1.1):
                colors.append(self.colors['danger'])
            else:
                colors.append(self.colors['warning'])
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v['value']} {v['unit']}" for v in vital_signs.values()],
            textposition='outside'
        ))
        
        # Add normal range indicators
        for i, (category, data) in enumerate(vital_signs.items()):
            fig.add_shape(
                type="line",
                x0=i-0.4, x1=i+0.4,
                y0=data['normal_max'], y1=data['normal_max'],
                line=dict(color="green", width=2, dash="dash")
            )
        
        fig.update_layout(
            title="Current Vital Signs",
            showlegend=False,
            height=400,
            yaxis_title="Value"
        )
        
        return fig
    
    def create_risk_breakdown_chart(self, employee_id):
        """Create a breakdown of different risk factors"""
        latest_assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        if not latest_assessment:
            return go.Figure()
        
        risks = {
            'Diabetes': latest_assessment.diabetes_risk or 0,
            'Heart Disease': latest_assessment.heart_disease_risk or 0,
            'Hypertension': latest_assessment.hypertension_risk or 0,
            'Kidney Disease': latest_assessment.kidney_disease_risk or 0,
            'Liver Disease': latest_assessment.liver_disease_risk or 0
        }
        
        # Sort by risk level
        risks = dict(sorted(risks.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure(go.Bar(
            x=list(risks.values()),
            y=list(risks.keys()),
            orientation='h',
            marker=dict(
                color=[self._get_risk_color(v) for v in risks.values()],
                line=dict(color='white', width=2)
            ),
            text=[f"{v*100:.1f}%" for v in risks.values()],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Risk Factor Breakdown",
            xaxis_title="Risk Probability",
            xaxis=dict(tickformat='.0%', range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_lifestyle_radar_chart(self, employee_id):
        """Create a radar chart for lifestyle factors"""
        latest_record = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()
        
        if not latest_record:
            return go.Figure()
        
        # Convert lifestyle factors to scores (0-100)
        lifestyle_scores = {
            'Exercise': self._score_exercise(latest_record.physical_activity),
            'Sleep': self._score_sleep(latest_record.sleep_hours),
            'Stress': 100 - (latest_record.stress_level * 10) if latest_record.stress_level else 50,
            'Diet': self._score_diet(latest_record.diet_type),
            'Habits': self._score_habits(latest_record.smoking_status, latest_record.alcohol_consumption)
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(lifestyle_scores.values()),
            theta=list(lifestyle_scores.keys()),
            fill='toself',
            name='Current',
            line_color=self.colors['primary']
        ))
        
        # Add ideal scores for comparison
        fig.add_trace(go.Scatterpolar(
            r=[80] * len(lifestyle_scores),
            theta=list(lifestyle_scores.keys()),
            fill='toself',
            name='Target',
            line_color=self.colors['success'],
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Lifestyle Factors Assessment",
            height=400
        )
        
        return fig
    
    def create_trend_analysis(self, employee_id):
        """Create comprehensive trend analysis"""
        st.subheader("📈 Health Trends Analysis")
        
        # Time range selector
        col1, col2 = st.columns([3, 1])
        with col2:
            time_range = st.selectbox(
                "Time Range",
                ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All time"]
            )
        
        # Get historical data based on time range
        days_map = {
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 6 months": 180,
            "Last year": 365,
            "All time": None
        }
        days = days_map.get(time_range)
        
        # Create trend charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self.create_multi_metric_trend(employee_id, days)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self.create_risk_score_trend(employee_id, days)
            st.plotly_chart(fig, use_container_width=True)
        
        # Predictive analysis
        st.subheader("🔮 Predictive Analysis")
        fig = self.create_predictive_chart(employee_id)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_multi_metric_trend(self, employee_id, days=None):
        """Create multi-metric trend chart"""
        query = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        )
        
        if days:
            start_date = datetime.now() - timedelta(days=days)
            query = query.filter(HealthRecord.recorded_at >= start_date)
        
        records = query.order_by(HealthRecord.recorded_at).all()
        
        if not records:
            return go.Figure()
        
        # Prepare data
        dates = [r.recorded_at for r in records]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Blood Pressure", "Glucose Levels", "Cholesterol", "BMI"),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
                # Blood Pressure
        fig.add_trace(
            go.Scatter(x=dates, y=[r.systolic_bp for r in records], 
                      name="Systolic", line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=[r.diastolic_bp for r in records], 
                      name="Diastolic", line=dict(color='blue')),
            row=1, col=1, secondary_y=True
        )
        
        # Glucose
        fig.add_trace(
            go.Scatter(x=dates, y=[r.blood_glucose_fasting for r in records], 
                      name="Glucose", line=dict(color='green')),
            row=1, col=2
        )
        
        # Cholesterol
        fig.add_trace(
            go.Scatter(x=dates, y=[r.total_cholesterol for r in records], 
                      name="Total", line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=[r.hdl_cholesterol for r in records], 
                      name="HDL", line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=[r.ldl_cholesterol for r in records], 
                      name="LDL", line=dict(color='brown')),
            row=2, col=1
        )
        
        # BMI
        fig.add_trace(
            go.Scatter(x=dates, y=[r.bmi for r in records], 
                      name="BMI", line=dict(color='teal')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="mmHg", row=1, col=1)
        fig.update_yaxes(title_text="mg/dL", row=1, col=2)
        fig.update_yaxes(title_text="mg/dL", row=2, col=1)
        fig.update_yaxes(title_text="kg/m²", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="Health Metrics Trends")
        
        return fig
    
    def create_risk_score_trend(self, employee_id, days=None):
        """Create risk score trend chart"""
        query = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        )
        
        if days:
            start_date = datetime.now() - timedelta(days=days)
            query = query.filter(RiskAssessment.assessed_at >= start_date)
        
        assessments = query.order_by(RiskAssessment.assessed_at).all()
        
        if not assessments:
            return go.Figure()
        
        dates = [a.assessed_at for a in assessments]
        
        fig = go.Figure()
        
        # Add traces for each risk type
        fig.add_trace(go.Scatter(
            x=dates,
            y=[a.diabetes_risk * 100 for a in assessments],
            name="Diabetes Risk",
            line=dict(color=self.colors['danger'])
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=[a.heart_disease_risk * 100 for a in assessments],
            name="Heart Disease Risk",
            line=dict(color=self.colors['warning'])
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=[a.overall_risk_score * 100 for a in assessments],
            name="Overall Risk",
            line=dict(color=self.colors['primary'], width=3)
        ))
        
        fig.update_layout(
            title="Risk Score Trends",
            xaxis_title="Date",
            yaxis_title="Risk Score (%)",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_predictive_chart(self, employee_id):
        """Create predictive analysis chart"""
        # Get historical data
        assessments = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at).all()
        
        if len(assessments) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for predictions. Need at least 2 assessments.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Prepare historical data
        dates = [a.assessed_at for a in assessments]
        risk_scores = [a.overall_risk_score * 100 for a in assessments]
        
        # Simple linear prediction (in production, use proper ML models)
        from sklearn.linear_model import LinearRegression
        
        # Convert dates to numeric for regression
        date_numeric = [(d - dates[0]).days for d in dates]
        X = np.array(date_numeric).reshape(-1, 1)
        y = np.array(risk_scores)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 90 days
        future_days = 90
        last_day = date_numeric[-1]
        future_X = np.array([last_day + i for i in range(1, future_days + 1)]).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Create future dates
        future_dates = [dates[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=dates,
            y=risk_scores,
            mode='lines+markers',
            name='Historical',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(color=self.colors['info'], width=2, dash='dash')
        ))
        
        # Add confidence interval
        std_dev = np.std(risk_scores)
        upper_bound = predictions + std_dev
        lower_bound = predictions - std_dev
        
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill='toself',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Risk Score Prediction (Next 90 Days)",
            xaxis_title="Date",
            yaxis_title="Risk Score (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        return fig
    
    def create_department_statistics(self, department):
        """Create department-wide statistics dashboard"""
        st.subheader(f"🏢 {department} Department Health Overview")
        
        # Get department statistics
        dept_employees = self.db.query(Employee).filter(
            Employee.department == department
        ).all()
        
        if not dept_employees:
            st.warning("No employees found in this department")
            return
        
        employee_ids = [e.id for e in dept_employees]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        latest_assessments = []
        for emp_id in employee_ids:
            assessment = self.db.query(RiskAssessment).filter(
                RiskAssessment.employee_id == emp_id
            ).order_by(RiskAssessment.assessed_at.desc()).first()
            if assessment:
                latest_assessments.append(assessment)
        
        if latest_assessments:
            avg_health_score = np.mean([(1 - a.overall_risk_score) * 100 for a in latest_assessments])
            high_risk_count = sum(1 for a in latest_assessments if a.risk_category == 'high')
            completion_rate = (len(latest_assessments) / len(employee_ids)) * 100
        else:
            avg_health_score = 0
            high_risk_count = 0
            completion_rate = 0
        
        with col1:
            st.metric("Average Health Score", f"{avg_health_score:.1f}/100")
        
        with col2:
            st.metric("High Risk Employees", high_risk_count)
        
        with col3:
            st.metric("Assessment Completion", f"{completion_rate:.0f}%")
        
        with col4:
            st.metric("Total Employees", len(employee_ids))
        
        # Department charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self.create_department_risk_distribution(latest_assessments)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self.create_department_comparison_chart(department)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top health concerns
        st.subheader("🚨 Top Health Concerns")
        concerns = self.analyze_department_concerns(latest_assessments)
        for i, (concern, count) in enumerate(concerns[:5]):
            st.write(f"{i+1}. **{concern}**: {count} employees at risk")
    
    def create_department_risk_distribution(self, assessments):
        """Create risk distribution pie chart for department"""
        if not assessments:
            return go.Figure()
        
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        for a in assessments:
            risk_counts[a.risk_category] += 1
        
        fig = go.Figure(data=[go.Pie(
            labels=[k.capitalize() for k in risk_counts.keys()],
            values=list(risk_counts.values()),
            hole=.3,
            marker_colors=[self.risk_colors['Low'], self.risk_colors['Medium'], self.risk_colors['High']]
        )])
        
        fig.update_layout(
            title="Risk Distribution",
            height=400
        )
        
        return fig
    
    def create_department_comparison_chart(self, current_dept):
        """Compare current department with others"""
        # Get all departments
        all_depts = self.db.query(Employee.department).distinct().all()
        dept_names = [d[0] for d in all_depts if d[0]]
        
        dept_scores = []
        for dept in dept_names:
            dept_employees = self.db.query(Employee).filter(
                Employee.department == dept
            ).all()
            
            scores = []
            for emp in dept_employees:
                assessment = self.db.query(RiskAssessment).filter(
                    RiskAssessment.employee_id == emp.id
                ).order_by(RiskAssessment.assessed_at.desc()).first()
                if assessment:
                    scores.append((1 - assessment.overall_risk_score) * 100)
            
            if scores:
                dept_scores.append({
                    'department': dept,
                    'avg_score': np.mean(scores),
                    'is_current': dept == current_dept
                })
        
        if not dept_scores:
            return go.Figure()
        
        # Sort by score
        dept_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        
        colors = [self.colors['primary'] if d['is_current'] else self.colors['info'] 
                 for d in dept_scores]
        
        fig = go.Figure(go.Bar(
            x=[d['department'] for d in dept_scores],
            y=[d['avg_score'] for d in dept_scores],
            marker_color=colors
        ))
        
        fig.update_layout(
            title="Department Health Score Comparison",
            xaxis_title="Department",
            yaxis_title="Average Health Score",
            height=400
        )
        
        return fig
    
    def create_risk_analysis_dashboard(self, employee_id):
        """Create detailed risk analysis dashboard"""
        st.subheader("🎯 Detailed Risk Analysis")
        
        latest_assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        if not latest_assessment:
            st.warning("No risk assessment found")
            return
        
        # Risk factor analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self.create_risk_factor_contribution(employee_id)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self.create_risk_mitigation_chart(employee_id)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed recommendations
        st.subheader("📋 Personalized Recommendations")
        recommendations = self.generate_personalized_recommendations(employee_id)
        
        for category, items in recommendations.items():
            with st.expander(f"{category}", expanded=True):
                for item in items:
                    st.write(f"• {item}")
    
    def create_insights_dashboard(self, employee_id):
        """Create insights and suggestions dashboard"""
        st.subheader("💡 Health Insights & Action Plan")
        
        # Get comprehensive health data
        insights = self.generate_health_insights(employee_id)
        
        # Display insights in cards
        for insight in insights:
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.write(insight['icon'])
                with col2:
                    st.write(f"**{insight['title']}**")
                    st.write(insight['description'])
                    if insight.get('action'):
                        st.info(f"💡 {insight['action']}")
                st.write("---")
    
    # Helper methods
    def _calculate_score_change(self, employee_id):
        """Calculate health score change from last assessment"""
        assessments = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).limit(2).all()
        
        if len(assessments) < 2:
            return 0
        
        current_score = (1 - assessments[0].overall_risk_score) * 100
        previous_score = (1 - assessments[1].overall_risk_score) * 100
        
        return int(current_score - previous_score)
    
    def _calculate_improvement_rate(self, employee_id):
        """Calculate improvement rate over time"""
        assessments = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at).all()
        
        if len(assessments) < 2:
            return 0.0
        
        # Calculate trend
        scores = [(1 - a.overall_risk_score) * 100 for a in assessments]
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        return np.mean(improvements) if improvements else 0.0
    
    def _get_score_color(self, score):
        """Get color based on health score"""
        if score >= 80:
            return self.colors['success']
        elif score >= 60:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def _get_risk_color(self, risk_value):
        """Get color based on risk value"""
        if risk_value < 0.3:
            return self.colors['success']
        elif risk_value < 0.6:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def _score_exercise(self, activity_level):
        """Convert exercise level to score (0-100)"""
        exercise_scores = {
            'sedentary': 20,
            'light': 40,
            'moderate': 70,
            'active': 90,
            None: 50
        }
        return exercise_scores.get(activity_level, 50)
    
    def _score_sleep(self, sleep_hours):
        """Convert sleep hours to score (0-100)"""
        if sleep_hours is None:
            return 50
        
        # Optimal sleep is 7-9 hours
        if 7 <= sleep_hours <= 9:
            return 90
        elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
            return 70
        elif 5 <= sleep_hours < 6 or 10 < sleep_hours <= 11:
            return 50
        else:
            return 30
    
    def _score_diet(self, diet_type):
        """Convert diet quality to score (0-100)"""
        diet_scores = {
            'poor': 20,
            'fair': 40,
            'good': 70,
            'excellent': 90,
            None: 50
        }
        return diet_scores.get(diet_type, 50)
    
    def _score_habits(self, smoking_status, alcohol_consumption):
        """Calculate habits score based on smoking and alcohol (0-100)"""
        smoking_scores = {
            'never': 40,
            'former': 30,
            'current': 0,
            None: 20
        }
        
        alcohol_scores = {
            'none': 30,
            'occasional': 25,
            'moderate': 20,
            'heavy': 0,
            None: 15
        }
        
        smoking_score = smoking_scores.get(smoking_status, 20)
        alcohol_score = alcohol_scores.get(alcohol_consumption, 15)
        
        # Combined score with additional points for good habits
        base_score = smoking_score + alcohol_score
        if smoking_status == 'never' and alcohol_consumption in ['none', 'occasional']:
            base_score += 30  # Bonus for excellent habits
        
        return min(base_score, 100)
    
    def analyze_department_concerns(self, assessments):
        """Analyze top health concerns in department"""
        if not assessments:
            return []
        
        concerns = {
            'Diabetes': 0,
            'Heart Disease': 0,
            'Hypertension': 0,
            'Obesity': 0,
            'High Stress': 0
        }
        
        for assessment in assessments:
            if assessment.diabetes_risk > 0.5:
                concerns['Diabetes'] += 1
            if assessment.heart_disease_risk > 0.5:
                concerns['Heart Disease'] += 1
            if assessment.hypertension_risk > 0.5:
                concerns['Hypertension'] += 1
        
        # Sort by count
        sorted_concerns = sorted(concerns.items(), key=lambda x: x[1], reverse=True)
        return sorted_concerns
    
    def create_risk_factor_contribution(self, employee_id):
        """Create chart showing contribution of different factors to overall risk"""
        latest_record = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()
        
        if not latest_record:
            return go.Figure()
        
        # Calculate risk contributions (simplified)
        factors = {
            'BMI': self._calculate_bmi_risk(latest_record.bmi),
            'Blood Pressure': self._calculate_bp_risk(latest_record.systolic_bp, latest_record.diastolic_bp),
            'Glucose': self._calculate_glucose_risk(latest_record.blood_glucose_fasting),
            'Cholesterol': self._calculate_cholesterol_risk(latest_record.total_cholesterol),
            'Lifestyle': self._calculate_lifestyle_risk(latest_record),
            'Family History': self._calculate_family_risk(latest_record)
        }
        
        # Sort by contribution
        factors = dict(sorted(factors.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure(go.Bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            marker=dict(
                color=[self._get_risk_color(v/100) for v in factors.values()],
                line=dict(color='white', width=1)
            ),
            text=[f"{v}%" for v in factors.values()],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Risk Factor Contributions",
            xaxis_title="Contribution to Overall Risk (%)",
            xaxis=dict(range=[0, 100]),
            height=400
        )
        
        return fig
    
    def create_risk_mitigation_chart(self, employee_id):
        """Create chart showing potential risk reduction through interventions"""
        current_risk = self._get_current_risk_score(employee_id)
        
        # Potential risk reductions
        interventions = {
            'Weight Loss (10%)': current_risk * 0.15,
            'Exercise Program': current_risk * 0.20,
            'Dietary Changes': current_risk * 0.18,
            'Stress Management': current_risk * 0.12,
            'Medication Compliance': current_risk * 0.10,
            'Sleep Improvement': current_risk * 0.08
        }
        
        # Calculate new risk levels
        new_risks = {k: max(0, current_risk - v) for k, v in interventions.items()}
        
        fig = go.Figure()
        
        # Current risk bar
        fig.add_trace(go.Bar(
            x=['Current Risk'],
            y=[current_risk],
            name='Current',
            marker_color=self._get_risk_color(current_risk/100)
        ))
        
        # Potential risks after interventions
        for intervention, new_risk in new_risks.items():
            fig.add_trace(go.Bar(
                x=[intervention],
                y=[new_risk],
                name=intervention,
                marker_color=self._get_risk_color(new_risk/100)
            ))
        
        fig.update_layout(
            title="Potential Risk Reduction Through Interventions",
            yaxis_title="Risk Score (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=400
        )
        
        return fig
    
    def generate_personalized_recommendations(self, employee_id):
        """Generate personalized health recommendations"""
        latest_record = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()
        
        latest_assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        recommendations = {
            "🏃 Exercise & Activity": [],
            "🥗 Nutrition": [],
            "😴 Sleep & Recovery": [],
            "🧘 Stress Management": [],
            "💊 Medical Follow-up": []
        }
        
        if latest_record:
            # Exercise recommendations
            if latest_record.physical_activity in ['sedentary', 'light']:
                recommendations["🏃 Exercise & Activity"].extend([
                    "Start with 30 minutes of brisk walking daily",
                    "Gradually increase to 150 minutes of moderate exercise per week",
                    "Include strength training 2 days per week",
                    "Consider joining group fitness classes for motivation"
                ])
            
            # Nutrition recommendations
            if latest_record.bmi > 25:
                recommendations["🥗 Nutrition"].extend([
                    "Reduce daily calorie intake by 500-750 calories",
                    "Increase vegetable and fruit servings to 5-7 per day",
                    "Limit processed foods and sugary drinks",
                    "Consider consulting with a nutritionist"
                ])
            
            # Sleep recommendations
            if latest_record.sleep_hours < 7 or latest_record.sleep_hours > 9:
                recommendations["😴 Sleep & Recovery"].extend([
                    "Aim for 7-9 hours of sleep per night",
                    "Establish a consistent sleep schedule",
                    "Create a relaxing bedtime routine",
                    "Limit screen time 1 hour before bed"
                ])
            
            # Stress recommendations
            if latest_record.stress_level > 6:
                recommendations["🧘 Stress Management"].extend([
                    "Practice daily meditation or deep breathing exercises",
                    "Consider yoga or tai chi classes",
                    "Schedule regular breaks during work",
                    "Explore stress management counseling options"
                ])
        
        if latest_assessment:
            # Medical recommendations based on risk
            if latest_assessment.diabetes_risk > 0.5:
                recommendations["💊 Medical Follow-up"].extend([
                    "Schedule HbA1c test within 2 weeks",
                    "Monitor blood glucose levels daily",
                    "Consult with endocrinologist",
                    "Review medication options with doctor"
                ])
            
            if latest_assessment.heart_disease_risk > 0.5:
                recommendations["💊 Medical Follow-up"].extend([
                    "Schedule cardiac evaluation",
                    "Monitor blood pressure daily",
                    "Discuss statin therapy with doctor",
                    "Consider cardiac rehabilitation program"
                ])
        
        # Remove empty categories
        recommendations = {k: v for k, v in recommendations.items() if v}
        
        return recommendations
    
    def generate_health_insights(self, employee_id):
        """Generate actionable health insights"""
        insights = []
        
        # Get latest data
        latest_record = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()
        
        latest_assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        # Get historical trends
        improvement_rate = self._calculate_improvement_rate(employee_id)
        
        if latest_record:
            # BMI insight
            if latest_record.bmi > 30:
                insights.append({
                    'icon': '⚠️',
                    'title': 'Weight Management Alert',
                    'description': f'Your BMI is {latest_record.bmi:.1f}, which is in the obese range. This significantly increases your health risks.',
                    'action': 'A 10% weight reduction could lower your disease risk by up to 30%'
                })
            
            # Blood pressure insight
            if latest_record.systolic_bp > 130 or latest_record.diastolic_bp > 80:
                insights.append({
                    'icon': '🩺',
                    'title': 'Blood Pressure Concern',
                    'description': f'Your blood pressure ({latest_record.systolic_bp}/{latest_record.diastolic_bp}) is elevated.',
                    'action': 'Regular monitoring and lifestyle changes can prevent hypertension'
                })
            
            # Positive insights
            if latest_record.physical_activity in ['moderate', 'active']:
                insights.append({
                    'icon': '🌟',
                    'title': 'Great Exercise Habits!',
                    'description': 'Your regular physical activity is protecting your health.',
                    'action': 'Keep it up! Consider varying your routine for continued benefits'
                })
        
        if latest_assessment:
            # Risk trend insight
            if improvement_rate > 0:
                insights.append({
                    'icon': '📈',
                    'title': 'Health Improving!',
                    'description': f'Your health score has improved by {improvement_rate:.1f}% on average.',
                    'action': 'Continue your current health practices'
                })
            elif improvement_rate < -5:
                insights.append({
                    'icon': '📉',
                    'title': 'Health Declining',
                    'description': 'Your health metrics show a concerning downward trend.',
                    'action': 'Schedule a comprehensive health review with your doctor'
                })
            
                        # High risk alert
            if latest_assessment.overall_risk_score > 0.7:
                insights.append({
                    'icon': '🚨',
                    'title': 'High Health Risk Alert',
                    'description': 'Your overall health risk is critically high.',
                    'action': 'Immediate medical consultation required. Contact your healthcare provider today.'
                })
        
        # Add general insights if no specific concerns
        if not insights:
            insights.append({
                'icon': '✅',
                'title': 'Health Status Good',
                'description': 'Your health metrics are within normal ranges.',
                'action': 'Maintain your healthy lifestyle and continue regular check-ups'
            })
        
        return insights
    
    def _calculate_bmi_risk(self, bmi):
        """Calculate risk contribution from BMI"""
        if bmi is None:
            return 0
        
        if bmi < 18.5 or bmi > 40:
            return 30
        elif bmi > 35:
            return 25
        elif bmi > 30:
            return 20
        elif bmi > 25:
            return 15
        else:
            return 5
    
    def _calculate_bp_risk(self, systolic, diastolic):
        """Calculate risk contribution from blood pressure"""
        if systolic is None or diastolic is None:
            return 0
        
        if systolic > 180 or diastolic > 120:
            return 35
        elif systolic > 140 or diastolic > 90:
            return 25
        elif systolic > 130 or diastolic > 80:
            return 15
        else:
            return 5
    
    def _calculate_glucose_risk(self, glucose):
        """Calculate risk contribution from glucose levels"""
        if glucose is None:
            return 0
        
        if glucose > 200:
            return 35
        elif glucose > 126:
            return 25
        elif glucose > 100:
            return 15
        else:
            return 5
    
    def _calculate_cholesterol_risk(self, cholesterol):
        """Calculate risk contribution from cholesterol"""
        if cholesterol is None:
            return 0
        
        if cholesterol > 300:
            return 30
        elif cholesterol > 240:
            return 20
        elif cholesterol > 200:
            return 10
        else:
            return 5
    
    def _calculate_lifestyle_risk(self, record):
        """Calculate risk contribution from lifestyle factors"""
        risk = 0
        
        if record.smoking_status == 'current':
            risk += 15
        elif record.smoking_status == 'former':
            risk += 5
        
        if record.alcohol_consumption == 'heavy':
            risk += 10
        elif record.alcohol_consumption == 'moderate':
            risk += 5
        
        if record.physical_activity == 'sedentary':
            risk += 10
        elif record.physical_activity == 'light':
            risk += 5
        
        if record.stress_level and record.stress_level > 7:
            risk += 10
        elif record.stress_level and record.stress_level > 5:
            risk += 5
        
        return min(risk, 40)
    
    def _calculate_family_risk(self, record):
        """Calculate risk contribution from family history"""
        risk = 0
        
        if record.family_diabetes:
            risk += 8
        if record.family_heart_disease:
            risk += 8
        if record.family_hypertension:
            risk += 6
        if record.family_cancer:
            risk += 5
        
        return min(risk, 25)
    
    def _get_current_risk_score(self, employee_id):
        """Get current overall risk score as percentage"""
        latest_assessment = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        if latest_assessment:
            return latest_assessment.overall_risk_score * 100
        return 50  # Default if no assessment


# Utility function to create advanced dashboard in your app
def create_advanced_health_dashboard(db_session, employee_id):
    """Create advanced health dashboard for an employee"""
    dashboard = AdvancedDashboard(db_session)
    dashboard.create_comprehensive_dashboard(employee_id)


# Function to export dashboard data
def export_dashboard_data(db_session, employee_id, format='excel'):
    """Export dashboard data to various formats"""
    dashboard = AdvancedDashboard(db_session)
    
    # Get all relevant data
    health_records = db_session.query(HealthRecord).filter(
        HealthRecord.employee_id == employee_id
    ).order_by(HealthRecord.recorded_at.desc()).all()
    
    risk_assessments = db_session.query(RiskAssessment).filter(
        RiskAssessment.employee_id == employee_id
    ).order_by(RiskAssessment.assessed_at.desc()).all()
    
    # Convert to DataFrames
    health_df = pd.DataFrame([{
        'Date': r.recorded_at,
        'BMI': r.bmi,
        'Systolic BP': r.systolic_bp,
        'Diastolic BP': r.diastolic_bp,
        'Glucose': r.blood_glucose_fasting,
        'Cholesterol': r.total_cholesterol,
        'Heart Rate': r.heart_rate
    } for r in health_records])
    
    risk_df = pd.DataFrame([{
        'Date': a.assessed_at,
        'Overall Risk': a.overall_risk_score,
        'Diabetes Risk': a.diabetes_risk,
        'Heart Disease Risk': a.heart_disease_risk,
        'Risk Category': a.risk_category
    } for a in risk_assessments])
    
    if format == 'excel':
        # Create Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            health_df.to_excel(writer, sheet_name='Health Records', index=False)
            risk_df.to_excel(writer, sheet_name='Risk Assessments', index=False)
        
        output.seek(0)
        return output
    
    elif format == 'csv':
        # Return health records as CSV
        return health_df.to_csv(index=False)
    
    else:
        # Return as JSON
        return {
            'health_records': health_df.to_dict('records'),
            'risk_assessments': risk_df.to_dict('records')
        }


# Function to generate health report PDF
def generate_health_report_pdf(db_session, employee_id):
    """Generate comprehensive health report in PDF format"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import io
    
    # This would require reportlab installation
    # Implementation would generate a professional PDF report
    pass


# Quick statistics function
def get_employee_health_stats(db_session, employee_id):
    """Get quick health statistics for an employee"""
    latest_record = db_session.query(HealthRecord).filter(
        HealthRecord.employee_id == employee_id
    ).order_by(HealthRecord.recorded_at.desc()).first()
    
    latest_assessment = db_session.query(RiskAssessment).filter(
        RiskAssessment.employee_id == employee_id
    ).order_by(RiskAssessment.assessed_at.desc()).first()
    
    if not latest_record or not latest_assessment:
        return None
    
    return {
        'health_score': int((1 - latest_assessment.overall_risk_score) * 100),
        'risk_category': latest_assessment.risk_category,
        'bmi': round(latest_record.bmi, 1),
        'blood_pressure': f"{latest_record.systolic_bp}/{latest_record.diastolic_bp}",
        'glucose': latest_record.blood_glucose_fasting,
        'last_assessment': latest_assessment.assessed_at.strftime('%Y-%m-%d'),
        'top_risks': [
            {'name': 'Diabetes', 'value': latest_assessment.diabetes_risk},
            {'name': 'Heart Disease', 'value': latest_assessment.heart_disease_risk},
            {'name': 'Hypertension', 'value': latest_assessment.hypertension_risk}
        ]
    }


