# app/export.py
import pandas as pd
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

class ReportGenerator:
    def __init__(self, db_session):
        self.db = db_session
        
    def export_health_data_csv(self, employee_id):
        """Export health records to CSV"""
        from .models import HealthRecord, Employee
        
        # Get employee info
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        
        # Get all health records
        records = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).all()
        
        if not records:
            return None
            
        data = []
        for record in records:
            data.append({
                'Date': record.recorded_at.strftime('%Y-%m-%d %H:%M'),
                'Height (cm)': record.height,
                'Weight (kg)': round(record.weight, 1),
                'BMI': round(record.bmi, 2),
                'Systolic BP': record.systolic_bp,
                'Diastolic BP': record.diastolic_bp,
                'Glucose (mg/dL)': round(record.blood_glucose_fasting, 1),
                'Total Cholesterol': round(record.cholesterol_total, 1),
                'Heart Rate': record.heart_rate,
                'Exercise (hrs/week)': round(record.exercise_hours, 1),
                'Sleep (hrs/day)': round(record.sleep_hours, 1),
                'Stress Level (1-10)': record.stress_level,
                'Smoking': 'Yes' if record.smoking else 'No',
                'Alcohol': record.alcohol_consumption
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def generate_health_report_pdf(self, employee_id):
        """Generate comprehensive PDF health report"""
        from .models import Employee, HealthRecord, RiskAssessment
        
        # Get employee data
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return None
            
        # Get latest health record
        latest_record = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()
        
        if not latest_record:
            return None
            
        # Get latest risk assessment
        latest_risk = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(f"<b>Health Report</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Employee Info
        info = Paragraph(f"<b>Name:</b> {employee.first_name} {employee.last_name}<br/>"
                        f"<b>Email:</b> {employee.email}<br/>"
                        f"<b>Department:</b> {employee.department or 'N/A'}<br/>"
                        f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d')}", 
                        styles['Normal'])
        story.append(info)
        story.append(Spacer(1, 0.3*inch))
        
        # Health Metrics Section
        story.append(Paragraph("<b>Current Health Metrics</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        # Health data table
        health_data = [
            ['Metric', 'Value', 'Status', 'Normal Range'],
            ['BMI', f"{latest_record.bmi:.1f}", self._get_bmi_status(latest_record.bmi), '18.5 - 24.9'],
            ['Blood Pressure', f"{latest_record.systolic_bp}/{latest_record.diastolic_bp}", 
             self._get_bp_status(latest_record.systolic_bp, latest_record.diastolic_bp), '<120/80'],
            ['Glucose', f"{latest_record.blood_glucose_fasting:.0f} mg/dL", 
             self._get_glucose_status(latest_record.blood_glucose_fasting), '70-99 mg/dL'],
            ['Total Cholesterol', f"{latest_record.cholesterol_total:.0f} mg/dL", 
             self._get_cholesterol_status(latest_record.cholesterol_total), '<200 mg/dL'],
            ['Heart Rate', f"{latest_record.heart_rate} bpm", 'Normal', '60-100 bpm']
        ]
        
        table = Table(health_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white])
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Assessment Section
        if latest_risk:
            story.append(Paragraph("<b>Risk Assessment</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            risk_data = [
                ['Risk Category', 'Score', 'Level'],
                ['Diabetes Risk', f"{latest_risk.diabetes_risk_score:.1%}", 
                 self._get_risk_level(latest_risk.diabetes_risk_score)],
                ['Heart Disease Risk', f"{latest_risk.heart_risk_score:.1%}", 
                 self._get_risk_level(latest_risk.heart_risk_score)],
                ['Overall Risk', f"{latest_risk.overall_score:.1%}", 
                 self._get_risk_level(latest_risk.overall_score)]
            ]
            
            risk_table = Table(risk_data, colWidths=[2.5*inch, 2*inch, 2*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white])
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Lifestyle Factors
        story.append(Paragraph("<b>Lifestyle Factors</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        lifestyle_text = f"""
        • Exercise: {latest_record.exercise_hours:.1f} hours/week<br/>
        • Sleep: {latest_record.sleep_hours:.1f} hours/night<br/>
        • Stress Level: {latest_record.stress_level}/10<br/>
        • Smoking: {'Yes' if latest_record.smoking else 'No'}<br/>
        • Alcohol: {latest_record.alcohol_consumption}
        """
        story.append(Paragraph(lifestyle_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        recommendations = self._generate_recommendations(latest_record, latest_risk)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _get_bmi_status(self, bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def _get_bp_status(self, systolic, diastolic):
        if systolic < 120 and diastolic < 80:
            return "Normal"
        elif systolic < 130 and diastolic < 80:
            return "Elevated"
        elif systolic < 140 or diastolic < 90:
            return "Stage 1 HTN"
        else:
            return "Stage 2 HTN"
    
    def _get_glucose_status(self, glucose):
        if glucose < 100:
            return "Normal"
        elif glucose < 126:
            return "Prediabetic"
        else:
            return "Diabetic"
    
    def _get_cholesterol_status(self, cholesterol):
        if cholesterol < 200:
            return "Desirable"
        elif cholesterol < 240:
            return "Borderline High"
        else:
            return "High"
    
    def _get_risk_level(self, score):
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        else:
            return "High"
    
    def _generate_recommendations(self, health_record, risk_assessment):
        recommendations = []
        
        # BMI recommendations
        if health_record.bmi > 25:
            recommendations.append("Consider a weight management program to achieve a healthy BMI")
        elif health_record.bmi < 18.5:
            recommendations.append("Consult a nutritionist to achieve a healthy weight")
        
        # BP recommendations
        if health_record.systolic_bp > 120 or health_record.diastolic_bp > 80:
            recommendations.append("Monitor blood pressure regularly and reduce sodium intake")
        
        # Glucose recommendations
        if health_record.blood_glucose_fasting > 100:
            recommendations.append("Regular glucose monitoring and consider dietary changes")
        
        # Exercise recommendations
        if health_record.exercise_hours < 3:
            recommendations.append("Increase physical activity to at least 150 minutes per week")
        
        # Sleep recommendations
        if health_record.sleep_hours < 7:
            recommendations.append("Aim for 7-9 hours of quality sleep per night")
        
        # Stress recommendations
        if health_record.stress_level > 6:
            recommendations.append("Consider stress management techniques like meditation or yoga")
        
        # Smoking
        if health_record.smoking:
            recommendations.append("Strongly consider smoking cessation programs")
        
        return recommendations if recommendations else ["Maintain your current healthy lifestyle!"] 
