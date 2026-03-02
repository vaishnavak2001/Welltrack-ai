# app/export.py
import pandas as pd
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


class ReportGenerator:
    """Handles generation and export of health reports in CSV and PDF formats."""

    def __init__(self, db_session):
        self.db = db_session

    def export_health_data_csv(self, employee_id):
        """Export all health records for an employee as CSV."""
        from .models import HealthRecord, Employee

        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
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
                'Weight (kg)': round(record.weight, 1) if record.weight else None,
                'BMI': round(record.bmi, 2) if record.bmi else None,
                'Systolic BP': record.systolic_bp,
                'Diastolic BP': record.diastolic_bp,
                'Glucose (mg/dL)': round(record.blood_glucose_fasting, 1) if record.blood_glucose_fasting else None,
                'Total Cholesterol': round(record.total_cholesterol, 1) if record.total_cholesterol else None,
                'Heart Rate': record.heart_rate,
                'Sleep (hrs/day)': round(record.sleep_hours, 1) if record.sleep_hours else None,
                'Stress Level (1-10)': record.stress_level,
                'Smoking': record.smoking_status or 'N/A',
                'Alcohol': record.alcohol_consumption or 'N/A',
            })

        df = pd.DataFrame(data)
        return df.to_csv(index=False)

    def generate_health_report_pdf(self, employee_id):
        """Generate a comprehensive PDF health report for an employee."""
        from .models import Employee, HealthRecord, RiskAssessment

        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return None

        latest_record = self.db.query(HealthRecord).filter(
            HealthRecord.employee_id == employee_id
        ).order_by(HealthRecord.recorded_at.desc()).first()

        if not latest_record:
            return None

        latest_risk = self.db.query(RiskAssessment).filter(
            RiskAssessment.employee_id == employee_id
        ).order_by(RiskAssessment.assessed_at.desc()).first()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Title
        story.append(Paragraph("<b>Health Report</b>", styles['Title']))
        story.append(Spacer(1, 0.2 * inch))

        # Employee info
        info = Paragraph(
            f"<b>Name:</b> {employee.first_name} {employee.last_name}<br/>"
            f"<b>Email:</b> {employee.email}<br/>"
            f"<b>Department:</b> {employee.department or 'N/A'}<br/>"
            f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d')}",
            styles['Normal'],
        )
        story.append(info)
        story.append(Spacer(1, 0.3 * inch))

        # Health metrics table
        story.append(Paragraph("<b>Current Health Metrics</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))

        bp_val = f"{latest_record.systolic_bp}/{latest_record.diastolic_bp}" if latest_record.systolic_bp else "N/A"
        glucose_val = f"{latest_record.blood_glucose_fasting:.0f} mg/dL" if latest_record.blood_glucose_fasting else "N/A"
        chol_val = f"{latest_record.total_cholesterol:.0f} mg/dL" if latest_record.total_cholesterol else "N/A"

        health_data = [
            ['Metric', 'Value', 'Status', 'Normal Range'],
            ['BMI', f"{latest_record.bmi:.1f}" if latest_record.bmi else "N/A",
             self._get_bmi_status(latest_record.bmi) if latest_record.bmi else "N/A", '18.5 - 24.9'],
            ['Blood Pressure', bp_val,
             self._get_bp_status(latest_record.systolic_bp, latest_record.diastolic_bp) if latest_record.systolic_bp else "N/A",
             '<120/80'],
            ['Glucose', glucose_val,
             self._get_glucose_status(latest_record.blood_glucose_fasting) if latest_record.blood_glucose_fasting else "N/A",
             '70-99 mg/dL'],
            ['Total Cholesterol', chol_val,
             self._get_cholesterol_status(latest_record.total_cholesterol) if latest_record.total_cholesterol else "N/A",
             '<200 mg/dL'],
            ['Heart Rate', f"{latest_record.heart_rate} bpm" if latest_record.heart_rate else "N/A",
             'Normal', '60-100 bpm'],
        ]

        table = Table(health_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3 * inch))

        # Risk assessment section
        if latest_risk:
            story.append(Paragraph("<b>Risk Assessment</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))

            risk_data = [
                ['Risk Category', 'Score', 'Level'],
                ['Diabetes Risk', f"{latest_risk.diabetes_risk:.1%}" if latest_risk.diabetes_risk else "N/A",
                 self._get_risk_level(latest_risk.diabetes_risk) if latest_risk.diabetes_risk else "N/A"],
                ['Heart Disease Risk', f"{latest_risk.heart_disease_risk:.1%}" if latest_risk.heart_disease_risk else "N/A",
                 self._get_risk_level(latest_risk.heart_disease_risk) if latest_risk.heart_disease_risk else "N/A"],
                ['Overall Risk', f"{latest_risk.overall_risk_score:.1%}" if latest_risk.overall_risk_score else "N/A",
                 self._get_risk_level(latest_risk.overall_risk_score) if latest_risk.overall_risk_score else "N/A"],
            ]

            risk_table = Table(risk_data, colWidths=[2.5 * inch, 2 * inch, 2 * inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 0.3 * inch))

        # Lifestyle section
        story.append(Paragraph("<b>Lifestyle Factors</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))

        lifestyle_text = (
            f"&bull; Physical Activity: {latest_record.physical_activity or 'Not recorded'}<br/>"
            f"&bull; Sleep: {latest_record.sleep_hours:.1f} hours/night<br/>"
            f"&bull; Stress Level: {latest_record.stress_level}/10<br/>"
            f"&bull; Smoking: {latest_record.smoking_status or 'Not recorded'}<br/>"
            f"&bull; Alcohol: {latest_record.alcohol_consumption or 'Not recorded'}"
        )
        story.append(Paragraph(lifestyle_text, styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

        # Recommendations
        story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))
        recommendations = self._generate_recommendations(latest_record, latest_risk)
        for rec in recommendations:
            story.append(Paragraph(f"&bull; {rec}", styles['Normal']))

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
        return "Obese"

    def _get_bp_status(self, systolic, diastolic):
        if systolic < 120 and diastolic < 80:
            return "Normal"
        elif systolic < 130 and diastolic < 80:
            return "Elevated"
        elif systolic < 140 or diastolic < 90:
            return "Stage 1 HTN"
        return "Stage 2 HTN"

    def _get_glucose_status(self, glucose):
        if glucose < 100:
            return "Normal"
        elif glucose < 126:
            return "Prediabetic"
        return "Diabetic"

    def _get_cholesterol_status(self, cholesterol):
        if cholesterol < 200:
            return "Desirable"
        elif cholesterol < 240:
            return "Borderline High"
        return "High"

    def _get_risk_level(self, score):
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        return "High"

    def _generate_recommendations(self, health_record, risk_assessment):
        recommendations = []

        if health_record.bmi and health_record.bmi > 25:
            recommendations.append("Consider a weight management program to achieve a healthy BMI")
        elif health_record.bmi and health_record.bmi < 18.5:
            recommendations.append("Consult a nutritionist to achieve a healthy weight")

        if health_record.systolic_bp and (health_record.systolic_bp > 120 or health_record.diastolic_bp > 80):
            recommendations.append("Monitor blood pressure regularly and reduce sodium intake")

        if health_record.blood_glucose_fasting and health_record.blood_glucose_fasting > 100:
            recommendations.append("Regular glucose monitoring and consider dietary changes")

        if health_record.sleep_hours and health_record.sleep_hours < 7:
            recommendations.append("Aim for 7-9 hours of quality sleep per night")

        if health_record.stress_level and health_record.stress_level > 6:
            recommendations.append("Consider stress management techniques like meditation or yoga")

        if health_record.smoking_status and health_record.smoking_status.lower() == 'current':
            recommendations.append("Strongly consider smoking cessation programs")

        return recommendations if recommendations else ["Maintain your current healthy lifestyle!"]
