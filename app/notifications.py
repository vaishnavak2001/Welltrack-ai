# app/notifications.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
import logging
import json

# Fix imports based on your project structure
try:
    from .models import SessionLocal, Employee, RiskAssessment
    from .database import DatabaseOperations
except ImportError:
    # If running directly
    from models import SessionLocal, Employee, RiskAssessment
    from database import DatabaseOperations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@welltrack.ai")
        
        # HR notification settings
        self.hr_email = os.getenv("HR_EMAIL", "hr@company.com")
        
        # Check if email is configured
        self.email_configured = bool(self.smtp_username and self.smtp_password)
        if not self.email_configured:
            logger.warning("Email not configured. Notifications will be logged only.")
    
    def send_health_alert(self, user_id: int, risk_data: dict):
        """Send health alerts based on risk assessment"""
        db = SessionLocal()
        try:
            # Get user information
            user = db.query(Employee).filter(Employee.id == user_id).first()
            if not user:
                logger.error(f"User {user_id} not found")
                return
            
            # Determine overall risk level
            risk_levels = {
                'diabetes': risk_data.get('diabetes', {}).get('category', 'Low'),
                'heart_disease': risk_data.get('heart_disease', {}).get('category', 'Low'),
                'hypertension': risk_data.get('hypertension', {}).get('category', 'Low')
            }
            
            # Check if any high risk
            high_risk_conditions = [k for k, v in risk_levels.items() if v == 'High']
            medium_risk_conditions = [k for k, v in risk_levels.items() if v == 'Medium']
            
            if high_risk_conditions:
                # Send urgent notifications
                self._send_urgent_notification(user, high_risk_conditions, risk_data)
                self._notify_hr_department(user, high_risk_conditions, risk_data)
                self._log_notification("HIGH_RISK", user.id, high_risk_conditions)
            elif medium_risk_conditions:
                # Send warning notifications
                self._send_warning_notification(user, medium_risk_conditions, risk_data)
                self._log_notification("MEDIUM_RISK", user.id, medium_risk_conditions)
            else:
                # Send positive reinforcement
                self._send_healthy_notification(user, risk_data)
                self._log_notification("LOW_RISK", user.id, [])
                
        except Exception as e:
            logger.error(f"Error sending health alert: {str(e)}")
        finally:
            db.close()
    
    def _send_urgent_notification(self, user, conditions, risk_data):
        """Send urgent notification for high-risk conditions"""
        subject = "🚨 URGENT: High Health Risk Detected"
        
        message = f"""
Dear {user.first_name} {user.last_name},

Your recent health assessment has identified HIGH RISK for the following conditions:
{', '.join([c.replace('_', ' ').title() for c in conditions])}

IMMEDIATE ACTIONS REQUIRED:
1. Schedule an appointment with your doctor within 48 hours
2. Begin monitoring vital signs daily
3. Follow the personalized recommendations in your dashboard

Risk Assessment Details:
"""
        
        for condition in conditions:
            if condition in risk_data:
                data = risk_data[condition]
                message += f"\n- {condition.replace('_', ' ').title()}: Risk Score: {data.get('probability', 0) * 100:.1f}%"
        
        message += f"""

Contact Information:
- Company Health Clinic: (555) 123-4567
- HR Health Coordinator: {self.hr_email}
- Emergency: 911

Please log in to WellTrackAI to view your full report.

Best regards,
WellTrackAI Team
"""
        
        self._send_email(user.email, subject, message)
    
    def _send_warning_notification(self, user, conditions, risk_data):
        """Send warning notification for medium-risk conditions"""
        subject = "⚠️ Health Risk Warning - Action Recommended"
        
        message = f"""
Dear {user.first_name} {user.last_name},

Your health assessment shows MEDIUM RISK for:
{', '.join([c.replace('_', ' ').title() for c in conditions])}

Recommended Actions:
1. Schedule a check-up within 2 weeks
2. Start implementing lifestyle changes
3. Monitor your health metrics weekly

Please log in to WellTrackAI to view detailed recommendations.

Best regards,
WellTrackAI Team
"""
        
        self._send_email(user.email, subject, message)
    
    def _send_healthy_notification(self, user, risk_data):
        """Send positive reinforcement for low-risk results"""
        subject = "✅ Great Health Assessment Results!"
        
        message = f"""
Dear {user.first_name} {user.last_name},

Congratulations! Your health assessment shows LOW RISK across all categories.

Keep up the great work! Continue with:
- Regular exercise
- Balanced nutrition
- Adequate sleep
- Stress management

Your next assessment is due in 3 months.

Best regards,
WellTrackAI Team
"""
        
        self._send_email(user.email, subject, message)
    
    def _notify_hr_department(self, user, conditions, risk_data):
        """Notify HR about high-risk employees"""
        subject = f"Employee Health Alert: {user.first_name} {user.last_name}"
        
        message = f"""
Employee Health Risk Alert

Employee: {user.first_name} {user.last_name} (ID: {user.employee_id})
Department: {user.department}
High Risk Conditions: {', '.join(conditions)}

Recommended HR Actions:
1. Ensure employee has access to healthcare resources
2. Consider workload adjustments if needed
3. Follow up on doctor's appointment scheduling
4. Review workplace wellness programs

This is an automated notification from WellTrackAI.
"""
        
        self._send_email(self.hr_email, subject, message)
    
    def _send_email(self, to_email, subject, body):
        """Send email notification"""
        if not self.email_configured:
            # Just log if email not configured
            logger.info(f"Email notification (not sent - email not configured):")
            logger.info(f"To: {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Body preview: {body[:100]}...")
            return True
        
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                
            logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
    
    def _log_notification(self, risk_level, user_id, conditions):
        """Log notification in database or file"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "risk_level": risk_level,
            "conditions": conditions
        }
        
        # Log to file
        log_file = "logs/notifications.log"
        os.makedirs("logs", exist_ok=True)
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"Notification logged: {risk_level} for user {user_id}")

# Simple function to use in your app
def send_notification_after_assessment(user_id, risk_data):
    """Send notification after health assessment"""
    try:
        service = NotificationService()
        service.send_health_alert(user_id, risk_data)
        return True
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")
        return False