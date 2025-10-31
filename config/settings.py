import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # App Settings
    APP_NAME = os.getenv("APP_NAME", "WellTrackAI")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # File Upload
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf', '.csv'}
    
    # AI Model Settings
    MODEL_UPDATE_FREQUENCY = "monthly"
    RISK_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8,
        "critical": 0.9
    }

settings = Settings()