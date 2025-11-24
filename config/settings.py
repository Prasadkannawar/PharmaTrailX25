from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = "postgresql://pharmax_user:pharmax_pass@localhost:5432/pharmax_db"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "pharmax_delay_prediction"
    
    # Data Paths
    RAW_DATA_PATH: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"
    
    # Model Configuration
    MODEL_REGISTRY_PATH: str = "models"
    DELAY_THRESHOLD: float = 0.5  # Probability threshold for delay classification
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Kafka Configuration (for future streaming)
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_TRIALS: str = "clinical_trials"
    
    # Feature Engineering
    FEATURE_WINDOW_DAYS: int = 30
    ANOMALY_CONTAMINATION: float = 0.1
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/pharmax.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
