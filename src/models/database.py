from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from config.settings import settings

Base = declarative_base()

class ClinicalTrial(Base):
    __tablename__ = "clinical_trials"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(String(50), nullable=False, index=True)
    trial_id = Column(String(50), nullable=False)
    phase = Column(String(20), nullable=False)
    therapeutic_area = Column(String(100), nullable=False)
    principal_investigator = Column(String(200))
    site_id = Column(String(50), nullable=False, index=True)
    region = Column(String(50))
    
    # Timeline tracking
    planned_start_date = Column(DateTime)
    actual_start_date = Column(DateTime)
    planned_end_date = Column(DateTime)
    actual_end_date = Column(DateTime)
    
    # Derived delay indicators
    is_delayed = Column(Boolean, default=False)
    delay_days = Column(Integer, default=0)
    delay_probability = Column(Float)  # AI prediction score
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patients = relationship("Patient", back_populates="trial")
    trial_metrics = relationship("TrialMetrics", back_populates="trial")

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String(50), nullable=False, index=True)
    trial_id = Column(UUID(as_uuid=True), ForeignKey("clinical_trials.id"))
    
    # Demographics
    age = Column(Integer)
    sex = Column(String(10))
    bmi = Column(Float)
    treatment_arm = Column(String(50))
    risk_score = Column(Float)
    
    # Enrollment tracking
    enrollment_date = Column(DateTime)
    last_visit_date = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trial = relationship("ClinicalTrial", back_populates="patients")
    visits = relationship("PatientVisit", back_populates="patient")

class PatientVisit(Base):
    __tablename__ = "patient_visits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"))
    week = Column(Integer, nullable=False)
    visit_date = Column(DateTime, nullable=False)
    
    # Vitals
    bp_systolic = Column(Float)
    bp_diastolic = Column(Float)
    
    # Lab values
    alt_level = Column(Float)
    ast_level = Column(Float)
    creatinine = Column(Float)
    
    # Adverse events
    adr_reported = Column(Boolean, default=False)
    adr_type = Column(String(100))
    adr_severity = Column(Integer)
    
    # Efficacy & Adherence
    efficacy_score = Column(Float)
    medication_adherence = Column(Float)
    satisfaction_score = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="visits")

class TrialMetrics(Base):
    __tablename__ = "trial_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_id = Column(UUID(as_uuid=True), ForeignKey("clinical_trials.id"))
    
    # Enrollment metrics
    target_enrollment = Column(Integer)
    current_enrollment = Column(Integer)
    enrollment_velocity = Column(Float)  # patients per week
    
    # Query and deviation metrics
    total_queries = Column(Integer, default=0)
    resolved_queries = Column(Integer, default=0)
    protocol_deviations = Column(Integer, default=0)
    major_deviations = Column(Integer, default=0)
    
    # Timeline metrics
    protocol_amendments = Column(Integer, default=0)
    data_entry_lag_days = Column(Float)
    
    # Calculated at time of record
    calculation_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trial = relationship("ClinicalTrial", back_populates="trial_metrics")

class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_id = Column(UUID(as_uuid=True), ForeignKey("clinical_trials.id"))
    
    # Model information
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'delay_prediction' or 'anomaly_detection'
    
    # Prediction results
    prediction_score = Column(Float, nullable=False)
    prediction_class = Column(String(20))  # 'delayed', 'on_time', 'anomaly'
    confidence = Column(Float)
    
    # Feature importance (JSON string)
    feature_importance = Column(Text)
    
    # Metadata
    prediction_date = Column(DateTime, default=datetime.utcnow)
    
# Database engine and session
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
