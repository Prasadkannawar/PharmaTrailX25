#!/usr/bin/env python3
"""
PharmaTrail-X Phase 1 Training Pipeline
Main script to execute the complete training pipeline from data ingestion to model deployment
"""

import sys
import os
from pathlib import Path
import pandas as pd
from loguru import logger
import click

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ingestion.data_ingester import DataIngester
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.ml_pipeline import MLPipeline
from src.models.database import create_tables, get_db
from config.settings import settings

# Configure logging
logger.add(
    settings.LOG_FILE,
    rotation="10 MB",
    retention="30 days",
    level=settings.LOG_LEVEL
)

@click.command()
@click.option('--csv-path', default='PharmaTrailX_ClinicalTrialMaster2.csv', 
              help='Path to clinical trials CSV file')
@click.option('--hyperparameter-tuning', is_flag=True, default=True,
              help='Enable hyperparameter tuning for XGBoost')
@click.option('--generate-fhir', is_flag=True, default=True,
              help='Generate mock FHIR data')
@click.option('--load-to-db', is_flag=True, default=False,
              help='Load processed data to PostgreSQL database')
def main(csv_path: str, hyperparameter_tuning: bool, generate_fhir: bool, load_to_db: bool):
    """
    Execute the complete PharmaTrail-X Phase 1 training pipeline
    """
    logger.info("🚀 Starting PharmaTrail-X Phase 1 Training Pipeline")
    
    try:
        # Initialize components
        data_ingester = DataIngester()
        feature_engineer = FeatureEngineer()
        ml_pipeline = MLPipeline()
        
        # Step 1: Data Ingestion
        logger.info("📊 Step 1: Data Ingestion")
        
        # Ingest clinical trials CSV
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            sys.exit(1)
        
        df = data_ingester.ingest_clinical_trials_csv(csv_path)
        logger.info(f"✅ Ingested {len(df)} records from {csv_path}")
        
        # Generate mock FHIR data
        if generate_fhir:
            fhir_data = data_ingester.generate_mock_fhir_data(num_trials=10)
            logger.info(f"✅ Generated {len(fhir_data)} mock FHIR trial records")
        
        # Step 2: Feature Engineering
        logger.info("🔧 Step 2: Feature Engineering")
        
        df_features = feature_engineer.engineer_features(df)
        X, feature_names = feature_engineer.prepare_ml_features(df_features)
        
        logger.info(f"✅ Created {len(feature_names)} features for ML")
        logger.info(f"📈 Feature matrix shape: {X.shape}")
        
        # Display top features by importance (if available)
        logger.info("🔍 Top engineered features:")
        for i, feature in enumerate(feature_names[:10]):
            logger.info(f"  {i+1}. {feature}")
        
        # Step 3: Model Training
        logger.info("🤖 Step 3: Model Training")
        
        # Prepare target variable
        y = (df_features['delay_probability'] > settings.DELAY_THRESHOLD).astype(int)
        logger.info(f"📊 Target distribution - Delayed: {y.sum()}, On-time: {(~y.astype(bool)).sum()}")
        
        # Train delay prediction model
        logger.info("🎯 Training XGBoost delay prediction model...")
        delay_metrics = ml_pipeline.train_delay_predictor(
            X, y, hyperparameter_tuning=hyperparameter_tuning
        )
        
        logger.info("✅ Delay prediction model training completed")
        logger.info(f"   📊 Accuracy: {delay_metrics['accuracy']:.3f}")
        logger.info(f"   📊 AUC: {delay_metrics['roc_auc']:.3f}")
        logger.info(f"   📊 F1-Score: {delay_metrics['f1_score']:.3f}")
        
        # Train anomaly detection model
        logger.info("🔍 Training Isolation Forest anomaly detection model...")
        anomaly_metrics = ml_pipeline.train_anomaly_detector(X)
        
        logger.info("✅ Anomaly detection model training completed")
        logger.info(f"   📊 Anomaly rate: {anomaly_metrics['anomaly_rate']:.3f}")
        logger.info(f"   📊 Samples processed: {anomaly_metrics['n_samples']}")
        
        # Step 4: Model Persistence
        logger.info("💾 Step 4: Model Persistence")
        
        # Create model directory
        model_path = Path(settings.MODEL_REGISTRY_PATH)
        model_path.mkdir(exist_ok=True)
        
        # Save models and preprocessors
        ml_pipeline.save_models(str(model_path))
        feature_engineer.save_preprocessors(str(model_path))
        
        logger.info(f"✅ Models saved to {model_path}")
        
        # Step 5: Database Loading (optional)
        if load_to_db:
            logger.info("🗄️ Step 5: Database Loading")
            
            # Create database tables
            create_tables()
            
            # Load data to database
            db_session = next(get_db())
            try:
                data_ingester.load_to_database(df_features, db_session)
                logger.info("✅ Data loaded to PostgreSQL database")
            except Exception as e:
                logger.error(f"❌ Database loading failed: {e}")
            finally:
                db_session.close()
        
        # Step 6: Model Validation
        logger.info("✅ Step 6: Model Validation")
        
        # Test prediction on sample data
        sample_data = X.iloc[:5]
        delay_pred, delay_prob = ml_pipeline.predict_delay(sample_data)
        anomaly_pred, anomaly_scores = ml_pipeline.detect_anomalies(sample_data)
        
        logger.info("🧪 Sample predictions:")
        for i in range(len(sample_data)):
            logger.info(f"   Sample {i+1}: Delay prob={delay_prob[i]:.3f}, "
                       f"Anomaly score={anomaly_scores[i]:.3f}")
        
        # Feature importance
        feature_importance = ml_pipeline.get_feature_importance('delay_prediction')
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info("🎯 Top 10 most important features:")
        for i, (feature, importance) in enumerate(top_features):
            logger.info(f"   {i+1}. {feature}: {importance:.4f}")
        
        # Summary
        logger.info("🎉 Phase 1 Training Pipeline Completed Successfully!")
        logger.info("📋 Summary:")
        logger.info(f"   📊 Data samples processed: {len(df)}")
        logger.info(f"   🔧 Features engineered: {len(feature_names)}")
        logger.info(f"   🎯 Delay prediction AUC: {delay_metrics['roc_auc']:.3f}")
        logger.info(f"   🔍 Anomaly detection rate: {anomaly_metrics['anomaly_rate']:.3f}")
        logger.info(f"   💾 Models saved to: {model_path}")
        
        logger.info("🚀 Ready for Phase 2: NLP Engine and Blockchain Audit Layer")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
