#!/usr/bin/env python3
"""
PharmaTrail-X Quick Start Script
Automated setup and execution of Phase 1 pipeline
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import click
from loguru import logger

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 
        'mlflow', 'fastapi', 'uvicorn', 'sqlalchemy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        logger.info("💡 Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies satisfied")
    return True

def setup_directories():
    """Create necessary directories"""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "data/raw", "data/processed", "logs", "models",
        "src/ingestion", "src/preprocessing", "src/models", "src/api"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directories created")

def run_training_pipeline():
    """Execute the training pipeline"""
    logger.info("🚀 Starting training pipeline...")
    
    try:
        result = subprocess.run([
            sys.executable, "train_pipeline.py", 
            "--hyperparameter-tuning", "--generate-fhir"
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            logger.info("✅ Training pipeline completed successfully")
            return True
        else:
            logger.error(f"❌ Training pipeline failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Training pipeline timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Training pipeline error: {e}")
        return False

def start_api_server(port=8000):
    """Start the FastAPI server"""
    logger.info(f"🌐 Starting API server on port {port}...")
    
    try:
        # Change to API directory
        api_dir = Path("src/api")
        
        # Start uvicorn server
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0", "--port", str(port), "--reload"
        ], cwd=api_dir)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        logger.info(f"✅ API server started at http://localhost:{port}")
        logger.info(f"📚 API docs available at http://localhost:{port}/docs")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")
        return False

def start_mlflow_ui(port=5000):
    """Start MLflow UI"""
    logger.info(f"📊 Starting MLflow UI on port {port}...")
    
    try:
        subprocess.Popen([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--port", str(port)
        ])
        
        time.sleep(2)
        logger.info(f"✅ MLflow UI started at http://localhost:{port}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start MLflow UI: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    logger.info("🧪 Testing API endpoints...")
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/analytics/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ API health check passed")
            
            # Test model info endpoint
            response = requests.get("http://localhost:8000/analytics/model_info", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Model info endpoint working")
                return True
            else:
                logger.warning("⚠️ Model info endpoint not ready")
                return False
        else:
            logger.error("❌ API health check failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False

@click.command()
@click.option('--skip-training', is_flag=True, help='Skip training pipeline')
@click.option('--api-port', default=8000, help='API server port')
@click.option('--mlflow-port', default=5000, help='MLflow UI port')
@click.option('--test-api', is_flag=True, default=True, help='Test API after startup')
def main(skip_training, api_port, mlflow_port, test_api):
    """
    PharmaTrail-X Quick Start
    
    This script will:
    1. Check dependencies
    2. Set up directories
    3. Run training pipeline (optional)
    4. Start API server
    5. Start MLflow UI
    6. Test endpoints
    """
    
    logger.info("🚀 PharmaTrail-X Phase 1 Quick Start")
    logger.info("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Please install requirements first.")
        sys.exit(1)
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Training pipeline
    if not skip_training:
        if not run_training_pipeline():
            logger.error("❌ Training failed. Continuing with API startup...")
    else:
        logger.info("⏭️ Skipping training pipeline")
    
    # Step 4: Start API server
    if not start_api_server(api_port):
        logger.error("❌ Failed to start API server")
        sys.exit(1)
    
    # Step 5: Start MLflow UI
    start_mlflow_ui(mlflow_port)
    
    # Step 6: Test API
    if test_api:
        time.sleep(5)  # Wait for services to fully start
        test_api_endpoints()
    
    # Summary
    logger.info("🎉 PharmaTrail-X Phase 1 is ready!")
    logger.info("=" * 50)
    logger.info(f"🌐 API Server: http://localhost:{api_port}")
    logger.info(f"📚 API Docs: http://localhost:{api_port}/docs")
    logger.info(f"📊 MLflow UI: http://localhost:{mlflow_port}")
    logger.info("=" * 50)
    
    # Example API calls
    logger.info("💡 Example API Usage:")
    logger.info(f"""
    # Health check
    curl http://localhost:{api_port}/analytics/health
    
    # Model info
    curl http://localhost:{api_port}/analytics/model_info
    
    # Make prediction
    curl -X POST http://localhost:{api_port}/analytics/predict \\
         -H "Content-Type: application/json" \\
         -d '{{"trial_data": {{"study_id": "PHX-2025-01", "phase": "Phase III"}}, "include_anomaly_detection": true}}'
    """)
    
    logger.info("🔄 Services are running. Press Ctrl+C to stop.")
    
    try:
        # Keep script running
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("👋 Shutting down PharmaTrail-X...")

if __name__ == "__main__":
    main()
