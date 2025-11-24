import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import joblib
from pathlib import Path
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from config.settings import settings
import json

class MLPipeline:
    def __init__(self):
        self.delay_model = None
        self.anomaly_model = None
        self.feature_importance = {}
        self.model_metrics = {}
        
        # Initialize MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    def train_delay_predictor(self, X: pd.DataFrame, y: pd.Series, 
                            hyperparameter_tuning: bool = True) -> Dict:
        """
        Train XGBoost delay prediction model
        """
        logger.info("Training delay prediction model")
        
        with mlflow.start_run(run_name="delay_prediction_training"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Log data info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("positive_class_ratio", y.mean())
            
            if hyperparameter_tuning:
                # Hyperparameter tuning
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                
                xgb_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss'
                )
                
                logger.info("Performing hyperparameter tuning...")
                grid_search = GridSearchCV(
                    xgb_model, param_grid, 
                    cv=5, scoring='roc_auc', 
                    n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                
                self.delay_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                # Log best parameters
                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)
                
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                
            else:
                # Use default parameters
                self.delay_model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss'
                )
                
                self.delay_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.delay_model.predict(X_test)
            y_pred_proba = self.delay_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Feature importance
            feature_importance = dict(zip(X.columns, self.delay_model.feature_importances_))
            self.feature_importance['delay_prediction'] = feature_importance
            
            # Log top 10 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(top_features):
                mlflow.log_metric(f"feature_importance_rank_{i+1}", importance)
                mlflow.log_param(f"top_feature_{i+1}", feature)
            
            # Log model
            mlflow.xgboost.log_model(self.delay_model, "delay_prediction_model")
            
            # Store metrics
            self.model_metrics['delay_prediction'] = metrics
            
            logger.info(f"Delay prediction model trained. AUC: {metrics['roc_auc']:.3f}")
            
            return metrics
    
    def train_anomaly_detector(self, X: pd.DataFrame) -> Dict:
        """
        Train Isolation Forest anomaly detection model
        """
        logger.info("Training anomaly detection model")
        
        with mlflow.start_run(run_name="anomaly_detection_training"):
            # Train Isolation Forest
            self.anomaly_model = IsolationForest(
                contamination=settings.ANOMALY_CONTAMINATION,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0
            )
            
            self.anomaly_model.fit(X)
            
            # Get anomaly scores
            anomaly_scores = self.anomaly_model.decision_function(X)
            anomaly_predictions = self.anomaly_model.predict(X)
            
            # Calculate metrics
            n_anomalies = np.sum(anomaly_predictions == -1)
            anomaly_rate = n_anomalies / len(X)
            
            metrics = {
                'n_samples': len(X),
                'n_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'mean_anomaly_score': np.mean(anomaly_scores),
                'std_anomaly_score': np.std(anomaly_scores)
            }
            
            # Log parameters and metrics
            mlflow.log_param("contamination", settings.ANOMALY_CONTAMINATION)
            mlflow.log_param("n_estimators", 100)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(self.anomaly_model, "anomaly_detection_model")
            
            # Store metrics
            self.model_metrics['anomaly_detection'] = metrics
            
            logger.info(f"Anomaly detection model trained. Anomaly rate: {anomaly_rate:.3f}")
            
            return metrics
    
    def predict_delay(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict delay probability for new data
        """
        if self.delay_model is None:
            raise ValueError("Delay prediction model not trained yet")
        
        predictions = self.delay_model.predict(X)
        probabilities = self.delay_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def detect_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new data
        """
        if self.anomaly_model is None:
            raise ValueError("Anomaly detection model not trained yet")
        
        predictions = self.anomaly_model.predict(X)
        scores = self.anomaly_model.decision_function(X)
        
        return predictions, scores
    
    def get_feature_importance(self, model_type: str = 'delay_prediction') -> Dict:
        """
        Get feature importance for specified model
        """
        if model_type not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_type}")
        
        return self.feature_importance[model_type]
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate trained model on test data
        """
        if self.delay_model is None:
            raise ValueError("Model not trained yet")
        
        # Predictions
        y_pred = self.delay_model.predict(X_test)
        y_pred_proba = self.delay_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'metrics': metrics,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_models(self, path: str):
        """
        Save trained models to disk
        """
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        if self.delay_model is not None:
            joblib.dump(self.delay_model, save_path / "delay_prediction_model.pkl")
            logger.info(f"Saved delay prediction model to {save_path}")
        
        if self.anomaly_model is not None:
            joblib.dump(self.anomaly_model, save_path / "anomaly_detection_model.pkl")
            logger.info(f"Saved anomaly detection model to {save_path}")
        
        # Save feature importance and metrics
        with open(save_path / "feature_importance.json", 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        with open(save_path / "model_metrics.json", 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
    
    def load_models(self, path: str):
        """
        Load trained models from disk
        """
        load_path = Path(path)
        
        delay_model_path = load_path / "delay_prediction_model.pkl"
        if delay_model_path.exists():
            self.delay_model = joblib.load(delay_model_path)
            logger.info("Loaded delay prediction model")
        
        anomaly_model_path = load_path / "anomaly_detection_model.pkl"
        if anomaly_model_path.exists():
            self.anomaly_model = joblib.load(anomaly_model_path)
            logger.info("Loaded anomaly detection model")
        
        # Load feature importance and metrics
        feature_importance_path = load_path / "feature_importance.json"
        if feature_importance_path.exists():
            with open(feature_importance_path, 'r') as f:
                self.feature_importance = json.load(f)
        
        metrics_path = load_path / "model_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.model_metrics = json.load(f)
    
    def get_model_info(self) -> Dict:
        """
        Get information about trained models
        """
        info = {
            'delay_model_trained': self.delay_model is not None,
            'anomaly_model_trained': self.anomaly_model is not None,
            'metrics': self.model_metrics,
            'feature_importance_available': bool(self.feature_importance)
        }
        
        if self.delay_model is not None:
            info['delay_model_type'] = type(self.delay_model).__name__
            info['delay_model_params'] = self.delay_model.get_params()
        
        if self.anomaly_model is not None:
            info['anomaly_model_type'] = type(self.anomaly_model).__name__
            info['anomaly_model_params'] = self.anomaly_model.get_params()
        
        return info
