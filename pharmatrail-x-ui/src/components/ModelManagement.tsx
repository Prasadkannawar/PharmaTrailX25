import React, { useState, useEffect } from 'react';
import { Brain, Settings, TrendingUp, RefreshCw, Play, Info } from 'lucide-react';
import { apiService } from '../services/api';

const ModelManagement: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trainingResult, setTrainingResult] = useState<any>(null);

  const fetchModelInfo = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await apiService.getModelInfo();
      setModelInfo(result);
      
    } catch (err: any) {
      console.error('Model info error:', err);
      setError(err.response?.data?.detail || 'Failed to fetch model information.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTraining = async () => {
    try {
      setIsTraining(true);
      setError(null);
      
      // Mock training data for demo
      const trainingData = {
        model_type: 'delay_predictor',
        training_config: {
          epochs: 100,
          batch_size: 32,
          learning_rate: 0.001,
          validation_split: 0.2,
        },
        data_source: 'clinical_trials_dataset',
      };
      
      const result = await apiService.trainModel(trainingData);
      setTrainingResult(result);
      
      // Refresh model info after training
      setTimeout(() => {
        fetchModelInfo();
      }, 2000);
      
    } catch (err: any) {
      console.error('Training error:', err);
      setError(err.response?.data?.detail || 'Failed to start model training.');
    } finally {
      setIsTraining(false);
    }
  };

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const getModelStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'active':
      case 'ready':
        return 'text-green-600 bg-green-50';
      case 'training':
        return 'text-yellow-600 bg-yellow-50';
      case 'error':
      case 'failed':
        return 'text-red-600 bg-red-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Model Management</h1>
        <p className="text-gray-600 mt-2">
          Train, monitor, and manage machine learning models
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Model Information */}
        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Current Models</h2>
              <button
                onClick={fetchModelInfo}
                disabled={isLoading}
                className="btn-secondary flex items-center space-x-2"
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            )}

            {modelInfo ? (
              <div className="space-y-4">
                {/* Delay Prediction Model */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <Brain className="w-5 h-5 text-blue-600" />
                      <h3 className="font-semibold text-gray-900">Delay Prediction Model</h3>
                    </div>
                    <span className={`status-badge ${getModelStatusColor(modelInfo.delay_model?.status || 'unknown')}`}>
                      {modelInfo.delay_model?.status || 'Unknown'}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Algorithm:</span>
                      <span className="font-medium ml-2">{modelInfo.delay_model?.algorithm || 'XGBoost'}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Version:</span>
                      <span className="font-medium ml-2">{modelInfo.delay_model?.version || '1.0.0'}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Accuracy:</span>
                      <span className="font-medium ml-2">
                        {modelInfo.delay_model?.accuracy ? `${(modelInfo.delay_model.accuracy * 100).toFixed(1)}%` : '87.3%'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">Last Trained:</span>
                      <span className="font-medium ml-2">
                        {formatTimestamp(modelInfo.delay_model?.last_trained)}
                      </span>
                    </div>
                  </div>

                  {modelInfo.delay_model?.metrics && (
                    <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Performance Metrics</h4>
                      <div className="grid grid-cols-3 gap-3 text-xs">
                        <div>
                          <span className="text-gray-600">Precision:</span>
                          <span className="font-medium ml-1">{(modelInfo.delay_model.metrics.precision * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Recall:</span>
                          <span className="font-medium ml-1">{(modelInfo.delay_model.metrics.recall * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">F1-Score:</span>
                          <span className="font-medium ml-1">{(modelInfo.delay_model.metrics.f1_score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Feature Engineering */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3 mb-3">
                    <Settings className="w-5 h-5 text-purple-600" />
                    <h3 className="font-semibold text-gray-900">Feature Engineering</h3>
                  </div>
                  
                  <div className="text-sm space-y-2">
                    <div>
                      <span className="text-gray-600">Features:</span>
                      <span className="font-medium ml-2">{modelInfo.feature_count || 15} features</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Preprocessing:</span>
                      <span className="font-medium ml-2">StandardScaler, OneHotEncoder</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Feature Selection:</span>
                      <span className="font-medium ml-2">Recursive Feature Elimination</span>
                    </div>
                  </div>
                </div>

                {/* Training History */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3 mb-3">
                    <TrendingUp className="w-5 h-5 text-green-600" />
                    <h3 className="font-semibold text-gray-900">Training History</h3>
                  </div>
                  
                  <div className="text-sm space-y-2">
                    <div>
                      <span className="text-gray-600">Total Runs:</span>
                      <span className="font-medium ml-2">{modelInfo.training_runs || 12}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Best Accuracy:</span>
                      <span className="font-medium ml-2">89.7%</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Training Data:</span>
                      <span className="font-medium ml-2">{modelInfo.training_samples || '2,847'} samples</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : isLoading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Loading model information...</p>
              </div>
            ) : (
              <div className="text-center py-8">
                <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No model information available.</p>
              </div>
            )}
          </div>
        </div>

        {/* Training Panel */}
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Model Training</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Training Configuration
                </label>
                <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Epochs</label>
                      <input type="number" value="100" className="input-field text-sm" readOnly />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Batch Size</label>
                      <input type="number" value="32" className="input-field text-sm" readOnly />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Learning Rate</label>
                      <input type="number" value="0.001" step="0.001" className="input-field text-sm" readOnly />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Validation Split</label>
                      <input type="number" value="0.2" step="0.1" className="input-field text-sm" readOnly />
                    </div>
                  </div>
                </div>
              </div>

              <button
                onClick={handleTraining}
                disabled={isTraining}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>{isTraining ? 'Training in Progress...' : 'Start Training'}</span>
              </button>

              {isTraining && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600 mr-2"></div>
                    <span className="text-yellow-700 font-medium">Training Model</span>
                  </div>
                  <p className="text-yellow-600 text-sm mt-2">
                    This may take several minutes. Please wait...
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Training Result */}
          {trainingResult && (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Results</h3>
              <div className="space-y-3">
                <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                  <p className="text-green-800 font-medium">✅ Training Completed Successfully</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Training Time:</span>
                    <span className="font-medium ml-2">{trainingResult.training_time || '3.2 minutes'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Final Accuracy:</span>
                    <span className="font-medium ml-2">{trainingResult.accuracy || '88.5%'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Model Size:</span>
                    <span className="font-medium ml-2">{trainingResult.model_size || '2.3 MB'}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Status:</span>
                    <span className="font-medium text-green-600 ml-2">Active</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Model Insights */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Insights</h3>
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <Info className="w-4 h-4 text-blue-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Feature Importance</p>
                  <p className="text-xs text-gray-600">Top predictors: Patient age, medication adherence, ADR rate</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <TrendingUp className="w-4 h-4 text-green-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Performance Trend</p>
                  <p className="text-xs text-gray-600">Model accuracy has improved 12% over the last 3 training cycles</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <Settings className="w-4 h-4 text-purple-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Optimization</p>
                  <p className="text-xs text-gray-600">Consider hyperparameter tuning for further improvements</p>
                </div>
              </div>
            </div>
          </div>

          {/* Demo Notice */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center">
              <Info className="w-5 h-5 text-blue-600 mr-2" />
              <span className="text-blue-700 font-medium">Demo Mode</span>
            </div>
            <p className="text-blue-600 text-sm mt-2">
              This interface demonstrates model management capabilities. 
              Actual training connects to the Phase 1 ML pipeline and MLflow tracking.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelManagement;
