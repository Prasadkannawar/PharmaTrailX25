import React, { useState } from 'react';
import { AlertTriangle, TrendingUp, Calculator, Zap } from 'lucide-react';
import Plot from 'react-plotly.js';
import { apiService, TrialData, PredictionResponse } from '../services/api';

const DelayPredictor: React.FC = () => {
  const [formData, setFormData] = useState<TrialData>({
    Age: 65,
    Medication_Adherence: 80,
    ADR_Rate: 0.15,
    BP_Systolic: 130,
    ALT_Level: 25,
    trial_id: 'PRED-' + Date.now(),
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: keyof TrialData, value: number | string) => {
    setFormData(prev => ({
      ...prev,
      [field]: typeof value === 'string' ? value : Number(value),
    }));
  };

  const handlePredict = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await apiService.predictDelay(formData);
      setPrediction(result);
      
    } catch (err: any) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.detail || 'Failed to get prediction. Please check if the API is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleIntegratedPredict = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await apiService.getIntegratedPrediction(formData);
      setPrediction({
        delay_probability: result.delay_probability,
        delay_prediction: result.delay_prediction,
        confidence: result.confidence,
        risk_factors: result.risk_factors,
      });
      
    } catch (err: any) {
      console.error('Integrated prediction error:', err);
      setError(err.response?.data?.detail || 'Failed to get integrated prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  // Create gauge chart data
  const createGaugeData = (value: number, title: string) => {
    return {
      type: 'indicator' as const,
      mode: 'gauge+number+delta' as const,
      value: value * 100,
      domain: { x: [0, 1], y: [0, 1] },
      title: { text: title },
      gauge: {
        axis: { range: [0, 100] },
        bar: { color: value > 0.7 ? '#ef4444' : value > 0.4 ? '#f59e0b' : '#10b981' },
        steps: [
          { range: [0, 40], color: '#dcfce7' },
          { range: [40, 70], color: '#fef3c7' },
          { range: [70, 100], color: '#fee2e2' },
        ],
        threshold: {
          line: { color: 'red', width: 4 },
          thickness: 0.75,
          value: 80,
        },
      },
    };
  };

  const getRiskColor = (probability: number) => {
    if (probability > 0.7) return 'text-red-600 bg-red-50 border-red-200';
    if (probability > 0.4) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-green-600 bg-green-50 border-green-200';
  };

  const getRiskLabel = (probability: number) => {
    if (probability > 0.7) return 'High Risk';
    if (probability > 0.4) return 'Medium Risk';
    return 'Low Risk';
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">AI Delay Predictor</h1>
        <p className="text-gray-600 mt-2">
          Predict trial delays using machine learning and patient data
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Trial Parameters</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Trial ID
              </label>
              <input
                type="text"
                value={formData.trial_id}
                onChange={(e) => handleInputChange('trial_id', e.target.value)}
                className="input-field"
                placeholder="Enter trial identifier"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Average Patient Age
              </label>
              <input
                type="number"
                value={formData.Age}
                onChange={(e) => handleInputChange('Age', parseFloat(e.target.value))}
                className="input-field"
                min="18"
                max="100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Medication Adherence (%)
              </label>
              <input
                type="number"
                value={formData.Medication_Adherence}
                onChange={(e) => handleInputChange('Medication_Adherence', parseFloat(e.target.value))}
                className="input-field"
                min="0"
                max="100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Adverse Drug Reaction Rate
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.ADR_Rate}
                onChange={(e) => handleInputChange('ADR_Rate', parseFloat(e.target.value))}
                className="input-field"
                min="0"
                max="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Average Systolic BP (mmHg)
              </label>
              <input
                type="number"
                value={formData.BP_Systolic}
                onChange={(e) => handleInputChange('BP_Systolic', parseFloat(e.target.value))}
                className="input-field"
                min="80"
                max="200"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Average ALT Level (U/L)
              </label>
              <input
                type="number"
                value={formData.ALT_Level}
                onChange={(e) => handleInputChange('ALT_Level', parseFloat(e.target.value))}
                className="input-field"
                min="0"
                max="200"
              />
            </div>

            <div className="flex space-x-3 pt-4">
              <button
                onClick={handlePredict}
                disabled={isLoading}
                className="btn-primary flex items-center space-x-2 flex-1"
              >
                <Calculator className="w-4 h-4" />
                <span>{isLoading ? 'Predicting...' : 'Predict Delay'}</span>
              </button>
              
              <button
                onClick={handleIntegratedPredict}
                disabled={isLoading}
                className="btn-secondary flex items-center space-x-2 flex-1"
              >
                <Zap className="w-4 h-4" />
                <span>Integrated AI</span>
              </button>
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
                <span className="text-red-700 font-medium">Prediction Error</span>
              </div>
              <p className="text-red-600 text-sm mt-2">{error}</p>
            </div>
          )}

          {prediction && (
            <>
              {/* Risk Gauge */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Delay Risk Assessment</h3>
                <div className="flex items-center justify-center">
                  <Plot
                    data={[createGaugeData(prediction.delay_probability, 'Delay Probability')]}
                    layout={{
                      width: 300,
                      height: 250,
                      margin: { l: 20, r: 20, t: 40, b: 20 },
                      font: { size: 12 },
                    }}
                    config={{ displayModeBar: false }}
                  />
                </div>
                
                <div className={`mt-4 p-4 rounded-lg border ${getRiskColor(prediction.delay_probability)}`}>
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">Risk Level: {getRiskLabel(prediction.delay_probability)}</span>
                    <span className="text-sm">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm mt-2">
                    Prediction: <strong>{prediction.delay_prediction}</strong>
                  </p>
                </div>
              </div>

              {/* Risk Factors */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Factors</h3>
                {prediction.risk_factors.length > 0 ? (
                  <div className="space-y-2">
                    {prediction.risk_factors.map((factor, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{factor}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No significant risk factors identified.</p>
                )}
              </div>

              {/* Recommendations */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommendations</h3>
                <div className="space-y-3">
                  {prediction.delay_probability > 0.7 && (
                    <div className="flex items-start space-x-3">
                      <TrendingUp className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-gray-900">High Risk Mitigation</p>
                        <p className="text-sm text-gray-600">Consider additional site monitoring and patient support programs.</p>
                      </div>
                    </div>
                  )}
                  
                  {prediction.delay_probability > 0.4 && (
                    <div className="flex items-start space-x-3">
                      <TrendingUp className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-gray-900">Enhanced Monitoring</p>
                        <p className="text-sm text-gray-600">Implement more frequent check-ins and data quality reviews.</p>
                      </div>
                    </div>
                  )}
                  
                  <div className="flex items-start space-x-3">
                    <TrendingUp className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">Digital Twin Analysis</p>
                      <p className="text-sm text-gray-600">Run scenario modeling to optimize trial parameters and reduce risk.</p>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {!prediction && !error && (
            <div className="card text-center py-12">
              <Calculator className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Ready for Prediction</h3>
              <p className="text-gray-600">
                Enter trial parameters and click "Predict Delay" to get AI-powered risk assessment.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DelayPredictor;
