import React, { useState } from 'react';
import { FileText, Zap, AlertTriangle, CheckCircle } from 'lucide-react';
import { apiService, NLPResponse } from '../services/api';

const NLPInterface: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [trialId, setTrialId] = useState('NLP-' + Date.now());
  const [nlpResults, setNlpResults] = useState<NLPResponse | null>(null);
  const [summaryResults, setSummaryResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'ae' | 'summary'>('ae');

  const sampleTexts = {
    adverseEvent: `Patient reported severe headache and nausea following administration of study drug. Symptoms began approximately 2 hours post-dose and persisted for 6 hours. Patient's blood pressure was elevated at 160/95 mmHg. No other significant findings. Patient recovered without intervention.`,
    labResults: `Laboratory results show elevated ALT levels at 85 U/L (normal range 7-40 U/L). Creatinine levels are within normal limits at 1.1 mg/dL. Complete blood count reveals mild leukopenia with WBC count of 3.2 x10^3/μL. Patient remains asymptomatic.`,
    protocolDeviation: `Protocol deviation noted: Patient received study medication 3 hours late due to scheduling conflict. Patient was supposed to receive dose at 08:00 but received at 11:00. No adverse effects observed. Principal investigator notified and deviation documented.`,
  };

  const handleExtractAE = async () => {
    if (!inputText.trim()) {
      setError('Please enter clinical text to analyze.');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      const result = await apiService.extractAdverseEvents(inputText, trialId);
      setNlpResults(result);
      
    } catch (err: any) {
      console.error('AE extraction error:', err);
      setError(err.response?.data?.detail || 'Failed to extract adverse events. Please check if the API is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSummarize = async () => {
    if (!inputText.trim()) {
      setError('Please enter clinical text to summarize.');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      const result = await apiService.summarizeText(inputText, 150);
      setSummaryResults(result);
      
    } catch (err: any) {
      console.error('Summarization error:', err);
      setError(err.response?.data?.detail || 'Failed to summarize text. Please check if the API is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleText = (type: keyof typeof sampleTexts) => {
    setInputText(sampleTexts[type]);
    setNlpResults(null);
    setSummaryResults(null);
    setError(null);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'severe':
      case 'life-threatening':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'moderate':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'mild':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600';
    if (confidence > 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Clinical NLP Engine</h1>
        <p className="text-gray-600 mt-2">
          Extract adverse events and summarize clinical text using advanced NLP
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <div className="space-y-6">
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Clinical Text Input</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Trial ID
                </label>
                <input
                  type="text"
                  value={trialId}
                  onChange={(e) => setTrialId(e.target.value)}
                  className="input-field"
                  placeholder="Enter trial identifier"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Clinical Text
                </label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  className="input-field min-h-[200px]"
                  placeholder="Paste clinical text, adverse event reports, or lab results here..."
                />
                <p className="text-xs text-gray-500 mt-1">
                  {inputText.length} characters
                </p>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={handleExtractAE}
                  disabled={isLoading}
                  className="btn-primary flex items-center space-x-2 flex-1"
                >
                  <Zap className="w-4 h-4" />
                  <span>{isLoading && activeTab === 'ae' ? 'Extracting...' : 'Extract AEs'}</span>
                </button>
                
                <button
                  onClick={handleSummarize}
                  disabled={isLoading}
                  className="btn-secondary flex items-center space-x-2 flex-1"
                >
                  <FileText className="w-4 h-4" />
                  <span>{isLoading && activeTab === 'summary' ? 'Summarizing...' : 'Summarize'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Sample Texts */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Sample Clinical Texts</h3>
            <div className="space-y-2">
              <button
                onClick={() => handleSampleText('adverseEvent')}
                className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
              >
                <div className="font-medium text-sm text-gray-900">Adverse Event Report</div>
                <div className="text-xs text-gray-500 mt-1">Patient reported severe headache and nausea...</div>
              </button>
              
              <button
                onClick={() => handleSampleText('labResults')}
                className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
              >
                <div className="font-medium text-sm text-gray-900">Laboratory Results</div>
                <div className="text-xs text-gray-500 mt-1">Laboratory results show elevated ALT levels...</div>
              </button>
              
              <button
                onClick={() => handleSampleText('protocolDeviation')}
                className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
              >
                <div className="font-medium text-sm text-gray-900">Protocol Deviation</div>
                <div className="text-xs text-gray-500 mt-1">Protocol deviation noted: Patient received...</div>
              </button>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <AlertTriangle className="w-5 h-5 text-red-600 mr-2" />
                <span className="text-red-700 font-medium">Processing Error</span>
              </div>
              <p className="text-red-600 text-sm mt-2">{error}</p>
            </div>
          )}

          {/* Tab Navigation */}
          <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab('ae')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'ae'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Adverse Events
            </button>
            <button
              onClick={() => setActiveTab('summary')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'summary'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Summary
            </button>
          </div>

          {/* Adverse Events Results */}
          {activeTab === 'ae' && nlpResults && (
            <div className="space-y-4">
              {/* Overall Classification */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Results</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Overall Severity</p>
                    <span className={`status-badge ${getSeverityColor(nlpResults.severity_classification)}`}>
                      {nlpResults.severity_classification}
                    </span>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Confidence Score</p>
                    <span className={`font-semibold ${getConfidenceColor(nlpResults.confidence_score)}`}>
                      {(nlpResults.confidence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Extracted Entities */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Extracted Entities ({nlpResults.entities.length})
                </h3>
                {nlpResults.entities.length > 0 ? (
                  <div className="space-y-3">
                    {nlpResults.entities.map((entity, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900">{entity.text}</span>
                          <span className="text-xs text-gray-500">
                            {entity.confidence ? `${(entity.confidence * 100).toFixed(0)}%` : 'N/A'}
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="status-badge bg-blue-100 text-blue-800">
                            {entity.label}
                          </span>
                          {entity.severity && (
                            <span className={`status-badge ${getSeverityColor(entity.severity)}`}>
                              {entity.severity}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">No entities extracted from the text.</p>
                )}
              </div>

              {/* Adverse Events */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Adverse Events ({nlpResults.ae_events.length})
                </h3>
                {nlpResults.ae_events.length > 0 ? (
                  <div className="space-y-3">
                    {nlpResults.ae_events.map((ae, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900">{ae.event}</span>
                          <span className="text-xs text-gray-500">
                            {ae.confidence ? `${(ae.confidence * 100).toFixed(0)}%` : 'N/A'}
                          </span>
                        </div>
                        <span className={`status-badge ${getSeverityColor(ae.severity)}`}>
                          {ae.severity}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">No adverse events detected in the text.</p>
                )}
              </div>
            </div>
          )}

          {/* Summary Results */}
          {activeTab === 'summary' && summaryResults && (
            <div className="space-y-4">
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Text Summary</h3>
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <p className="text-gray-800">{summaryResults.summary}</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Word Count Reduction:</span>
                    <span className="font-semibold text-green-600 ml-2">
                      {(summaryResults.word_count_reduction * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">Confidence:</span>
                    <span className={`font-semibold ml-2 ${getConfidenceColor(summaryResults.confidence_score)}`}>
                      {(summaryResults.confidence_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Points</h3>
                {summaryResults.key_points && summaryResults.key_points.length > 0 ? (
                  <ul className="space-y-2">
                    {summaryResults.key_points.map((point: string, index: number) => (
                      <li key={index} className="flex items-start space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{point}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-gray-500 text-sm">No key points extracted.</p>
                )}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!nlpResults && !summaryResults && !error && (
            <div className="card text-center py-12">
              <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Ready for Analysis</h3>
              <p className="text-gray-600">
                Enter clinical text and choose an analysis type to get started.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default NLPInterface;
