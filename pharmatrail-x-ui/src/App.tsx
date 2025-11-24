import React, { useState, useEffect } from 'react';
import { Activity, Brain, Database, FileText, Settings, Shield } from 'lucide-react';
import './App.css';

// Import components
import DashboardOverview from './components/DashboardOverview';
import DelayPredictor from './components/DelayPredictor';
import NLPInterface from './components/NLPInterface';
import DigitalTwin from './components/DigitalTwin';
import BlockchainViewer from './components/BlockchainViewer';
import DataIngestion from './components/DataIngestion';
import ModelManagement from './components/ModelManagement';
import { apiService } from './services/api';

type TabType = 'overview' | 'analytics' | 'nlp' | 'twin' | 'blockchain' | 'data' | 'models';

interface HealthStatus {
  status: string;
  phase2_components: {
    nlp_engine: boolean;
    blockchain_ledger: boolean;
  };
  phase3_components: {
    digital_twin: boolean;
  };
  capabilities: {
    nlp_processing: boolean;
    blockchain_audit: boolean;
    digital_twin_simulation: boolean;
    scenario_modeling: boolean;
    integrated_intelligence: boolean;
  };
}

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch health status on component mount
  useEffect(() => {
    const fetchHealthStatus = async () => {
      try {
        setIsLoading(true);
        const health = await apiService.getHealth();
        setHealthStatus(health);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch health status:', err);
        setError('Unable to connect to PharmaTrail-X backend. Please ensure the API is running on port 8006.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchHealthStatus();
    
    // Refresh health status every 10 minutes to reduce demo-time flicker
    const interval = setInterval(fetchHealthStatus, 600000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'overview' as TabType, label: 'Dashboard', icon: Activity, description: 'Trial health overview and KPIs' },
    { id: 'analytics' as TabType, label: 'Analytics', icon: Brain, description: 'AI delay prediction and risk analysis' },
    { id: 'nlp' as TabType, label: 'NLP Engine', icon: FileText, description: 'Adverse event extraction and text analysis' },
    { id: 'twin' as TabType, label: 'Digital Twin', icon: Settings, description: 'Trial simulation and scenario modeling' },
    { id: 'blockchain' as TabType, label: 'Audit Trail', icon: Shield, description: 'Blockchain event logging and verification' },
    { id: 'data' as TabType, label: 'Data Ingestion', icon: Database, description: 'CT.gov file upload and data management' },
    { id: 'models' as TabType, label: 'Model Management', icon: Brain, description: 'ML model training and inspection' },
  ];

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'overview':
        return <DashboardOverview healthStatus={healthStatus} />;
      case 'analytics':
        return <DelayPredictor />;
      case 'nlp':
        return <NLPInterface />;
      case 'twin':
        return <DigitalTwin />;
      case 'blockchain':
        return <BlockchainViewer />;
      case 'data':
        return <DataIngestion />;
      case 'models':
        return <ModelManagement />;
      default:
        return <DashboardOverview healthStatus={healthStatus} />;
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Connecting to PharmaTrail-X...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold text-gray-900">
                  PharmaTrail-X
                </h1>
              </div>
              <div className="ml-4">
                <span className="text-sm text-gray-500">
                  Clinical Trial Intelligence Platform
                </span>
              </div>
            </div>
            
            {/* System Status */}
            <div className="flex items-center space-x-4">
              {error ? (
                <div className="flex items-center space-x-2 text-red-600">
                  <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                  <span className="text-sm font-medium">Disconnected</span>
                </div>
              ) : healthStatus ? (
                <div className="flex items-center space-x-2 text-green-600">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium">Connected</span>
                </div>
              ) : null}
              
              {healthStatus && (
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-500">
                    Phase 1+2+3 Active
                  </span>
                  <div className="flex space-x-1">
                    <div className={`w-2 h-2 rounded-full ${healthStatus.phase2_components.nlp_engine ? 'bg-green-400' : 'bg-gray-300'}`}></div>
                    <div className={`w-2 h-2 rounded-full ${healthStatus.phase2_components.blockchain_ledger ? 'bg-green-400' : 'bg-gray-300'}`}></div>
                    <div className={`w-2 h-2 rounded-full ${healthStatus.phase3_components.digital_twin ? 'bg-green-400' : 'bg-gray-300'}`}></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
              <p className="text-xs text-red-600 mt-1">
                Start the backend with: <code>python phase3_integrated_api.py</code>
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex space-x-8">
          {/* Sidebar Navigation */}
          <div className="w-64 flex-shrink-0">
            <nav className="space-y-2">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center px-4 py-3 text-left rounded-lg transition-colors duration-200 ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 border border-blue-200'
                        : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                    }`}
                  >
                    <Icon className={`w-5 h-5 mr-3 ${isActive ? 'text-blue-600' : 'text-gray-400'}`} />
                    <div>
                      <div className="font-medium">{tab.label}</div>
                      <div className="text-xs text-gray-500 mt-1">{tab.description}</div>
                    </div>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1 min-w-0">
            {renderActiveComponent()}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
