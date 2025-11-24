import React, { useState } from 'react';
import { Settings, Play, BarChart3, Users, Clock } from 'lucide-react';
import { apiService, TrialConfig, SiteConfig, SimulationResponse } from '../services/api';

const DigitalTwin: React.FC = () => {
  const [trialConfig, setTrialConfig] = useState<TrialConfig>({
    trial_id: 'TWIN-' + Date.now(),
    target_enrollment: 100,
    planned_duration_weeks: 52,
    sites: [
      {
        site_id: 'SITE-001',
        name: 'Main Research Center',
        region: 'North America',
        capacity: 50,
        staff_count: 4,
        staff_skill_level: 0.8,
        enrollment_rate: 2.5,
        query_resolution_time: 5.0,
        data_entry_delay: 2.0,
      }
    ],
  });

  const [simulationDays, setSimulationDays] = useState(365);
  const [simulationResults, setSimulationResults] = useState<SimulationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSiteChange = (index: number, field: keyof SiteConfig, value: string | number) => {
    const updatedSites = [...trialConfig.sites];
    updatedSites[index] = {
      ...updatedSites[index],
      [field]: typeof value === 'string' ? value : Number(value),
    };
    setTrialConfig(prev => ({ ...prev, sites: updatedSites }));
  };

  const addSite = () => {
    const newSite: SiteConfig = {
      site_id: `SITE-${String(trialConfig.sites.length + 1).padStart(3, '0')}`,
      name: `Research Site ${trialConfig.sites.length + 1}`,
      region: 'North America',
      capacity: 30,
      staff_count: 3,
      staff_skill_level: 0.7,
      enrollment_rate: 2.0,
      query_resolution_time: 6.0,
      data_entry_delay: 2.5,
    };
    setTrialConfig(prev => ({
      ...prev,
      sites: [...prev.sites, newSite],
    }));
  };

  const removeSite = (index: number) => {
    if (trialConfig.sites.length > 1) {
      setTrialConfig(prev => ({
        ...prev,
        sites: prev.sites.filter((_, i) => i !== index),
      }));
    }
  };

  const runSimulation = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await apiService.runSimulation(trialConfig, simulationDays);
      setSimulationResults(result);
      
    } catch (err: any) {
      console.error('Simulation error:', err);
      setError(err.response?.data?.detail || 'Failed to run simulation. Please check if the API is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateString: string | undefined) => {
    if (!dateString) return 'Not estimated';
    return new Date(dateString).toLocaleDateString();
  };

  const getRiskColor = (riskScore: number) => {
    if (riskScore > 0.7) return 'text-red-600 bg-red-50';
    if (riskScore > 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-green-600 bg-green-50';
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Digital Twin Simulation</h1>
        <p className="text-gray-600 mt-2">
          Model and optimize clinical trial operations with AI-powered simulation
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="xl:col-span-2 space-y-6">
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Trial Configuration</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Trial ID
                </label>
                <input
                  type="text"
                  value={trialConfig.trial_id}
                  onChange={(e) => setTrialConfig(prev => ({ ...prev, trial_id: e.target.value }))}
                  className="input-field"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Enrollment
                </label>
                <input
                  type="number"
                  value={trialConfig.target_enrollment}
                  onChange={(e) => setTrialConfig(prev => ({ ...prev, target_enrollment: Number(e.target.value) }))}
                  className="input-field"
                  min="10"
                  max="10000"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Planned Duration (weeks)
                </label>
                <input
                  type="number"
                  value={trialConfig.planned_duration_weeks}
                  onChange={(e) => setTrialConfig(prev => ({ ...prev, planned_duration_weeks: Number(e.target.value) }))}
                  className="input-field"
                  min="4"
                  max="260"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Simulation Days
                </label>
                <input
                  type="number"
                  value={simulationDays}
                  onChange={(e) => setSimulationDays(Number(e.target.value))}
                  className="input-field"
                  min="30"
                  max="1095"
                />
              </div>
            </div>

            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Sites Configuration</h3>
              <button
                onClick={addSite}
                className="btn-secondary text-sm"
              >
                Add Site
              </button>
            </div>

            <div className="space-y-4">
              {trialConfig.sites.map((site, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-medium text-gray-900">Site {index + 1}</h4>
                    {trialConfig.sites.length > 1 && (
                      <button
                        onClick={() => removeSite(index)}
                        className="text-red-600 hover:text-red-800 text-sm"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Site ID
                      </label>
                      <input
                        type="text"
                        value={site.site_id}
                        onChange={(e) => handleSiteChange(index, 'site_id', e.target.value)}
                        className="input-field text-sm"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Name
                      </label>
                      <input
                        type="text"
                        value={site.name}
                        onChange={(e) => handleSiteChange(index, 'name', e.target.value)}
                        className="input-field text-sm"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Region
                      </label>
                      <select
                        value={site.region}
                        onChange={(e) => handleSiteChange(index, 'region', e.target.value)}
                        className="input-field text-sm"
                      >
                        <option value="North America">North America</option>
                        <option value="Europe">Europe</option>
                        <option value="Asia Pacific">Asia Pacific</option>
                        <option value="Latin America">Latin America</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Capacity
                      </label>
                      <input
                        type="number"
                        value={site.capacity}
                        onChange={(e) => handleSiteChange(index, 'capacity', Number(e.target.value))}
                        className="input-field text-sm"
                        min="1"
                        max="200"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Staff Count
                      </label>
                      <input
                        type="number"
                        value={site.staff_count}
                        onChange={(e) => handleSiteChange(index, 'staff_count', Number(e.target.value))}
                        className="input-field text-sm"
                        min="1"
                        max="20"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Staff Skill Level
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={site.staff_skill_level}
                        onChange={(e) => handleSiteChange(index, 'staff_skill_level', Number(e.target.value))}
                        className="input-field text-sm"
                        min="0.1"
                        max="1.0"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Enrollment Rate (patients/week)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={site.enrollment_rate}
                        onChange={(e) => handleSiteChange(index, 'enrollment_rate', Number(e.target.value))}
                        className="input-field text-sm"
                        min="0.1"
                        max="10.0"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Query Resolution Time (days)
                      </label>
                      <input
                        type="number"
                        step="0.5"
                        value={site.query_resolution_time}
                        onChange={(e) => handleSiteChange(index, 'query_resolution_time', Number(e.target.value))}
                        className="input-field text-sm"
                        min="0.5"
                        max="30.0"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        Data Entry Delay (days)
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={site.data_entry_delay}
                        onChange={(e) => handleSiteChange(index, 'data_entry_delay', Number(e.target.value))}
                        className="input-field text-sm"
                        min="0.1"
                        max="14.0"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="pt-6">
              <button
                onClick={runSimulation}
                disabled={isLoading}
                className="btn-primary flex items-center space-x-2 w-full justify-center"
              >
                <Play className="w-4 h-4" />
                <span>{isLoading ? 'Running Simulation...' : 'Run Digital Twin Simulation'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <Settings className="w-5 h-5 text-red-600 mr-2" />
                <span className="text-red-700 font-medium">Simulation Error</span>
              </div>
              <p className="text-red-600 text-sm mt-2">{error}</p>
            </div>
          )}

          {simulationResults && (
            <>
              {/* Key Metrics */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Results</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Trial ID:</span>
                    <span className="font-medium">{simulationResults.trial_id}</span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Simulation Days:</span>
                    <span className="font-medium">{simulationResults.simulation_days}</span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Completion Date:</span>
                    <span className="font-medium">{formatDate(simulationResults.completion_date)}</span>
                  </div>
                  
                  <div className={`p-3 rounded-lg ${getRiskColor(simulationResults.delay_risk_score)}`}>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Delay Risk Score:</span>
                      <span className="font-bold">{(simulationResults.delay_risk_score * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Patient Metrics */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Patient Metrics</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <Users className="w-4 h-4 text-blue-500" />
                    <span className="text-sm text-gray-600">Total Enrolled:</span>
                    <span className="font-semibold text-blue-600">{simulationResults.total_enrolled}</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Users className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-gray-600">Completed:</span>
                    <span className="font-semibold text-green-600">{simulationResults.total_completed}</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Users className="w-4 h-4 text-red-500" />
                    <span className="text-sm text-gray-600">Dropouts:</span>
                    <span className="font-semibold text-red-600">{simulationResults.total_dropouts}</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Users className="w-4 h-4 text-yellow-500" />
                    <span className="text-sm text-gray-600">Active:</span>
                    <span className="font-semibold text-yellow-600">{simulationResults.active_patients}</span>
                  </div>
                </div>
              </div>

              {/* Query Metrics */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Query Metrics</h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="w-4 h-4 text-blue-500" />
                    <span className="text-sm text-gray-600">Total Queries:</span>
                    <span className="font-semibold">{simulationResults.total_queries}</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-gray-600">Resolved:</span>
                    <span className="font-semibold text-green-600">{simulationResults.resolved_queries}</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="w-4 h-4 text-red-500" />
                    <span className="text-sm text-gray-600">Pending:</span>
                    <span className="font-semibold text-red-600">{simulationResults.pending_queries}</span>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Clock className="w-4 h-4 text-yellow-500" />
                    <span className="text-sm text-gray-600">Avg Data Delay:</span>
                    <span className="font-semibold">{simulationResults.avg_data_entry_delay.toFixed(1)} days</span>
                  </div>
                </div>
              </div>

              {/* Cost Estimate */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost Estimate</h3>
                <div className="text-center">
                  <span className="text-2xl font-bold text-green-600">
                    ${simulationResults.cost_estimate.toLocaleString()}
                  </span>
                  <p className="text-sm text-gray-600 mt-1">Estimated total cost</p>
                </div>
              </div>

              {/* Timeline Data Preview */}
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Timeline Data</h3>
                <p className="text-sm text-gray-600 mb-2">
                  {simulationResults.timeline_data.length} data points captured
                </p>
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-xs text-gray-500 space-y-1">
                    <div>Latest: Day {simulationResults.timeline_data[simulationResults.timeline_data.length - 1]?.day || 0}</div>
                    <div>Enrolled: {simulationResults.timeline_data[simulationResults.timeline_data.length - 1]?.enrolled_patients || 0}</div>
                    <div>Active: {simulationResults.timeline_data[simulationResults.timeline_data.length - 1]?.active_patients || 0}</div>
                  </div>
                </div>
              </div>
            </>
          )}

          {!simulationResults && !error && (
            <div className="card text-center py-12">
              <Settings className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Ready for Simulation</h3>
              <p className="text-gray-600">
                Configure your trial parameters and run the digital twin simulation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DigitalTwin;
