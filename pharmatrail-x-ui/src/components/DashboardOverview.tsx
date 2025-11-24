import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle, Clock, Users, TrendingUp } from 'lucide-react';
import Plot from 'react-plotly.js';

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

interface DashboardOverviewProps {
  healthStatus: HealthStatus | null;
}

const DashboardOverview: React.FC<DashboardOverviewProps> = ({ healthStatus }) => {
  const [kpis] = useState({
    activeTrials: 12,
    totalPatients: 2847,
    completionRate: 78.5,
    avgDelayRisk: 0.23,
    adverseEvents: 156,
    queryResolutionTime: 4.2,
  });

  // Mock data for charts
  const enrollmentData = {
    x: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
    y: [45, 67, 89, 112, 134, 156],
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    name: 'Patient Enrollment',
    line: { color: '#3b82f6' },
  };

  const riskDistribution = {
    values: [65, 25, 10],
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    type: 'pie' as const,
    marker: {
      colors: ['#10b981', '#f59e0b', '#ef4444'],
    },
  };

  const getStatusColor = (isActive: boolean) => {
    return isActive ? 'text-green-600' : 'text-red-600';
  };

  const getStatusIcon = (isActive: boolean) => {
    return isActive ? CheckCircle : AlertTriangle;
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard Overview</h1>
        <p className="text-gray-600 mt-2">
          Real-time insights into your clinical trial portfolio
        </p>
      </div>

      {/* System Status Cards */}
      {healthStatus && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="card">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Activity className="w-5 h-5 text-blue-600" />
                </div>
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-medium text-gray-900">NLP Engine</h3>
                <div className="flex items-center mt-1">
                  {React.createElement(
                    getStatusIcon(healthStatus.phase2_components.nlp_engine),
                    { className: `w-4 h-4 ${getStatusColor(healthStatus.phase2_components.nlp_engine)} mr-1` }
                  )}
                  <span className={`text-sm font-medium ${getStatusColor(healthStatus.phase2_components.nlp_engine)}`}>
                    {healthStatus.phase2_components.nlp_engine ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Activity className="w-5 h-5 text-purple-600" />
                </div>
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-medium text-gray-900">Blockchain Ledger</h3>
                <div className="flex items-center mt-1">
                  {React.createElement(
                    getStatusIcon(healthStatus.phase2_components.blockchain_ledger),
                    { className: `w-4 h-4 ${getStatusColor(healthStatus.phase2_components.blockchain_ledger)} mr-1` }
                  )}
                  <span className={`text-sm font-medium ${getStatusColor(healthStatus.phase2_components.blockchain_ledger)}`}>
                    {healthStatus.phase2_components.blockchain_ledger ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <Activity className="w-5 h-5 text-green-600" />
                </div>
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-medium text-gray-900">Digital Twin</h3>
                <div className="flex items-center mt-1">
                  {React.createElement(
                    getStatusIcon(healthStatus.phase3_components.digital_twin),
                    { className: `w-4 h-4 ${getStatusColor(healthStatus.phase3_components.digital_twin)} mr-1` }
                  )}
                  <span className={`text-sm font-medium ${getStatusColor(healthStatus.phase3_components.digital_twin)}`}>
                    {healthStatus.phase3_components.digital_twin ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-blue-600" />
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Active Trials</p>
              <p className="text-2xl font-bold text-gray-900">{kpis.activeTrials}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                <Users className="w-5 h-5 text-green-600" />
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Patients</p>
              <p className="text-2xl font-bold text-gray-900">{kpis.totalPatients.toLocaleString()}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-purple-600" />
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Completion Rate</p>
              <p className="text-2xl font-bold text-gray-900">{kpis.completionRate}%</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-yellow-600" />
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Delay Risk</p>
              <p className="text-2xl font-bold text-gray-900">{(kpis.avgDelayRisk * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-red-600" />
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Adverse Events</p>
              <p className="text-2xl font-bold text-gray-900">{kpis.adverseEvents}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center">
                <Clock className="w-5 h-5 text-indigo-600" />
              </div>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Query Resolution</p>
              <p className="text-2xl font-bold text-gray-900">{kpis.queryResolutionTime} days</p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Patient Enrollment Trend</h3>
          <Plot
            data={[enrollmentData]}
            layout={{
              width: 400,
              height: 300,
              margin: { l: 50, r: 50, t: 20, b: 50 },
              xaxis: { title: { text: 'Time Period' } },
              yaxis: { title: { text: 'Patients Enrolled' } },
              showlegend: false,
            }}
            config={{ displayModeBar: false }}
          />
        </div>

        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Risk Distribution</h3>
          <Plot
            data={[riskDistribution]}
            layout={{
              width: 400,
              height: 300,
              margin: { l: 50, r: 50, t: 20, b: 50 },
              showlegend: true,
              legend: { x: 0, y: 1 },
            }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
        <div className="space-y-3">
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            <span className="text-sm text-gray-600">
              <strong>TRIAL-2024-001:</strong> 15 new patients enrolled
            </span>
            <span className="text-xs text-gray-400">2 hours ago</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
            <span className="text-sm text-gray-600">
              <strong>NLP Engine:</strong> Processed 23 adverse event reports
            </span>
            <span className="text-xs text-gray-400">4 hours ago</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
            <span className="text-sm text-gray-600">
              <strong>Digital Twin:</strong> Scenario analysis completed for TRIAL-2024-003
            </span>
            <span className="text-xs text-gray-400">6 hours ago</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
            <span className="text-sm text-gray-600">
              <strong>Blockchain:</strong> 12 new audit events logged
            </span>
            <span className="text-xs text-gray-400">8 hours ago</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardOverview;
