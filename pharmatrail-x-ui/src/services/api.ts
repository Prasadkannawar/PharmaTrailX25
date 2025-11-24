import axios from 'axios';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8006';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Types
export interface TrialData {
  Age: number;
  Medication_Adherence: number;
  ADR_Rate: number;
  BP_Systolic: number;
  ALT_Level: number;
  trial_id?: string;
}

export interface PredictionResponse {
  delay_probability: number;
  delay_prediction: string;
  confidence: number;
  risk_factors: string[];
}

export interface NLPResponse {
  entities: Array<{
    text: string;
    label: string;
    start: number;
    end: number;
    confidence: number;
    severity?: string;
  }>;
  ae_events: Array<{
    event: string;
    severity: string;
    confidence: number;
  }>;
  severity_classification: string;
  confidence_score: number;
  processing_timestamp: string;
}

export interface SiteConfig {
  site_id: string;
  name: string;
  region: string;
  capacity: number;
  staff_count: number;
  staff_skill_level: number;
  enrollment_rate: number;
  query_resolution_time: number;
  data_entry_delay: number;
}

export interface TrialConfig {
  trial_id: string;
  target_enrollment: number;
  planned_duration_weeks: number;
  sites: SiteConfig[];
  visit_schedule?: number[];
  dropout_base_rate?: number;
  query_rate_per_visit?: number;
  data_entry_sla?: number;
}

export interface SimulationResponse {
  trial_id: string;
  simulation_days: number;
  completion_date?: string;
  total_enrolled: number;
  total_completed: number;
  total_dropouts: number;
  active_patients: number;
  total_queries: number;
  resolved_queries: number;
  pending_queries: number;
  avg_data_entry_delay: number;
  delay_risk_score: number;
  cost_estimate: number;
  timeline_data: Array<{
    day: number;
    timestamp: number;
    enrolled_patients: number;
    active_patients: number;
    completed_patients: number;
    dropout_patients: number;
    open_queries: number;
    resolved_queries: number;
    avg_data_delay: number;
  }>;
  site_metrics: Record<string, {
    enrolled_patients: number;
    completed_patients: number;
    dropout_patients: number;
    active_queries: number;
    resolved_queries: number;
    enrollment_rate: number;
    completion_rate: number;
    avg_data_delay: number;
  }>;
  scenario_id?: string;
}

export interface BlockchainEvent {
  index: number;
  timestamp: string;
  data: {
    event_type: string;
    trial_id?: string;
    [key: string]: any;
  };
  hash: string;
  previous_hash: string;
}

// API Service Functions
export const apiService = {
  // Health Check
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  // Analytics APIs
  async predictDelay(trialData: TrialData): Promise<PredictionResponse> {
    const response = await api.post('/analytics/predict', trialData);
    return response.data;
  },

  async getModelInfo() {
    const response = await api.get('/analytics/model_info');
    return response.data;
  },

  async trainModel(trainingData: any) {
    const response = await api.post('/analytics/train', trainingData);
    return response.data;
  },

  // NLP APIs
  async extractAdverseEvents(text: string, trialId: string): Promise<NLPResponse> {
    const response = await api.post('/nlp/ae', {
      text,
      trial_id: trialId,
    });
    return response.data;
  },

  async summarizeText(text: string, maxLength: number = 150) {
    const response = await api.post('/nlp/summary', {
      text,
      max_length: maxLength,
    });
    return response.data;
  },

  // Digital Twin APIs
  async runSimulation(trialConfig: TrialConfig, simulationDays: number = 365): Promise<SimulationResponse> {
    const response = await api.post('/twin/simulate', {
      trial_config: trialConfig,
      simulation_days: simulationDays,
    });
    return response.data;
  },

  async runScenarioAnalysis(scenarioRequest: any) {
    const response = await api.post('/twin/scenario', scenarioRequest);
    return response.data;
  },

  async getScenarioRecommendations(trialConfig: TrialConfig, simulationDays: number = 365) {
    const response = await api.post('/twin/recommendations', trialConfig, {
      params: { simulation_days: simulationDays }
    });
    return response.data;
  },

  async getTwinInfo() {
    const response = await api.get('/twin/info');
    return response.data;
  },

  // Blockchain APIs
  async getBlockchainChain(limit?: number): Promise<{ blocks: BlockchainEvent[]; stats: any }> {
    const response = await api.get('/blockchain/get_chain', {
      params: limit ? { limit } : {},
    });
    return response.data;
  },

  async logBlockchainEvent(eventType: string, eventData: any) {
    const response = await api.post('/blockchain/log_event', {
      event_type: eventType,
      event_data: eventData,
    });
    return response.data;
  },

  // Integrated Prediction
  async getIntegratedPrediction(trialData: TrialData, clinicalText?: string) {
    const response = await api.post('/predict/integrated', {
      trial_data: trialData,
      clinical_text: clinicalText,
      include_nlp: !!clinicalText,
      include_blockchain_audit: true,
    });
    return response.data;
  },

  // File Upload (placeholder for CT.gov data ingestion)
  async uploadFile(file: File, endpoint: string = '/ingest/upload') {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default api;
