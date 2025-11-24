# PharmaTrail-X: Clinical Trial Intelligence Platform

This repo delivers an end-to-end clinical-operations intelligence stack across four execution phases: data/AI foundation, clinical NLP + blockchain audit, SimPy-based Digital Twin, and a React UI with cloud-ready deployment.

## Phase Overview (1–4)
- **Phase 1 — Data & AI Foundation:** Multi-source ingestion (ClinicalTrials.gov CSV + mocked FHIR/HL7), raw/processed data lake (PostgreSQL + Parquet under `data/raw` and `data/processed`), feature engineering for timeline gaps, site deviation, query density, enrollment velocity, protocol impacts, and data-entry lag. Trains XGBoost delay predictor (probability 0–1) plus Isolation Forest for anomalies; experiments tracked in MLflow. Exposes FastAPI endpoints `/analytics/train`, `/analytics/predict`, `/analytics/model_info`.
- **Phase 2 — NLP Engine + Blockchain Audit:** Clinical NLP (AE extraction, lab/protocol deviation detection, summarization) exposed via `/nlp/ae` and `/nlp/summary`. Blockchain audit ledger for tamper-proof events with `/blockchain/log_event` and `/blockchain/get_chain`, hashing every prediction/ingestion/NLP or simulation event.
- **Phase 3 — Digital Twin Simulation:** SimPy engine modeling enrollment, query cycles, data-entry delay, dropout, staffing effects. Scenario modeling to test operational levers (add sites, staff multipliers, query-time and data-delay reductions) with FastAPI endpoints `/twin/simulate` and `/twin/scenario`, returning timeline trajectories and summary metrics.
- **Phase 4 — UI + Deployment:** React single-page dashboard connecting to all APIs (/analytics/*, /nlp/*, /twin/*, /blockchain/*, ingestion). SaaS-style console with analytics, NLP, Digital Twin, blockchain viewer, data ingestion, and model management panels. Containerized (Docker) for FastAPI + frontend and deployable to cloud targets.

## Phase 1 — Detailed
In Phase 1 we build the core intelligence layer: ingest ClinicalTrials.gov CSVs and mocked FHIR/HL7 feeds into a unified data lake (`data/raw`, `data/processed`, PostgreSQL for structured storage). Preprocessing normalizes numerics/dates and engineers time-based features (delay score, timeline gaps, site deviation index, query-to-patient density, enrollment velocity, protocol amendment impact, data-entry lag index). The ML layer trains an XGBoost classifier for delay probability and an Isolation Forest for anomaly detection; all runs, hyperparameters, and metrics (AUC, accuracy, feature importance) are tracked in MLflow. FastAPI endpoints `/analytics/train`, `/analytics/predict`, and `/analytics/model_info` expose training, scoring, and model inspection. Deliverables: clean data lake, validated delay-prediction model, MLflow tracking UI, and an end-to-end pipeline from ingestion to prediction that later phases reuse.

## Phase 2 — NLP + Blockchain
Phase 2 layers clinical NLP and blockchain audit on top of the Phase 1 foundation. The NLP engine (rule-based here, BioBERT/ClinicalBERT-ready) extracts adverse events, lab abnormalities, protocol deviations, and summarizations via FastAPI endpoints `/nlp/ae` and `/nlp/summary`, returning entities, severity, AE categories, timestamps, and confidence. The blockchain ledger provides immutable audit logging (not cryptocurrency): every prediction, ingestion, NLP extraction, or simulation result is hashed and appended with `{trial_id, timestamp, event_type, payload_hash, previous_hash}`. APIs: `/blockchain/log_event`, `/blockchain/get_chain`. Deliverables: operational NLP service, AE extraction and summarization, integrated ledger, and immutable audit logs.

## Phase 3 — Digital Twin Simulation
Phase 3 introduces a SimPy-based Digital Twin that mirrors trial operations. It simulates patient enrollment, site-level query resolution, data-entry delays, dropout behavior, and staffing effects over configurable horizons (30–365 days). Scenario modeling enables “what-if” changes (add sites, increase staff, reduce query cycle time, improve data delays) and returns completion estimates, risk scores, and cost/time deltas. FastAPI endpoints `/twin/simulate` (baseline) and `/twin/scenario` (scenario analysis) output dashboard-friendly JSON (timelines, site metrics, comparison data). Deliverables: modular SimPy engine, scenario API, parameter schema, and outputs ready for visualization.

## Phase 4 — UI + Deployment
Phase 4 delivers the production UI and deployment: a React single-page app (see `pharmatrail-x-ui`) with tabs/cards for Analytics, NLP, Digital Twin, Blockchain, Data Ingestion, and Model Management. Visualizations (gauges, lines, bars, timelines/heatmaps) consume backend JSON in real time with proper loading/error states. The stack (FastAPI + simulation services + React) is containerized via Docker and deployable to cloud targets (e.g., ECS/Azure/Render/ngrok for demos). Deliverables: integrated SaaS-style UI, running backend with persistence, deployment endpoint, and a cohesive AI+NLP+Blockchain+Twin experience.

## Quick Start
### Prerequisites
- Python 3.8+
- PostgreSQL (optional, for DB storage)
- MLflow (for experiment tracking)

### Backend setup
```bash
pip install -r requirements.txt
# optional .env
DATABASE_URL=postgresql://user:pass@localhost:5432/pharmax_db
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### Train pipeline (Phase 1)
```bash
python train_pipeline.py --hyperparameter-tuning --generate-fhir
```

### Run analytics API (Phase 1/2)
```bash
cd src/api
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run integrated API with Digital Twin (Phase 3)
```bash
python phase3_integrated_api.py  # serves /nlp/*, /blockchain/*, /twin/*, /analytics/*
```

### Frontend (Phase 4)
```bash
cd pharmatrail-x-ui
npm install
npm start   # expects backend on port 8006 (configurable in services/api)
```

## Key Endpoints
- Analytics: `/analytics/train`, `/analytics/predict`, `/analytics/model_info`, `/analytics/batch_predict`
- NLP: `/nlp/ae`, `/nlp/summary`
- Blockchain: `/blockchain/log_event`, `/blockchain/get_chain`
- Digital Twin: `/twin/simulate`, `/twin/scenario`, `/twin/recommendations`, `/twin/info`
- Health: `/health` (integrated API root in Phase 3), `/` root summaries

## Data & Models
- Source CSV: `PharmaTrailX_ClinicalTrialMaster2.csv`
- Processed parquet: `data/processed/clinical_trials_processed.parquet`
- Model registry: `models/` (XGBoost delay predictor, Isolation Forest, preprocessors, feature metadata)
- Blockchain data: `blockchain_data/` (JSON chain snapshots)

## Project Structure (high level)
```
data/
  raw/ | processed/
src/
  ingestion/        # data_ingester.py
  preprocessing/    # feature_engineer.py
  models/           # ml_pipeline.py, database.py
  api/              # main.py (Phase1/2), digital_twin_service.py
  nlp/              # nlp_engine.py
  blockchain/       # ledger.py
  simulation/       # digital_twin_engine.py, scenario_engine.py
pharmatrail-x-ui/   # React SPA for Phase 4
phase3_integrated_api.py
train_pipeline.py
```

## Testing
- Phase 3 digital twin tests: `test_phase3_digital_twin.py` (engine, scenario, service, API smoke)
- Additional API tests: `test_api_endpoints.py`, `test_phase2_real.py`, etc.

## Support
- Logs: `logs/pharmax.log`
- MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (http://localhost:5000)
- Blockchain integrity: `/blockchain/get_chain`
