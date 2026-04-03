# Bike Rental ML End-to-End Project

## Project Overview

Bike Rental Prediction is an end-to-end machine learning pipeline for predicting hourly bike rentals using XGBoost. The project demonstrates ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization, AWS cloud deployment, and basic testing.

The system includes both a REST API for model inference and a Streamlit dashboard for interactive predictions.

## Architecture

The codebase is organized into pipelines following this flow:

# Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Serve → Dashboard

### Core Modules
- **`rc/feature_pipeline/`**s: Data loading, preprocessing, and feature engineering
- **`load.py`**: Load historical bike rental and weather data
- **`preprocess.py`**: Handle missing values, normalize features, and outlier removal
- **`feature_engineering.py`**: Create time-based features (hour, weekday, month) and encode categorical variables
- **`src/training_pipeline`**/: Model training and hyperparameter optimization
- **`train.py`**: Baseline XGBoost training with configurable parameters
- **`tune.py`**: Optuna-based hyperparameter tuning with MLflow integration
- **`eval.py`**: Model evaluation and metrics calculation (RMSE, R²)
- **`src/inference_pipeline/`**: Production inference
- **`inference.py`**: Applies preprocessing and feature engineering using trained encoders
- **`src/api/`**: FastAPI web service
- **`main.py`**: REST API exposing /predict and /health endpoints

- **``**
### Web Applications

- **`app.py`**: Streamlit dashboard for interactive bike rental predictions
- Real-time predictions via FastAPI API
- Input features: season, year, month, hour, holiday, weather, temperature, humidity, wind, user type
- Displays predicted rentals dynamically

### Cloud Infrastructure & Deployment

## AWS S3: Stores data and trained models
- **Amazon ECR**: Container registry for API and Streamlit Docker images
- **Amazon ECS (Fargate)**: Container orchestration
- **Application Load Balancer (ALB)**: Routes traffic to API and Streamlit dashboard
- **Service Connect / Cloud Map**: Internal service discovery for API and dashboard communication

#### ECS Services:
- **bike-api-service**: FastAPI backend (port 8000)
- **bike-streamlit-service**: Streamlit dashboard (port 8501)

## Common Commands

### Environment Setup

# Install Python dependencies
```bash
pip install -r requirements.txt
Data Pipeline
# 1. Load raw data
python src/feature_pipeline/load.py

# 2. Preprocess
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
Training Pipeline
# Train baseline model
python src/training_pipeline/train.py

# Hyperparameter tuning
python src/training_pipeline/tune.py

# Evaluate model
python src/training_pipeline/eval.py
Inference
# Single inference
python src/inference_pipeline/inference.py --input data/raw/sample.csv --output predictions.csv
API Service
# Start FastAPI locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Streamlit Dashboard
# Run Streamlit locally
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Docker
# Build API container
docker build -t bike-api -f Dockerfile.api .

# Build Streamlit container
docker build -t bike-streamlit -f Dockerfile.streamlit .

# Run API container
docker run -p 8000:8000 bike-api

# Run Streamlit container
docker run -p 8501:8501 bike-streamlit

# Key Design Patterns

Pipeline Modularity: Each pipeline component can run independently.
Cloud-Native Architecture: Dockerized services with optional auto-scaling via ECS Fargate.
Service Discovery: Streamlit communicates with API using Service Connect / Cloud Map DNS.
Encoder Persistence: Encoders saved and loaded for consistent inference.
Environment-based Config: API_URL and other endpoints configurable via environment variables.

# File Structure Notes

data/: Raw and processed datasets
models/: Trained models and saved encoders
src/: Pipeline modules, API, and batch scripts
app.py: Streamlit dashboard
tests/: Unit and integration tests

# Production URLs (if deployed)

API: http://bike-app-alb-765107221.eu-west-2.elb.amazonaws.com
Streamlit Dashboard: http://bike-app-alb-765107221.eu-west-2.elb.amazonaws.com/dashboard

# Dependencies

ML/Data: xgboost, pandas, numpy, scikit-learn
API: fastapi, uvicorn
Dashboard: streamlit, plotly
Cloud: boto3 (AWS integration)
Experimentation: mlflow, optuna
```