# End-to-End Fraud Detection MLOps System

Production-ready machine learning system for real-time credit card fraud detection with full MLOps pipeline.


![CI](https://github.com/Roy16Keane/End-to-End-Fraud-detection-system/actions/workflows/ci.yml/badge.svg)

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)

![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![GitHub%20Actions](https://img.shields.io/badge/GitHub%20Actions-CI/CD-2088FF?logo=githubactions)

![Architecture](https://img.shields.io/badge/Architecture-REST%20Microservice-lightgrey)
![MLOps](https://img.shields.io/badge/MLOps-End--to--End-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)



## Key Features
- Real-time fraud prediction API
- Interactive Streamlit dashboard
- Reproducible training with DVC
- Experiment tracking with MLflow
- Automated testing & Docker builds via CI/CD

## Business Problem

Financial fraud causes billions in losses annually.  
This project builds a real-time fraud detection system that:

- Predicts probability of fraud for each transaction
- Supports real-time decision making
- Provides a scalable and reproducible ML pipeline


## System Architecture

![Architecture](docs/architecture.png)

## Project Structure
.



## Model Performance

Forward-time validation (monthly split):

- Mean AUC: **0.94**
- Std: 0.004
- Minimum: 0.935
- Maximum: 0.947

## MLOps Capabilities

- **Data Versioning**: DVC tracks datasets and pipeline outputs
- **Experiment Tracking**: MLflow logs parameters, metrics, and artifacts
- **Reproducible Pipeline**: `dvc repro` rebuilds the entire workflow
- **API Serving**: FastAPI for real-time inference
- **UI Layer**: Streamlit dashboard for user interaction
- **Containerization**: Docker for API and UI
- **Orchestration**: Docker Compose runs services together
- **CI/CD**:
  - Pytest runs on every push
  - Docker images built automatically via GitHub Actions
## Run Locally

### Clone repository
from root repo docker compose --build 

Access:
- API: http://localhost:8000/docs  
- Streamlit: http://localhost:8501
## API Example

POST /predict

Request:
{
  "transaction": {
    "TransactionDT": 100000,
    "TransactionAmt": 49.99,
    "ProductCD": "W",
    "card1": 1234,
    "addr1": 200,
    "P_emaildomain": "gmail.com"
  },
  "threshold": 0.5
}

{
  "fraud_proba": 0.05,
  "fraud_label": 0,
  "threshold": 0.5
}

## Future Improvements

- Deploy API to AWS / Render
- Add monitoring (Evidently / Prometheus)
- Implement model versioning & rollback
- Add batch inference pipeline
## Author

Roy Keane Syangu  
MSc Robotics & AI | Machine Learning & MLOps Engineer  

