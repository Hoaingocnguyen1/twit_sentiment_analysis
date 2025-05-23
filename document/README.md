# Project Documentation

## Overview
This project implements a complete end-to-end sentiment analysis pipeline for Twitter data. It integrates data ingestion, preprocessing, model training, and real-time inference, all managed through Airflow for orchestration and MLflow for model tracking and deployment. The system is containerized using Docker and leverages CI/CD via GitHub Actions for automated testing and deployment.

The architecture follows modular design principles, separating data engineering and machine learning workflows while maintaining reproducibility and scalability.

## Project objectives
The main objective of this project is to build a fully automated, scalable pipeline that can:

- Crawl Twitter data using custom operators in Apache Airflow.

- Preprocess and clean text data in a structured and reusable format.

- Train and validate deep learning models for sentiment classification.

- Log experiments and manage model versions using MLflow.

- Serve the best model via a FastAPI app in production.

- Automate the entire workflow using CI/CD pipelines to ensure consistency from development to deployment.

This system can be adapted to other NLP tasks and serves as a robust framework for applied MLOps in real-world projects.
## Project workflow
![alt text](<image.png>)

##  Repository Structure
The repository is organized into modular components to separate concerns and improve maintainability:

- The .github/workflows/ directory contains GitHub Actions workflows for CI/CD, including test automation and deployment triggers.

- The airflow/ directory includes all Airflow configurations, with DAGs for data extraction, training, and retention, as well as custom plugins like sensors and operators.

- The mlflow/ directory handles MLflow-related components, including model training scripts, model registration logic, and a FastAPI app for model serving.

- The src/ directory contains the core logic for data processing and model training routines.

- The config/ directory stores project constants and utility scripts for handling data clients or external services.

- The notebooks/ directory includes Jupyter notebooks used create cookies file

- The tests/ folder contains unit and integration tests.

- The root directory holds the .env file (for environment variables), docker-compose.yml (to spin up the environment), and the main README.md.

This modular structure ensures that each component is independently manageable and testable, while still functioning cohesively in the overall pipeline.


## How to Run & Deploy the Project

This project uses GitHub Actions for automatic deployment and Docker Compose to run all services locally. Below are the steps to get started.

1. Clone the repository
```bash 
git clone https://github.com/Hoaingocnguyen1/twitter-sentiment-mlops.git
cd twitter-sentiment-mlops
```
2. Set up environment variables (for local development only)
Create a .env file in the project root directory. It should contain sensitive keys and service URLs. 

3. Configure GitHub Secrets

Navigate to:

GitHub Repository → Settings → Secrets and Variables → Actions

Add all required secrets. These are injected automatically into the GitHub Actions workflow.

4. CI/CD Pipeline
Once secrets are configured, push your changes to the main branch:
```bash
git add .
git commit -m "Update pipeline"
git push origin main
```
GitHub Actions will automatically:

- Set up the environment

- Run unit tests and code checks

- Build Docker containers 

- Deploy the application to the configured environment 
You can track the progress under the Actions tab in your GitHub repository.