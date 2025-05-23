# API Documentation for Sentiment Analysis Service

## 1. Overview

The "app" folder implements a scalable Sentiment Analysis API using FastAPI, integrated with MLflow for model management and Prometheus for monitoring. The API exposes endpoints for performing sentiment predictions on input text data, health checks, model reloading, and metric collection. The system architecture ensures modularity, fault tolerance, and observability, making it suitable for production-level deployment.

## 2. Architecture and Design

The dependencies.py module is responsible for:

- Loading the registered ML model from the MLflow Model Registry using an alias (champion).

- Exposing a singleton instance (ModelStore) for consistent access to the model across the application.

- Providing prediction (.predict()) and model reloading (.reload()) functionalities.

- Supporting dependency injection through the get_model_store() function.

The main.py file contains the core API logic and defines the lifecycle of the application:

- Startup and Shutdown Events are managed through lifespan, ensuring proper resource handling.

- Environment Configuration is loaded using dotenv, including variables such as MODEL_NAME and allowed CORS origins.


## 3. Endpoints

3.1 /api/v1/predict — POST: Receives a list of texts and returns predicted sentiment labels (0 = NEGATIVE, 1 = NEUTRAL, 2 = POSITIVE).

3.2 /reload-model — POST: Reloads the ML model from the MLflow Registry.

3.3 /health — GET: Returns the health status of the currently loaded model.

3.4 /metrics — GET: Exposes application metrics in Prometheus format.

Metrics Tracked:

- predict_requests_total: Total predictions made.

- predict_duration_seconds: Time taken per prediction.

- reload_model_requests_total: Total reload attempts.

- reload_model_duration_seconds: Reload latency.

- model_serving_info: Active model version.

- service_health_status: Application health.

- list_item_count: Total number of texts processed.


## 4. Deployment

The API can be launched with the following command:

```bash
uvicorn main:api --host 0.0.0.0 --port 8000
``` 

The project uses an environment file .env for all sensitive and configurable variables. Typical entries include:
```init
MODEL_NAME=your_mlflow_model_name
ALLOWED_ORIGINS=http://localhost,http://yourfrontend.com
```