pip install openai transformers mlflow pandas numpy pydantic fastapi uvicorn matplotlib twikit

export AIRFLOW_VERSION=2.10.5
export CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-312.txt"
pip install "apache-airflow[postgres,azure]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

