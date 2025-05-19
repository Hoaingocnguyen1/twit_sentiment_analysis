FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy and create Conda environment
COPY conda.yaml ./
RUN conda env create -f conda.yaml \
    && conda clean -afy

# Add conda environment to PATH and create convenience activation script
RUN echo "source activate twit-sentiment-group-2" > ~/.bashrc && \
    echo "conda activate twit-sentiment-group-2" >> ~/.bashrc

# Set PATH to include the conda environment by default
ENV PATH /opt/conda/envs/twit-sentiment-group-2/bin:$PATH

# Expose MLflow server port
EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]


# FROM continuumio/miniconda3
# FROM python:3.12-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt ./

# # Cài riêng các gói “nặng”
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir \
#       --default-timeout=120 \
#       --retries=5 \
#       torch>=2.2.0 \
#       scikit-learn>=1.4.0

# # Cài phần còn lại trong requirements.txt (đã **loại bỏ** torch, torchvision, scikit-learn)
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 5000

# CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]