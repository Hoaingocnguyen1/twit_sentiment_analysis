# FROM apache/airflow:2.5.0-python3.9

# USER airflow

# COPY ./requirements.txt /requirements.txt
# RUN pip install --no-cache-dir -r /requirements.txt

# RUN chown -R airflow: /opt/airflow

FROM apache/airflow:2.10.5-python3.12

USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  && apt-get autoremove -yqq --purge \
  && apt-get clean

COPY ./requirements.txt /requirements.txt

USER airflow

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 
RUN pip install transformers==4.51.3
RUN chown -R airflow: /opt/airflow
