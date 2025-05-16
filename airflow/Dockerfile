# # FROM apache/airflow:2.5.0-python3.9

# # USER airflow

# # COPY ./requirements.txt /requirements.txt
# # RUN pip install --no-cache-dir -r /requirements.txt

# # RUN chown -R airflow: /opt/airflow

# FROM apache/airflow:2.10.5-python3.12

# USER root

# RUN apt-get update \
#   && apt-get install -y --no-install-recommends \
#   && apt-get autoremove -yqq --purge \
#   && apt-get clean

# COPY ./requirements.txt /requirements.txt

# USER airflow

# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir -r /requirements.txt
# RUN chown -R airflow: /opt/airflow

FROM apache/airflow:2.10.5-python3.12

USER root

# Cài đặt các gói cần thiết
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     wget \
     bzip2 \
     ca-certificates \
     curl \
     git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Cài đặt Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Thiết lập biến môi trường
ENV PATH /opt/conda/bin:$PATH

# Sao chép file conda.yaml vào container
COPY ./conda.yaml /conda.yaml

# Tạo môi trường từ conda.yaml
RUN conda env create -f /conda.yaml


USER airflow

# Kích hoạt môi trường conda khi khởi động
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $(head -1 /conda.yaml | cut -d' ' -f2)" >> ~/.bashrc

# Cài đặt các gói Python cần thiết
RUN pip install --no-cache-dir --upgrade pip
RUN chown -R airflow: /opt/airflow
