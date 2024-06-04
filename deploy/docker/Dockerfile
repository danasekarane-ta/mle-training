FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

COPY . /app

RUN ls

RUN conda env create -f /app/mlflow-learning/deploy/conda/env.yml

# Activate the new environment
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

COPY . /app
WORKDIR /app/mlflow-learning

ENV MLFLOW_TRACKING_URI=http://localhost:5008
EXPOSE 5008


CMD [ "/bin/sh", "-c", "mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5008 \
    & python3 mlflow_prediction.py"]