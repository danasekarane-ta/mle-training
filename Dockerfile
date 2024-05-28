FROM continuumio/miniconda3

COPY deploy/env.yml .
RUN conda env create -f env.yml
RUN echo "source activate mle-dev" > ~/.bashrc
ENV path /opt/conda/envs/myenv/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /mlflow-learning/
WORKDIR /mlflow-learning/
ENV MLFLOW_TRACKING_URI=http://localhost:5008
EXPOSE 5008

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y python3

RUN python3 -m build && \
    pip install dist/*.whl --force-reinstall

CMD ["sh", "-c", "\
    mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5008 & \
    python3 prediction.py
"]