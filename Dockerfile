FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the env file
COPY deploy/env.yml /app

RUN conda env create -f /app/env.yml

# Activate the new environment
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

COPY . /app
WORKDIR /app/mlflow-learning/

ENV MLFLOW_TRACKING_URI=http://localhost:5008
EXPOSE 5008

CMD python mlflow_prediction.py && mlflow server --host 0.0.0.0