FROM python:3.10.14-slim
EXPOSE 5000
RUN apt-get autoremove
RUN apt-get clean
RUN apt-get update
RUN apt-get install -y git python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install mlflow
CMD mlflow ui --host 0.0.0.0