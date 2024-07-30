FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
# FROM rapidsai/rapidsai:23.08a-cuda11.8.0-py3.10

EXPOSE 5000

RUN apt-get update
RUN apt-get install -y git python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install torch_geometric networkx
RUN pip3 install numpy
RUN pip3 install hydra-optuna-sweeper mlflow
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
# RUN pip3 install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.6.* cugraph-cu11==24.6.*

WORKDIR /root/prog