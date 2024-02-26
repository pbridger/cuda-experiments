FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
    pkg-config \
    g++-11 \
    ccache

RUN python -m pip install --upgrade pip && \
    python -m pip install \
    ninja \   
    wurlitzer

WORKDIR /app

