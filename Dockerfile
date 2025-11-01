# Use NVIDIA PyTorch base image with CUDA 12
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    wget \
    build-essential \
    pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install Python packages
RUN pip install --upgrade pip && \
    pip install opencv-python-headless ultralytics pyyaml numpy

# Install the yolopose vertical mapper package in development mode
COPY . /workspace/yoloplay
WORKDIR /workspace/yoloplay
RUN pip install -e .

# Create a directory for mounting data
RUN mkdir -p /workspace/data

# Set the entrypoint
WORKDIR /workspace
#ENTRYPOINT ["/opt/conda/bin/python", "-m", "yolopose_vertical_mapper.main"]