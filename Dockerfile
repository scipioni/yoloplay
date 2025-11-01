# Build stage
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

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
    libgl1-mesa-glx \
    libglib2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install Python packages
RUN pip install --upgrade pip && \
    pip install opencv-python-headless ultralytics pyyaml numpy

# Install the yoloplay package in development mode
COPY . /workspace/yoloplay
WORKDIR /workspace/yoloplay
RUN pip install -e .

# Runtime stage
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda

# Copy Python environment from builder
COPY --from=builder /opt/conda /opt/conda

# Copy the installed yoloplay package
COPY --from=builder /workspace/yoloplay /workspace/yoloplay

# Set working directory
WORKDIR /workspace/yoloplay

# Create a directory for mounting data
RUN mkdir -p /workspace/data

# Install GUI libraries in the runtime stage to support OpenCV GUI functions when needed
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install opencv-python-headless in the runtime stage (this should override if any full opencv was copied)
RUN pip install opencv-python-headless

