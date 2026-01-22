# ULTRA KG Reasoning Service
# Reference: https://github.com/DeepGraphLearning/ULTRA
#
# Build:
#   docker build -t viva-ultra .
#
# Run (stdio mode for Elixir Port):
#   docker run -i --gpus all viva-ultra
#
# The service communicates via stdin/stdout JSON protocol.

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyG (torch-geometric) dependencies
# Using pre-built wheels for torch 2.2.0 + CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Install ULTRA dependencies
# numpy<2 required for torch 2.2.0 compatibility
RUN pip3 install --no-cache-dir \
    ninja \
    easydict \
    pyyaml \
    "numpy>=1.24.0,<2" \
    huggingface_hub>=0.20.0

# Download ULTRA checkpoint from HuggingFace
# Files: model.safetensors (684KB), config.json, modeling.py, ultra/ folder
RUN mkdir -p /app/ckpts && \
    python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='mgalkin/ultra_50g', filename='model.safetensors', local_dir='/app/ckpts'); \
    hf_hub_download(repo_id='mgalkin/ultra_50g', filename='config.json', local_dir='/app/ckpts')" || \
    echo 'Checkpoint download skipped - will run in mock mode'

# Copy application code
COPY . .

# Set environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV ULTRA_CHECKPOINT=/app/ckpts/ultra_50g.pth
ENV PYTHONUNBUFFERED=1

# Stdio service - no port needed
CMD ["python3", "-u", "ultra_service.py"]
