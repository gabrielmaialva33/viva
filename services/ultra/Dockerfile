# ULTRA KG Reasoning Service
# Reference: https://github.com/DeepGraphLearning/ULTRA
#
# Build:
#   docker build -t viva-ultra .
#
# Run:
#   docker run -p 8765:8765 --gpus all viva-ultra

FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA
RUN pip3 install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyG dependencies
RUN pip3 install --no-cache-dir \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-geometric==2.4.0 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.5.0 \
    ninja \
    easydict \
    pyyaml \
    httpx>=0.26.0 \
    numpy>=1.24.0

# Download ULTRA checkpoint
RUN mkdir -p /app/ckpts && \
    python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='mgalkin/ultra_50g', filename='ultra_50g.pth', local_dir='/app/ckpts')" || \
    echo "Checkpoint download skipped - will run in mock mode"

# Copy application code
COPY . .

# Set environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV ULTRA_CHECKPOINT=/app/ckpts/ultra_50g.pth

EXPOSE 8765

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
