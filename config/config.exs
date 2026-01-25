import Config

# Configure Nx to use EXLA backend by default
config :nx, default_backend: EXLA.Backend

# Configure EXLA for CUDA (RTX 4090)
config :exla,
  default_client: :cuda,
  clients: [
    cuda: [
      platform: :cuda,
      memory_fraction: 0.8,  # Use 80% of VRAM (19.2GB of 24GB)
      preallocate: false     # Don't preallocate, let it grow
    ],
    host: [
      platform: :host
    ]
  ]

# Set XLA flags for better performance
System.put_env("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda")
