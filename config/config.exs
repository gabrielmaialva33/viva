# This file is responsible for configuring your application
# and its dependencies with the aid of the Config module.
#
# This configuration file is loaded before any dependency and
# is restricted to this project.

# General application configuration
import Config

config :viva,
  ecto_repos: [Viva.Repo],
  generators: [timestamp_type: :utc_datetime, binary_id: true]

# NVIDIA NIM Configuration
config :viva, :nim,
  # API Configuration
  base_url: System.get_env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
  api_key: System.get_env("NIM_API_KEY"),
  timeout: 120_000,

  # Models - Maximum Quality Selection
  models: %{
    # Primary LLM - Best reasoning, tool calling, instruction following
    llm: "nvidia/llama-3.1-nemotron-ultra-253b-v1",

    # Embeddings - Multilingual, highest quality
    embedding: "nvidia/nv-embedqa-mistral-7b-v2",

    # Reranking - For RAG pipeline
    rerank: "nvidia/llama-3.2-nemoretriever-500m-rerank-v2",

    # Text-to-Speech - Multilingual (pt-BR support)
    tts: "nvidia/magpie-tts-multilingual",

    # Speech-to-Text - 25 languages, streaming
    asr: "nvidia/parakeet-1.1b-rnnt-multilingual-asr",

    # Vision-Language Model - Understands images/video
    vlm: "nvidia/cosmos-nemotron-34b",

    # Safety & Content Moderation
    safety: "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    safety_multimodal: "meta/llama-guard-4-12b",
    jailbreak_detect: "nvidia/nemoguard-jailbreak-detect",

    # === NEW: Extended Capabilities ===

    # Image Generation - Avatar profile pictures
    image_gen: "stabilityai/stable-diffusion-3.5-large",
    image_edit: "black-forest-labs/FLUX.1-Kontext-dev",

    # 3D Avatar - Model generation and lipsync
    avatar_3d: "microsoft/TRELLIS",
    audio2face: "nvidia/audio2face-3d",

    # Translation - Global matchmaking (36 languages)
    translate: "nvidia/riva-translate-1.6b",

    # Audio Enhancement - Premium voice quality
    audio_enhance: "nvidia/studiovoice",
    noise_removal: "nvidia/Background Noise Removal",

    # Advanced Reasoning - Complex autonomous decisions
    reasoning: "deepseek-ai/deepseek-r1-0528"
  }

# Oban background job configuration
config :viva, Oban,
  engine: Oban.Engines.Basic,
  queues: [
    default: 10,
    avatar_simulation: 20,
    matchmaking: 5,
    memory_processing: 10,
    conversations: 15
  ],
  repo: Viva.Repo

# Configure the endpoint
config :viva, VivaWeb.Endpoint,
  url: [host: "localhost"],
  adapter: Bandit.PhoenixAdapter,
  render_errors: [
    formats: [html: VivaWeb.ErrorHTML, json: VivaWeb.ErrorJSON],
    layout: false
  ],
  pubsub_server: Viva.PubSub,
  live_view: [signing_salt: "CZXYNH0E"]

# Configure esbuild (the version is required)
config :esbuild,
  version: "0.25.4",
  viva: [
    args:
      ~w(js/app.js --bundle --target=es2022 --outdir=../priv/static/assets/js --external:/fonts/* --external:/images/* --alias:@=.),
    cd: Path.expand("../assets", __DIR__),
    env: %{"NODE_PATH" => [Path.expand("../deps", __DIR__), Mix.Project.build_path()]}
  ]

# Configure tailwind (the version is required)
config :tailwind,
  version: "4.1.12",
  viva: [
    args: ~w(
      --input=assets/css/app.css
      --output=priv/static/assets/css/app.css
    ),
    cd: Path.expand("..", __DIR__)
  ]

# Configure Elixir's Logger
config :logger, :default_formatter,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id]

# Use Jason for JSON parsing in Phoenix
config :phoenix, :json_library, Jason

# Import environment specific config. This must remain at the bottom
# of this file so it overrides the configuration defined above.
import_config "#{config_env()}.exs"
