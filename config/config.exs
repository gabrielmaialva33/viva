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
  base_url: System.get_env("NIM_BASE_URL", "http://localhost:8000"),
  api_key: System.get_env("NIM_API_KEY"),
  models: %{
    llm: System.get_env("NIM_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"),
    embedding: System.get_env("NIM_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
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
