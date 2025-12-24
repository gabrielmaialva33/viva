import Config

# config/runtime.exs is executed for all environments, including
# during releases. It is executed after compilation and before the
# system starts, so it is typically used to load production configuration
# and secrets from environment variables or elsewhere. Do not define
# any compile-time configuration in here, as it won't be applied.
# The block below contains prod specific runtime configuration.

# Load .env file automatically in dev/test environments
# In production, we only use System.get_env()
env =
  if config_env() in [:dev, :test] do
    Dotenvy.source!([".env", System.get_env()])
  else
    System.get_env()
  end

# Helper to get environment variables
env_get = fn key, default -> Map.get(env, key, default) end

# ## Using releases
#
# If you use `mix release`, you need to explicitly enable the server
# by passing the PHX_SERVER=true when you start it:
#
#     PHX_SERVER=true bin/viva start
#
# Alternatively, you can use `mix phx.gen.release` to generate a `bin/server`
# script that automatically sets the env var above.
if env_get.("PHX_SERVER", nil) do
  config :viva, VivaWeb.Endpoint, server: true
end

port_str = env_get.("PORT", "4000")
port = String.to_integer(port_str)

config :viva, VivaWeb.Endpoint, http: [port: port]

# NVIDIA NIM configuration (all environments)
nim_base_url = env_get.("NIM_BASE_URL", nil)
nim_api_key = env_get.("NIM_API_KEY", nil)
nim_timeout_str = env_get.("NIM_TIMEOUT", "60000")
nim_timeout = String.to_integer(nim_timeout_str)

if nim_base_url && nim_api_key do
  config :viva, :nim,
    base_url: nim_base_url,
    api_key: nim_api_key,
    image_base_url: env_get.("NIM_IMAGE_BASE_URL", "https://ai.api.nvidia.com/v1/genai"),
    timeout: nim_timeout,
    models: %{
      llm: env_get.("NIM_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"),
      embedding: env_get.("NIM_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    }
end

if config_env() == :prod do
  database_url =
    env_get.("DATABASE_URL", nil) ||
      raise """
      environment variable DATABASE_URL is missing.
      For example: ecto://USER:PASS@HOST/DATABASE
      """

  maybe_ipv6 = if env_get.("ECTO_IPV6", nil) in ~w(true 1), do: [:inet6], else: []

  pool_size_str = env_get.("POOL_SIZE", "10")
  pool_size = String.to_integer(pool_size_str)

  config :viva, Viva.Repo,
    url: database_url,
    pool_size: pool_size,
    types: Viva.PostgrexTypes,
    socket_options: maybe_ipv6

  # Redis configuration
  config :viva, :redis, url: env_get.("REDIS_URL", "redis://localhost:6379")

  # Qdrant vector database
  config :viva, :qdrant,
    url: env_get.("QDRANT_URL", "http://localhost:6333"),
    api_key: env_get.("QDRANT_API_KEY", nil)

  # NVIDIA NIM configuration (prod overrides to require key)
  nim_timeout_prod_str = env_get.("NIM_TIMEOUT", "60000")
  nim_timeout_prod = String.to_integer(nim_timeout_prod_str)

  config :viva, :nim,
    base_url: env_get.("NIM_BASE_URL", nil) || raise("NIM_BASE_URL is required"),
    api_key: env_get.("NIM_API_KEY", nil),
    image_base_url: env_get.("NIM_IMAGE_BASE_URL", "https://ai.api.nvidia.com/v1/genai"),
    timeout: nim_timeout_prod,
    models: %{
      llm: env_get.("NIM_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"),
      embedding: env_get.("NIM_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    }

  # Oban configuration for production
  avatar_queue_size_str = env_get.("OBAN_AVATAR_QUEUE_SIZE", "20")
  avatar_queue_size = String.to_integer(avatar_queue_size_str)

  config :viva, Oban,
    engine: Oban.Engines.Basic,
    queues: [
      default: 10,
      avatar_simulation: avatar_queue_size,
      matchmaking: 5,
      memory_processing: 10,
      conversations: 15
    ],
    repo: Viva.Repo

  # The secret key base is used to sign/encrypt cookies and other secrets.
  secret_key_base =
    env_get.("SECRET_KEY_BASE", nil) ||
      raise """
      environment variable SECRET_KEY_BASE is missing.
      You can generate one by calling: mix phx.gen.secret
      """

  host = env_get.("PHX_HOST", "example.com")

  config :viva, :dns_cluster_query, env_get.("DNS_CLUSTER_QUERY", nil)

  config :viva, VivaWeb.Endpoint,
    url: [host: host, port: 443, scheme: "https"],
    http: [
      ip: {0, 0, 0, 0, 0, 0, 0, 0}
    ],
    secret_key_base: secret_key_base
end
