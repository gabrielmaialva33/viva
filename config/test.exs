import Config

# Configure your database
#
# The MIX_TEST_PARTITION environment variable can be used
# to provide built-in test partitioning in CI environment.
# Run `mix help test` for more information.
config :viva, Viva.Repo,
  username: "postgres",
  password: "postgres",
  hostname: "localhost",
  database: "viva_test#{System.get_env("MIX_TEST_PARTITION")}",
  pool: Ecto.Adapters.SQL.Sandbox,
  pool_size: System.schedulers_online() * 2,
  types: Viva.Infrastructure.PostgrexTypes

# We don't run a server during test. If one is required,
# you can enable the server option below.
config :viva, VivaWeb.Endpoint,
  http: [ip: {127, 0, 0, 1}, port: 4002],
  secret_key_base: "uGw3Iw/eBi8qLqDtCYtXyyNCHA87wcGJHXFc8cpJ5HUszEVW9Mp9DnlkpglCh4iB",
  server: false

# Print only warnings and errors during test
config :logger, level: :warning

# Initialize plugs at runtime for faster test compilation
config :phoenix, :plug_init_mode, :runtime

# Enable helpful, but potentially expensive runtime checks
config :phoenix_live_view,
  enable_expensive_runtime_checks: true

# Disable Oban queues and plugins in test to avoid DB connection errors
config :viva, Oban, queues: false, plugins: false

# Disable starting active avatars automatically in tests
config :viva, start_active_avatars: false

# Use Mock LLM Client in tests
config :viva, :llm_client, Viva.AI.LLM.MockClient

# Use Mock EventBus in tests
config :viva, :event_bus, Viva.Infrastructure.MockEventBus
