import Config

# Test configuration - use in-memory backend when NIFs are skipped
config :viva_core,
  memory_backend: :in_memory

# Reduce log noise during tests
config :logger, level: :warning

# Disable SporeLogger in tests (only use console)
config :logger, backends: [:console]
