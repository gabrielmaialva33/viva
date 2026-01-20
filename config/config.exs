# This file is responsible for configuring your umbrella
# and **all applications** and their dependencies with the
# help of the Config module.
#
# Note that all applications in your umbrella share the
# same configuration and dependencies, which is why they
# all use the same configuration file. If you want different
# configurations or dependencies per app, it is best to
# move said applications out of the umbrella.
import Config

# Sample configuration:
#
#     config :logger, :default_handler,
#       level: :info
#
#     config :logger, :default_formatter,
#       format: "$date $time [$level] $metadata$message\n",
#       metadata: [:user_id]
#
config :viva_core,
  memory_backend: :rust_native,
  native_memory_path: Path.expand("~/.viva/memory")

# Configure Logger to use Console and SporeLogger (Mycelial Memory)
config :logger,
  backends: [:console, VivaCore.Logging.SporeLogger]

# Import environment-specific config
import_config "#{config_env()}.exs"
