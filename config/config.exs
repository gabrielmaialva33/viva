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
  # HYBRID: episodic → Rust HNSW (~1ms), semantic/emotional → Qdrant (persistent)
  memory_backend: :hybrid,
  native_memory_path: Path.expand("~/.viva/memory")

# i18n Configuration
# Set VIVA_LOCALE environment variable to change language
# Supported: en (default), pt_BR, zh_CN
config :viva, :locale, System.get_env("VIVA_LOCALE", "en")

# Gettext configuration
config :viva_core, Viva.Gettext,
  default_locale: "en",
  locales: ~w(en pt_BR zh_CN)

# Configure Logger (SporeLogger deprecated, using console only)
# Note: The :backends key is deprecated in OTP 28+
# config :logger, :default_handler, level: :info

# Import environment-specific config
import_config "#{config_env()}.exs"
