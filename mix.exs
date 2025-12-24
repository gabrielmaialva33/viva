defmodule Viva.MixProject do
  use Mix.Project

  def project do
    [
      app: :viva,
      version: "0.1.0",
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps(),
      compilers: [:phoenix_live_view] ++ Mix.compilers(),
      listeners: [Phoenix.CodeReloader]
    ]
  end

  # Configuration for the OTP application.
  #
  # Type `mix help compile.app` for more information.
  def application do
    [
      mod: {Viva.Application, []},
      extra_applications: [:logger, :runtime_tools]
    ]
  end

  def cli do
    [
      preferred_envs: [precommit: :test]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Specifies your project dependencies.
  #
  # Type `mix help deps` for examples and options.
  defp deps do
    [
      # Phoenix Core
      {:phoenix, "~> 1.8.2"},
      {:phoenix_ecto, "~> 4.5"},
      {:ecto_sql, "~> 3.13"},
      {:postgrex, ">= 0.0.0"},
      {:phoenix_html, "~> 4.1"},
      {:phoenix_live_reload, "~> 1.2", only: :dev},
      {:phoenix_live_view, "~> 1.1.0"},
      {:lazy_html, ">= 0.1.0", only: :test},
      {:esbuild, "~> 0.10", runtime: Mix.env() == :dev},
      {:tailwind, "~> 0.3", runtime: Mix.env() == :dev},
      {:heroicons,
       github: "tailwindlabs/heroicons",
       tag: "v2.2.0",
       sparse: "optimized",
       app: false,
       compile: false,
       depth: 1},
      {:telemetry_metrics, "~> 1.0"},
      {:telemetry_poller, "~> 1.0"},
      {:gettext, "~> 1.0"},
      {:jason, "~> 1.2"},
      {:dns_cluster, "~> 0.2.0"},
      {:bandit, "~> 1.5"},

      # HTTP Client for NIM APIs
      {:req, "~> 0.5"},

      # Vector embeddings (pgvector)
      {:pgvector, "~> 0.3"},

      # Background jobs
      {:oban, "~> 2.18"},

      # Caching
      {:cachex, "~> 3.6"},

      # UUID generation

      # Password hashing
      {:bcrypt_elixir, "~> 3.1"},

      # Qdrant vector database client
      {:qdrant, "~> 0.0.8"},

      # Message Broker pipeline
      {:broadway_rabbitmq, "~> 0.7"},
      {:amqp, "~> 4.0"},

      # Redis Client
      {:redix, "~> 1.1"},
      {:castore, ">= 0.0.0"},

      # Environment variables from .env file
      {:dotenvy, "~> 0.8"},

      # =========================================================================
      # Development & Code Quality Tools
      # =========================================================================

      # Static code analysis
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},

      # Type checking (Dialyzer wrapper)
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},

      # Documentation generator (override: true to resolve conflict with qdrant)
      {:ex_doc, "~> 0.34", only: [:dev, :test], runtime: false, override: true},

      # Test coverage
      {:excoveralls, "~> 0.18", only: :test},

      # Mocks for testing
      {:mox, "~> 1.0"}
    ]
  end

  # Aliases are shortcuts or tasks specific to the current project.
  # For example, to install project dependencies and perform other setup tasks, run:
  #
  #     $ mix setup
  #
  # See the documentation for `Mix` for more info on aliases.
  defp aliases do
    [
      setup: ["deps.get", "ecto.setup", "assets.setup", "assets.build"],
      "ecto.setup": ["ecto.create", "ecto.migrate", "run priv/repo/seeds.exs"],
      "ecto.reset": ["ecto.drop", "ecto.setup"],
      test: ["ecto.create --quiet", "ecto.migrate --quiet", "test"],
      "assets.setup": ["tailwind.install --if-missing", "esbuild.install --if-missing"],
      "assets.build": ["compile", "tailwind viva", "esbuild viva"],
      "assets.deploy": [
        "tailwind viva --minify",
        "esbuild viva --minify",
        "phx.digest"
      ],
      precommit: ["compile --warnings-as-errors", "deps.unlock --unused", "format", "test"]
    ]
  end
end
