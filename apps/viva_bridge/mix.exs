defmodule VivaBridge.MixProject do
  use Mix.Project

  @version "0.4.0"

  def project do
    [
      app: :viva_bridge,
      version: @version,
      build_path: "../../_build",
      config_path: "../../config/config.exs",
      deps_path: "../../deps",
      lockfile: "../../mix.lock",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {VivaBridge.Application, []}
    ]
  end

  defp deps do
    [
      {:viva_common, in_umbrella: true},
      {:rustler, "~> 0.35.0"},
      {:phoenix_pubsub, "~> 2.1"},
      {:circuits_uart, "~> 1.5"}
    ]
  end
end
