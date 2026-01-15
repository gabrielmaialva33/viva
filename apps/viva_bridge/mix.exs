defmodule VivaBridge.MixProject do
  use Mix.Project

  @version "0.1.0"

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
      {:rustler, "~> 0.35.0"},
      {:viva_core, in_umbrella: true}
    ]
  end
end
