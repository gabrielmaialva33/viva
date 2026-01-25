defmodule VivaNxProject.MixProject do
  use Mix.Project

  def project do
    [
      app: :viva_nx_project,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: ["ffi"],
      deps: deps()
    ]
  end

  def application do
    [
      mod: {Viva.Application, []},
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      # Numerical computing
      {:nx, "~> 0.9"},

      # EXLA backend (Google XLA - supports CUDA)
      {:exla, "~> 0.9"},

      # Optional: Axon for high-level neural networks
      {:axon, "~> 0.7", optional: true},

      # JSON encoding
      {:jason, "~> 1.4"},

      # Rust NIF support
      {:rustler, "~> 0.34"}
    ]
  end
end
