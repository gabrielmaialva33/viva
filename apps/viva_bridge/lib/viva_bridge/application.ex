defmodule VivaBridge.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    # Check if we should start BodyServer
    # Skip in test environment or when VIVA_SKIP_NIF=true
    skip_nif = System.get_env("VIVA_SKIP_NIF") == "true"
    is_test = Mix.env() == :test

    children =
      if skip_nif or is_test do
        [
          {VivaBridge.Music, []}
        ]
      else
        [
          # BodyServer - unified interoception (hardware sensing + emotional dynamics)
          # Broadcasts state to "body:state" topic every 500ms
          {VivaBridge.BodyServer,
           tick_interval: 500,
           cusp_enabled: true,
           cusp_sensitivity: 0.5,
           pubsub: Viva.PubSub,
           topic: "body:state"},
          {VivaBridge.Music, []}
        ]
      end

    opts = [strategy: :one_for_one, name: VivaBridge.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
