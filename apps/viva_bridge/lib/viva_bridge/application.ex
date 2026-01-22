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

    # MetaLearner starts paused by default (safety first)
    # Enable with: VivaBridge.Firmware.MetaLearner.resume()
    meta_learner_paused = System.get_env("VIVA_META_LEARNER_ACTIVE") != "true"

    children =
      if skip_nif or is_test do
        [
          {VivaBridge.Music, []},
          # MetaLearner monitors performance and triggers evolution
          {VivaBridge.Firmware.MetaLearner, [paused: meta_learner_paused]}
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
          {VivaBridge.Music, []},
          # Chronos - The Time Lord (Transformer-based Interoception)
          {VivaBridge.Chronos, []},
          # ULTRA - The Graph Weaver (Knowledge Graph Reasoning)
          {VivaBridge.Ultra, []},
          # CORTEX - The Liquid Brain (Biological Physics)
          {VivaBridge.Cortex, []},
          # MetaLearner monitors performance and triggers evolution
          {VivaBridge.Firmware.MetaLearner, [paused: meta_learner_paused]}
        ]
      end

    opts = [strategy: :one_for_one, name: VivaBridge.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
