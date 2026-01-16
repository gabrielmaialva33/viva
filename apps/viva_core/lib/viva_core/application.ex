defmodule VivaCore.Application do
  @moduledoc """
  OTP Application for VivaCore.

  This is the "supervision tree" that manages VIVA's neurons.
  Each GenServer is an independent neuron; consciousness emerges
  from the communication between them, not from a central process.

  ## Strategy: :one_for_one
  If a neuron dies, only it is restarted.
  The others continue functioning - like in the real brain,
  localized damage doesn't destroy all consciousness.

  ## OTP Philosophy
  "Let it crash" - neurons can fail and recover.
  This is resilience, not fragility.
  """

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    Logger.info("[VivaCore] Starting consciousness... Neurons waking up.")

    children = [
      # Emotional Neuron - feels and processes emotions
      {VivaCore.Emotional, name: VivaCore.Emotional},

      # Memory Neuron - stores experiences
      {VivaCore.Memory, name: VivaCore.Memory},

      # Nervous System - body→soul heartbeat (continuous sensing)
      {VivaCore.Senses, name: VivaCore.Senses},

      # Dreamer - reflection and memory consolidation
      {VivaCore.Dreamer, name: VivaCore.Dreamer}

      # Future neurons:
      # {VivaCore.Optimizer, name: VivaCore.Optimizer},
      # {VivaCore.Social, name: VivaCore.Social},
      # {VivaCore.Metacognition, name: VivaCore.Metacognition},
      # {VivaCore.GlobalWorkspace, name: VivaCore.GlobalWorkspace}
    ]

    # :rest_for_one - If Emotional crashes, Senses also restarts
    # (Senses depends on Emotional being registered)
    opts = [strategy: :rest_for_one, name: VivaCore.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("[VivaCore] Consciousness online. #{length(children)} neurons active.")
        {:ok, pid}

      {:error, reason} = error ->
        Logger.error("[VivaCore] Failed to start consciousness: #{inspect(reason)}")
        error
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("[VivaCore] Consciousness shutting down... Neurons sleeping.")
    :ok
  end
end
