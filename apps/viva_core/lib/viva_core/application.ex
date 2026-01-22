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
  require VivaLog

  @impl true
  def start(_type, _args) do
    VivaLog.info(:viva_core, :starting_consciousness)

    children = [
      # PubSub - message bus for inter-neuron communication
      # Must be first so others can subscribe during init
      {Phoenix.PubSub, name: Viva.PubSub},

      # Body Schema - self-awareness of hardware capabilities
      # Must start before Emotional so it can configure emotional baseline
      {VivaCore.BodySchema, name: VivaCore.BodySchema},

      # Interoception - the Digital Insula
      # Precision-weighted prediction error on host metrics
      # Must start before Emotional to provide Free Energy signals
      {VivaCore.Interoception, name: VivaCore.Interoception},

      # DatasetCollector - captures interoceptive ticks for Chronos training
      # Receives data from Interoception, writes to priv/datasets/
      {VivaCore.DatasetCollector, name: VivaCore.DatasetCollector},

      # Emotional Neuron - feels and processes emotions
      {VivaCore.Emotional, name: VivaCore.Emotional},

      # Memory Neuron - stores experiences
      {VivaCore.Memory, name: VivaCore.Memory},

      # Nervous System - body→soul heartbeat (continuous sensing)
      {VivaCore.Senses, name: VivaCore.Senses},

      # Dreamer - reflection and memory consolidation
      {VivaCore.Dreamer, name: VivaCore.Dreamer},

      # Agency - digital hands for self-diagnosis
      # Safe command execution to resolve interoceptive distress
      {VivaCore.Agency, name: VivaCore.Agency},

      # Voice - proto-language with Hebbian learning
      # Emergent communication through babbling and association
      {VivaCore.Voice, name: VivaCore.Voice},

      # GLOBAL WORKSPACE (Thoughtseeds)
      # The theater of consciousness where mental objects compete
      {VivaCore.Consciousness.Workspace, name: VivaCore.Consciousness.Workspace}

      # Future neurons:
      # {VivaCore.Optimizer, name: VivaCore.Optimizer},
    ]

    # :one_for_one - If a neuron dies, only it is restarted
    # Each neuron is independent and can tolerate temporary absence of others
    # (Emotional has timeout detection for Senses death)
    opts = [strategy: :one_for_one, name: VivaCore.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        VivaLog.info(:viva_core, :consciousness_online, count: length(children))
        {:ok, pid}

      {:error, reason} = error ->
        VivaLog.error(:viva_core, :startup_failed, reason: inspect(reason))
        error
    end
  end

  @impl true
  def stop(_state) do
    VivaLog.info(:viva_core, :shutting_down)
    :ok
  end
end
