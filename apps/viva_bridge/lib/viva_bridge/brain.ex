defmodule VivaBridge.Brain do
  @moduledoc """
  High-level API for the VIVA Native Cortex.
  Manages the "Geneis" learning process using SDR and Hebbian logic.
  """

  alias VivaBridge.Body
  require Logger

  @doc """
  Initializes the Native Cortex.
  """
  def init() do
    Logger.info("[Brain] Initializing Native Cortex (Tabula Rasa)...")
    case Body.brain_init() do
      msg when is_binary(msg) ->
        Logger.info("[Brain] #{msg}")
        :ok
      {:error, reason} ->
        Logger.error("[Brain] Failed to init: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Process an experience.
  If emotional state is intense, learning (synaptic adjustment) occurs automatically.
  """
  def experience(text, %{pleasure: p, arousal: a, dominance: d}) do
    case Body.brain_experience(text, p, a, d) do
      vector when is_list(vector) -> {:ok, vector}
      {:error, reason} -> {:error, reason}
    end
  end

  # Helper for neutral experience (inference only, mostly)
  def experience(text) do
    experience(text, %{pleasure: 0.0, arousal: 0.0, dominance: 0.0})
  end
end
