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

      :stub ->
        Logger.debug("[Brain] Using stub (NIF not implemented)")
        :ok
    end
  end

  @doc """
  Process an experience.
  If emotional state is intense, learning (synaptic adjustment) occurs automatically.
  """
  def experience(text, %{pleasure: p, arousal: a, dominance: d}) do
    vector = Body.brain_experience(text, p, a, d)
    {:ok, vector}
  end

  # Helper for neutral experience (inference only, mostly)
  def experience(text) do
    experience(text, %{pleasure: 0.0, arousal: 0.0, dominance: 0.0})
  end

  @doc """
  Recalls memories related to the given text.
  Embeds the text using the Cortex and then searches the Memory.
  """
  def recall(text, limit \\ 5) do
    # 1. Embed text to vector (using neutral experience as inference)
    case experience(text) do
      {:ok, vector} ->
        # 2. Search memory
        VivaBridge.Memory.search(vector, limit)

      _ ->
        []
    end
  end
end
