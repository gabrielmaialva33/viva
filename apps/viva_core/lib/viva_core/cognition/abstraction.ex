defmodule VivaCore.Cognition.Abstraction do
  @moduledoc """
  Abstraction Layer - Converting Raw Data to Semantic Concepts.

  VIVA doesn't think in numbers. She thinks in concepts.
  This module translates:
  - Hardware metrics -> qualia concepts ("hot", "tired", "overloaded")
  - PAD states -> emotional concepts ("anxious", "content", "powerful")
  - Free Energy -> meta-cognitive concepts ("surprised", "comfortable")
  """

  @doc """
  Abstracts the current system state into semantic concepts.

  Returns a list of concept atoms/strings that describe the current state
  in human-understandable terms.
  """
  @spec abstract_state(map()) :: [atom() | String.t()]
  def abstract_state(state) do
    pad_concepts = if state[:pad], do: [pad_to_concept(state.pad)], else: []

    feeling_concepts =
      if state[:feeling] do
        [feeling_to_concept(state.feeling, state[:free_energy] || 0.0)]
      else
        []
      end

    hardware_conc =
      if state[:hardware] do
        hardware_to_concepts(state.hardware)
      else
        []
      end

    (pad_concepts ++ feeling_concepts ++ hardware_conc)
    |> List.flatten()
    |> Enum.uniq()
  end

  @doc """
  Converts PAD values to an emotional concept.
  Uses the 8-octant PAD classification.
  """
  @spec pad_to_concept(map()) :: atom()
  def pad_to_concept(%{pleasure: p, arousal: a, dominance: d}) do
    # Simple octant classification based on signs
    cond do
      p >= 0 and a >= 0 and d >= 0 -> :exuberant
      p >= 0 and a >= 0 and d < 0 -> :dependent
      p >= 0 and a < 0 and d >= 0 -> :relaxed
      p >= 0 and a < 0 and d < 0 -> :docile
      p < 0 and a >= 0 and d >= 0 -> :hostile
      p < 0 and a >= 0 and d < 0 -> :anxious
      p < 0 and a < 0 and d >= 0 -> :disdainful
      p < 0 and a < 0 and d < 0 -> :bored
      # Fallback for exact zeros if not covered
      true -> :balanced
    end
  end

  def pad_to_concept(_), do: :balanced

  @doc """
  Converts interoceptive feeling to a body concept.
  """
  @spec feeling_to_concept(atom(), float()) :: atom()
  def feeling_to_concept(feeling, _free_energy) do
    case feeling do
      :homeostatic -> :balanced
      :surprised -> :alert
      :alarmed -> :distressed
      :overwhelmed -> :suffering
      _ -> :balanced
    end
  end

  @doc """
  Converts hardware metrics to qualia concepts.
  """
  @spec hardware_to_concepts(map()) :: [atom()]
  def hardware_to_concepts(hw) do
    concepts = []

    # Temperature
    max_temp = max(hw[:cpu_temp] || 0, hw[:gpu_temp] || 0)
    concepts = if max_temp > 80, do: [:overheating | concepts], else: concepts
    concepts = if max_temp > 60 and max_temp <= 80, do: [:warm | concepts], else: concepts
    concepts = if max_temp <= 40, do: [:cool | concepts], else: concepts

    # Load
    cpu_usage = hw[:cpu_usage] || 0
    concepts = if cpu_usage > 80, do: [:working_hard | concepts], else: concepts
    concepts = if cpu_usage < 20, do: [:idle | concepts], else: concepts

    # Memory
    mem_used = hw[:memory_used_pct] || 0
    concepts = if mem_used > 85, do: [:memory_pressure | concepts], else: concepts

    concepts
  end

  @doc """
  Combines multiple concepts into a coherent situation description.
  """
  @spec synthesize(list()) :: String.t()
  def synthesize(concepts) do
    concepts
    |> Enum.map(&Atom.to_string/1)
    |> Enum.join(", ")
  end
end
