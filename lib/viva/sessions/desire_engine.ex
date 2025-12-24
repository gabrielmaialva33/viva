defmodule Viva.Sessions.DesireEngine do
  @moduledoc """
  Handles desire determination logic based on neurochemistry and personality.
  Extracted from LifeProcess to reduce module dependencies.
  """

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality

  @type desire ::
          :wants_rest
          | :wants_attention
          | :wants_something_new
          | :wants_to_express
          | :wants_to_see_crush
          | :none

  @doc """
  Determine the avatar's current desire based on biological and emotional state.
  Uses fuzzy logic combining neurochemistry with personality traits.
  """
  @spec determine(BioState.t(), EmotionalState.t(), Personality.t()) :: desire()
  def determine(bio, emotional, personality) do
    cond do
      bio.adenosine > 0.8 -> :wants_rest
      wants_attention?(bio, personality) -> :wants_attention
      wants_novelty?(bio, personality) -> :wants_something_new
      needs_expression?(emotional) -> :wants_to_express
      wants_crush?(bio) -> :wants_to_see_crush
      true -> :none
    end
  end

  # === Private Functions ===

  defp wants_attention?(bio, personality) do
    bio.oxytocin < 0.3 and (personality.extraversion > 0.6 or :rand.uniform() < 0.4)
  end

  defp wants_novelty?(bio, personality) do
    bio.dopamine < 0.3 and (personality.openness > 0.6 or :rand.uniform() < 0.3)
  end

  defp needs_expression?(emotional) do
    emotional.arousal > 0.7 or emotional.pleasure < -0.6
  end

  defp wants_crush?(bio) do
    bio.libido > 0.6 and bio.oxytocin < 0.5
  end
end
