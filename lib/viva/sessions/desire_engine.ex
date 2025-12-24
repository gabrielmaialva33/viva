defmodule Viva.Sessions.DesireEngine do
  @moduledoc """
  Handles desire determination logic based on neurochemistry, personality,
  and somatic markers (body memory).

  Somatic markers influence desires by creating approach/avoidance biases
  based on past experiences. A body warning can suppress social desires,
  while body attraction can amplify them.
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

  @type somatic_bias :: %{
          bias: float(),
          signal: String.t() | nil,
          markers_activated: non_neg_integer()
        }

  @doc """
  Determine the avatar's current desire based on biological, emotional state,
  and somatic markers (body memory).

  Uses fuzzy logic combining neurochemistry with personality traits,
  modulated by somatic bias from past experiences.
  """
  @spec determine(BioState.t(), EmotionalState.t(), Personality.t(), somatic_bias() | nil) ::
          desire()
  def determine(bio, emotional, personality, somatic_bias \\ nil) do
    bias = get_bias(somatic_bias)

    cond do
      # Rest always takes priority
      bio.adenosine > 0.8 ->
        :wants_rest

      # Body warning suppresses social desires if strong enough
      bias < -0.4 and wants_social?(bio, personality) ->
        :wants_something_new

      # Normal desire determination with somatic modulation
      wants_attention?(bio, personality, bias) ->
        :wants_attention

      wants_novelty?(bio, personality) ->
        :wants_something_new

      needs_expression?(emotional) ->
        :wants_to_express

      # Body attraction can amplify crush desire
      wants_crush?(bio, bias) ->
        :wants_to_see_crush

      true ->
        :none
    end
  end

  # === Private Functions ===

  defp get_bias(nil), do: 0.0
  defp get_bias(%{bias: bias}), do: bias

  defp wants_social?(bio, personality) do
    bio.oxytocin < 0.4 and personality.extraversion > 0.5
  end

  defp wants_attention?(bio, personality, bias) do
    base_want = bio.oxytocin < 0.3 and (personality.extraversion > 0.6 or :rand.uniform() < 0.4)

    # Positive somatic bias makes social desire more likely
    if bias > 0.2 do
      base_want or (bio.oxytocin < 0.4 and :rand.uniform() < 0.3)
    else
      base_want
    end
  end

  defp wants_novelty?(bio, personality) do
    bio.dopamine < 0.3 and (personality.openness > 0.6 or :rand.uniform() < 0.3)
  end

  defp needs_expression?(emotional) do
    emotional.arousal > 0.7 or emotional.pleasure < -0.6
  end

  defp wants_crush?(bio, bias) do
    base_want = bio.libido > 0.6 and bio.oxytocin < 0.5

    # Positive somatic bias lowers threshold, negative raises it
    if bias > 0.3 do
      bio.libido > 0.4 and bio.oxytocin < 0.6
    else
      base_want
    end
  end
end
