defmodule Viva.Avatars.Systems.Simulation.PredictiveSimulator do
  @moduledoc """
  Implements Criteria 7 (Self-Modeling) and 6 (Agency).

  This module allows the avatar to run "mental simulations" of potential actions
  before executing them. By predicting the emotional and social outcomes of
  actions using its own Self-Model, the avatar demonstrates:

  1. Agency: Choosing the path that aligns best with goals/needs.
  2. Distinct Self-Model: Using "knowledge of self" to predict "future self".
  """

  alias Viva.Avatars.InternalState
  alias Viva.Avatars.Personality

  @type action_plan :: %{
          action: atom(),
          target: atom() | String.t(),
          predicted_outcome: map(),
          score: float()
        }

  @doc """
  Evaluates a list of potential actions and returns the best one based on
  simulated outcomes for the avatar's future state.
  """
  @spec evaluate_options(list(map()), InternalState.t(), Personality.t()) :: action_plan()
  def evaluate_options(options, current_state, personality) do
    # Parallel simulation of each option
    # In a real brain this is massively parallel; here we map
    simulated_plans =
      Enum.map(options, fn option ->
        simulate_outcome(option, current_state, personality)
      end)

    # Agency decision: Pick the one with highest score
    Enum.max_by(simulated_plans, fn plan -> plan.score end)
  end

  defp simulate_outcome(option, state, personality) do
    # 1. Project Future: How will I feel if I do X?
    # Uses SelfModel to predict emotional reaction
    predicted_emotion = predict_emotional_impact(option, state.consciousness.self_model)

    # 2. Project Social: How will they react?
    # (Simplified hook to social dynamics logic)
    predicted_social = predict_social_impact(option, state)

    # 3. Calculate "Agency Score"
    # Balance between:
    # - Hedonic gain (will I feel better?)
    # - Goal alignment (does this fit my personality/goals?)
    # - Risk (social rejection?)
    score = calculate_agency_score(predicted_emotion, predicted_social, personality)

    %{
      action: option.action,
      target: option.target,
      predicted_outcome: %{
        emotion: predicted_emotion,
        social: predicted_social
      },
      score: score
    }
  end

  defp predict_emotional_impact(option, self_model) do
    # Query SelfModel: "Have I enjoyed this before?"
    past_experience =
      Enum.find(self_model.emotional_patterns, fn p ->
        p.situation == option.action
      end)

    if past_experience do
      # Memory-based prediction
      past_experience.typical_emotion
    else
      # Heuristic prediction
      case option.action do
        :talk -> :interested
        :rest -> :relieved
        :explore -> :curious
        :conflict -> :anxious
        _ -> :neutral
      end
    end
  end

  defp predict_social_impact(option, _) do
    # Placeholder for recursive social prediction
    # "If I say X, they will likely feel Y"
    case option.action do
      :conflict -> :rejection
      :compliment -> :acceptance
      _ -> :neutral
    end
  end

  defp calculate_agency_score(pred_emotion, pred_social, personality) do
    # Base score driven by personality traits
    base_score = 0.5

    # Neurotics avoid anxiety-inducing actions
    anxiety_penalty = if pred_emotion == :anxious, do: personality.neuroticism * 0.8, else: 0.0

    # Extraverts seek social interaction
    social_bonus = if pred_social == :acceptance, do: personality.extraversion * 0.6, else: 0.0

    # Openness rewards curiosity
    novelty_bonus = if pred_emotion == :curious, do: personality.openness * 0.5, else: 0.0

    base_score - anxiety_penalty + social_bonus + novelty_bonus
  end
end
