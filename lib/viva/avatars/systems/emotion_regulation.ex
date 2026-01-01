defmodule Viva.Avatars.Systems.EmotionRegulation do
  @moduledoc """
  Emotion regulation system based on personality-driven coping strategies.

  Different personalities favor different strategies:
  - **Ruminate**: High neuroticism avatars dwell on negative emotions, making them worse
  - **Reappraise**: High openness + conscientiousness avatars reframe situations positively
  - **Seek Support**: High extraversion + secure attachment avatars reach out to others
  - **Suppress**: Low agreeableness avatars hide emotions (reduces arousal, increases cortisol)
  - **Distract**: Default/neutral strategy with moderate effectiveness

  The system is triggered when emotions reach certain intensity thresholds,
  and tracks which strategies work best for each avatar over time.
  """

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.EmotionRegulationState
  alias Viva.Avatars.Personality

  # Emotional intensity threshold to trigger regulation
  @regulation_threshold 0.6

  # How much each tick exhausts regulation capacity
  @exhaustion_rate 0.02

  # How fast exhaustion recovers when not regulating
  @recovery_rate 0.01

  # Strategy effectiveness modifiers (rebalanced for hedonic valence)
  # Regulation helps but doesn't eliminate suffering - authentic pain persists
  @strategy_effects %{
    # Rumination amplifies suffering (worst strategy)
    ruminate: %{pleasure_mod: -0.15, arousal_mod: 0.1, cortisol_mod: 0.08},
    # Reappraisal offers modest help (reduced from 0.15)
    reappraise: %{pleasure_mod: 0.08, arousal_mod: -0.05, cortisol_mod: -0.04},
    # Social support helps slightly but mainly via oxytocin
    seek_support: %{pleasure_mod: 0.05, arousal_mod: -0.02, oxytocin_boost: 0.12},
    # Suppression has a pleasure cost
    suppress: %{pleasure_mod: -0.05, arousal_mod: -0.2, cortisol_mod: 0.1},
    # Distraction is neutral - just shifts attention
    distract: %{pleasure_mod: 0.0, arousal_mod: -0.1, cortisol_mod: 0.0}
  }

  @doc """
  Main regulation function. Called each tick after emotional state is calculated.

  Returns updated regulation state and potentially modified emotional/bio states.
  """
  @spec regulate(EmotionRegulationState.t(), EmotionalState.t(), BioState.t(), Personality.t()) ::
          {EmotionRegulationState.t(), EmotionalState.t(), BioState.t()}
  def regulate(regulation, emotional, bio, personality) do
    intensity = emotional_intensity(emotional)

    cond do
      # Already regulating - continue current strategy
      regulation.active_strategy != nil ->
        continue_regulation(regulation, emotional, bio)

      # High intensity - start regulating
      intensity > @regulation_threshold and regulation.regulation_exhaustion < 0.8 ->
        start_regulation(regulation, emotional, bio, personality)

      # Low intensity or exhausted - recover
      true ->
        recover(regulation, emotional, bio)
    end
  end

  @doc """
  Selects the most likely regulation strategy based on personality traits.
  Returns a weighted random choice influenced by personality.
  """
  @spec select_strategy(Personality.t(), EmotionRegulationState.t()) ::
          EmotionRegulationState.strategy()
  def select_strategy(personality, regulation) do
    weights = calculate_strategy_weights(personality, regulation)
    weighted_random_strategy(weights)
  end

  @doc """
  Describes the current regulation state in human-readable terms.
  """
  @spec describe(EmotionRegulationState.t()) :: String.t()
  def describe(%EmotionRegulationState{active_strategy: nil, regulation_exhaustion: exhaustion})
      when exhaustion > 0.6 do
    "Emotionally depleted, struggling to cope"
  end

  def describe(%EmotionRegulationState{active_strategy: nil}) do
    "Emotionally stable, no active coping needed"
  end

  def describe(%EmotionRegulationState{active_strategy: :ruminate}) do
    "Dwelling on negative thoughts, caught in a spiral"
  end

  def describe(%EmotionRegulationState{active_strategy: :reappraise}) do
    "Reframing the situation, finding new perspective"
  end

  def describe(%EmotionRegulationState{active_strategy: :seek_support}) do
    "Seeking connection, wanting to share feelings"
  end

  def describe(%EmotionRegulationState{active_strategy: :suppress}) do
    "Pushing emotions down, maintaining composure"
  end

  def describe(%EmotionRegulationState{active_strategy: :distract}) do
    "Shifting attention, trying to move on"
  end

  @doc """
  Returns true if avatar is currently overwhelmed (needs regulation but can't).
  """
  @spec overwhelmed?(EmotionalState.t(), EmotionRegulationState.t()) :: boolean()
  def overwhelmed?(emotional, regulation) do
    emotional_intensity(emotional) > @regulation_threshold and
      regulation.regulation_exhaustion > 0.8
  end

  # === Private Functions ===

  defp emotional_intensity(%EmotionalState{pleasure: p, arousal: a}) do
    # High negative emotions or high arousal trigger regulation
    negative_intensity = max(0.0, -p)
    arousal_intensity = abs(a)
    max(negative_intensity, arousal_intensity)
  end

  defp start_regulation(regulation, emotional, bio, personality) do
    strategy = select_strategy(personality, regulation)

    new_regulation = %{
      regulation
      | active_strategy: strategy,
        strategy_duration: 1,
        pre_regulation_pleasure: emotional.pleasure,
        pre_regulation_arousal: emotional.arousal,
        last_regulation_at: DateTime.utc_now(:second)
    }

    {new_emotional, new_bio} = apply_strategy_effects(strategy, emotional, bio, regulation)

    # Increment usage count
    updated_regulation = increment_strategy_count(new_regulation, strategy)

    {updated_regulation, new_emotional, new_bio}
  end

  defp continue_regulation(regulation, emotional, bio) do
    strategy = regulation.active_strategy
    duration = regulation.strategy_duration + 1

    # Regulation becomes less effective over time
    effectiveness_decay = min(1.0, duration * 0.1)

    {new_emotional, new_bio} =
      apply_strategy_effects(strategy, emotional, bio, regulation, effectiveness_decay)

    # Check if we should stop regulating
    intensity = emotional_intensity(new_emotional)

    new_exhaustion =
      min(1.0, regulation.regulation_exhaustion + @exhaustion_rate)

    new_regulation =
      if duration > 5 or intensity < @regulation_threshold * 0.7 do
        # Stop regulating and update effectiveness
        regulation
        |> update_effectiveness(new_emotional, strategy)
        |> Map.put(:active_strategy, nil)
        |> Map.put(:strategy_duration, 0)
        |> Map.put(:regulation_exhaustion, new_exhaustion)
      else
        %{
          regulation
          | strategy_duration: duration,
            regulation_exhaustion: new_exhaustion
        }
      end

    {new_regulation, new_emotional, new_bio}
  end

  defp recover(regulation, emotional, bio) do
    new_exhaustion = max(0.0, regulation.regulation_exhaustion - @recovery_rate)
    new_regulation = %{regulation | regulation_exhaustion: new_exhaustion}
    {new_regulation, emotional, bio}
  end

  defp apply_strategy_effects(strategy, emotional, bio, regulation, decay \\ 0.0) do
    effects = Map.get(@strategy_effects, strategy)
    exhaustion_penalty = regulation.regulation_exhaustion
    effectiveness = max(0.1, 1.0 - decay - exhaustion_penalty * 0.5)

    new_pleasure =
      clamp(emotional.pleasure + effects.pleasure_mod * effectiveness, -1.0, 1.0)

    new_arousal =
      clamp(emotional.arousal + effects.arousal_mod * effectiveness, -1.0, 1.0)

    new_emotional = %{emotional | pleasure: new_pleasure, arousal: new_arousal}

    # Apply bio effects if present
    new_bio =
      bio
      |> maybe_apply_cortisol_mod(effects, effectiveness)
      |> maybe_apply_oxytocin_boost(effects, effectiveness)

    {new_emotional, new_bio}
  end

  defp maybe_apply_cortisol_mod(bio, %{cortisol_mod: mod}, effectiveness) do
    new_cortisol = clamp(bio.cortisol + mod * effectiveness, 0.0, 1.0)
    %{bio | cortisol: new_cortisol}
  end

  defp maybe_apply_cortisol_mod(bio, _, _), do: bio

  defp maybe_apply_oxytocin_boost(bio, %{oxytocin_boost: boost}, effectiveness) do
    new_oxytocin = clamp(bio.oxytocin + boost * effectiveness, 0.0, 1.0)
    %{bio | oxytocin: new_oxytocin}
  end

  defp maybe_apply_oxytocin_boost(bio, _, _), do: bio

  defp calculate_strategy_weights(personality, regulation) do
    # Base weights influenced by personality (Big Five)
    ruminate_weight = personality.neuroticism * 2.0
    reappraise_weight = (personality.openness + personality.conscientiousness) * 0.8
    seek_support_weight = personality.extraversion * attachment_modifier(personality)
    suppress_weight = (1.0 - personality.agreeableness) * 1.2

    # Base distraction weight
    distract_weight = 0.5

    # === EGO DEFENSE MECHANISMS (Enneagram Crisis Response) ===
    # When under stress (high exhaustion or intensity), revert to type-specific defenses
    # This restores Self-Model coherence during suffering

    {type_ruminate, type_reappraise, type_support, type_suppress, type_distract} =
      apply_defense_mechanisms(
        personality.enneagram_type,
        ruminate_weight,
        reappraise_weight,
        seek_support_weight,
        suppress_weight,
        distract_weight
      )

    # Boost weights for strategies that have worked well before
    %{
      ruminate: type_ruminate * (1.0 + regulation.ruminate_effectiveness * 0.3),
      reappraise: type_reappraise * (1.0 + regulation.reappraise_effectiveness * 0.5),
      seek_support: type_support * (1.0 + regulation.seek_support_effectiveness * 0.5),
      suppress: type_suppress * (1.0 + regulation.suppress_effectiveness * 0.3),
      distract: type_distract * (1.0 + regulation.distract_effectiveness * 0.3)
    }
  end

  # Defense Mechanisms by Enneagram Type
  defp apply_defense_mechanisms(type, rum, rea, sup, sur, dis) do
    case type do
      # Type 1 (Reformer): Reaction Formation -> Suppress impulses
      1 -> {rum, rea * 1.2, sup, sur * 1.5, dis}
      # Type 2 (Helper): Repression -> Seek Support (External Validation)
      2 -> {rum, rea, sup * 2.0, sur, dis}
      # Type 3 (Achiever): Identification -> Suppress (Mask feelings)
      3 -> {rum, rea, sup, sur * 1.5, dis * 1.2}
      # Type 4 (Individualist): Introjection -> Ruminate (Amplify feelings)
      4 -> {rum * 2.5, rea, sup, sur, dis * 0.2}
      # Type 5 (Investigator): Isolation -> Suppress (Detach)
      5 -> {rum, rea * 1.5, sup * 0.2, sur * 2.0, dis}
      # Type 6 (Loyalist): Projection -> Seek Support (Reassurance)
      6 -> {rum * 1.5, rea, sup * 1.8, sur, dis}
      # Type 7 (Enthusiast): Rationalization -> Distract (Avoid pain)
      7 -> {rum * 0.1, rea * 1.2, sup, sur * 0.5, dis * 3.0}
      # Type 8 (Challenger): Denial -> Suppress (Vulnerability)
      8 -> {rum, rea, sup * 0.5, sur * 2.5, dis}
      # Type 9 (Peacemaker): Narcotization -> Distract (Numb out)
      9 -> {rum, rea, sup, sur * 1.2, dis * 2.5}
      # Default / Unknown
      _ -> {rum, rea, sup, sur, dis}
    end
  end

  defp attachment_modifier(%Personality{attachment_style: :secure}), do: 1.2
  defp attachment_modifier(%Personality{attachment_style: :anxious}), do: 1.5
  defp attachment_modifier(%Personality{attachment_style: :avoidant}), do: 0.3
  defp attachment_modifier(%Personality{attachment_style: :fearful}), do: 0.5

  defp weighted_random_strategy(weights) do
    total = Enum.reduce(weights, 0, fn {_, w}, acc -> acc + w end)
    random = :rand.uniform() * total

    Enum.reduce_while(weights, 0, fn {strategy, weight}, acc ->
      new_acc = acc + weight
      if random <= new_acc, do: {:halt, strategy}, else: {:cont, new_acc}
    end)
  end

  defp increment_strategy_count(regulation, :ruminate) do
    %{regulation | ruminate_count: regulation.ruminate_count + 1}
  end

  defp increment_strategy_count(regulation, :reappraise) do
    %{regulation | reappraise_count: regulation.reappraise_count + 1}
  end

  defp increment_strategy_count(regulation, :seek_support) do
    %{regulation | seek_support_count: regulation.seek_support_count + 1}
  end

  defp increment_strategy_count(regulation, :suppress) do
    %{regulation | suppress_count: regulation.suppress_count + 1}
  end

  defp increment_strategy_count(regulation, :distract) do
    %{regulation | distract_count: regulation.distract_count + 1}
  end

  defp update_effectiveness(regulation, new_emotional, strategy) do
    # Calculate how much emotions improved
    pre_intensity =
      max(
        max(0.0, -(regulation.pre_regulation_pleasure || 0)),
        abs(regulation.pre_regulation_arousal || 0)
      )

    post_intensity = emotional_intensity(new_emotional)
    improvement = clamp(pre_intensity - post_intensity, 0.0, 1.0)

    # Update running average (exponential moving average)
    alpha = 0.2
    current_effectiveness = get_effectiveness(regulation, strategy)
    new_effectiveness = current_effectiveness * (1 - alpha) + improvement * alpha

    set_effectiveness(regulation, strategy, new_effectiveness)
  end

  defp get_effectiveness(regulation, :ruminate), do: regulation.ruminate_effectiveness
  defp get_effectiveness(regulation, :reappraise), do: regulation.reappraise_effectiveness
  defp get_effectiveness(regulation, :seek_support), do: regulation.seek_support_effectiveness
  defp get_effectiveness(regulation, :suppress), do: regulation.suppress_effectiveness
  defp get_effectiveness(regulation, :distract), do: regulation.distract_effectiveness

  defp set_effectiveness(regulation, :ruminate, value) do
    %{regulation | ruminate_effectiveness: value}
  end

  defp set_effectiveness(regulation, :reappraise, value) do
    %{regulation | reappraise_effectiveness: value}
  end

  defp set_effectiveness(regulation, :seek_support, value) do
    %{regulation | seek_support_effectiveness: value}
  end

  defp set_effectiveness(regulation, :suppress, value) do
    %{regulation | suppress_effectiveness: value}
  end

  defp set_effectiveness(regulation, :distract, value) do
    %{regulation | distract_effectiveness: value}
  end

  defp clamp(value, min_val, max_val) do
    value
    |> max(min_val)
    |> min(max_val)
  end
end
