defmodule Viva.Avatars.Systems.Motivation do
  @moduledoc """
  The Motivation Engine - Implements hierarchical drive system.

  Calculates which motivational drive is most urgent based on:
  1. Base urgencies from Enneagram type (primary drive)
  2. Physiological modulation (cortisol, dopamine, oxytocin, adenosine)
  3. Emotional state (pleasure, arousal, dominance)
  4. Personality traits (Big Five modulation)
  5. Frustration from blocked drives

  ## Drive Hierarchy (Maslow-inspired)
  Lower drives take priority when urgent:
  - :survival → Overrides all when critical (adenosine > 0.8, cortisol > 0.9)
  - :safety → Activated by high cortisol/fear
  - :belonging → Activated by low oxytocin/loneliness
  - :status → Activated by shame/low self-esteem
  - :autonomy → Activated by constraint/low dominance
  - :transcendence → Requires lower drives to be satisfied

  ## Integration points:
  - TICK: Called in LifeProcess after allostasis to update urgencies
  - OUTPUT: Provides current_urgent_drive for DesireEngine
  """

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.MotivationState
  alias Viva.Avatars.Personality

  # === Constants ===

  # Thresholds for biological urgency modulation
  @high_cortisol_threshold 0.6
  @low_dopamine_threshold 0.3
  @low_oxytocin_threshold 0.3
  @high_adenosine_threshold 0.7

  # Urgency modulation rates
  @cortisol_safety_boost 0.3
  @dopamine_autonomy_penalty 0.2
  @oxytocin_belonging_boost 0.3
  @adenosine_survival_boost 0.4

  # Frustration effects
  @frustration_urgency_boost 0.2
  @block_duration_threshold 5

  # === Type Definitions ===

  @type drive :: MotivationState.drive()

  # === Public API ===

  @doc """
  Main tick function. Updates motivation state based on current biological,
  emotional, and personality state.

  Called every tick (representing 10 simulated minutes).
  """
  @spec tick(MotivationState.t(), BioState.t(), EmotionalState.t(), Personality.t()) ::
          MotivationState.t()
  def tick(motivation, bio, emotional, personality) do
    # 1. Get base urgencies from current state
    base_urgencies = MotivationState.urgencies(motivation)

    # 2. Apply biological modulation
    bio_modulated = apply_biological_pressure(base_urgencies, bio)

    # 3. Apply emotional modulation
    emotional_modulated = apply_emotional_pressure(bio_modulated, emotional)

    # 4. Apply personality modulation
    personality_modulated = apply_personality_modulation(emotional_modulated, personality)

    # 5. Apply frustration effects
    final_urgencies = apply_frustration_boost(personality_modulated, motivation)

    # 6. Find most urgent drive
    urgent_drive = find_most_urgent(final_urgencies)

    # 7. Update state
    %{
      motivation
      | survival_urgency: final_urgencies.survival,
        safety_urgency: final_urgencies.safety,
        belonging_urgency: final_urgencies.belonging,
        status_urgency: final_urgencies.status,
        autonomy_urgency: final_urgencies.autonomy,
        transcendence_urgency: final_urgencies.transcendence,
        current_urgent_drive: urgent_drive,
        last_updated: DateTime.utc_now(:second)
    }
  end

  @doc """
  Returns the current most urgent drive.
  """
  @spec calculate_urgent_drive(MotivationState.t()) :: drive()
  def calculate_urgent_drive(%MotivationState{current_urgent_drive: drive}), do: drive

  @doc """
  Records that a drive was blocked (action failed or was prevented).
  Increases frustration which affects future urgency calculations.
  """
  @spec block_drive(MotivationState.t(), drive()) :: MotivationState.t()
  def block_drive(motivation, drive) do
    if motivation.blocked_drive == drive do
      # Same drive blocked again - increase duration
      %{motivation | block_duration: motivation.block_duration + 1}
    else
      # New blocked drive
      %{motivation | blocked_drive: drive, block_duration: 1}
    end
  end

  @doc """
  Clears drive block when the drive is satisfied.
  """
  @spec satisfy_drive(MotivationState.t(), drive()) :: MotivationState.t()
  def satisfy_drive(motivation, drive) do
    if motivation.blocked_drive == drive do
      %{motivation | blocked_drive: nil, block_duration: 0}
    else
      motivation
    end
  end

  @doc """
  Returns a human-readable description of current motivation state.
  """
  @spec describe(MotivationState.t()) :: String.t()
  def describe(%MotivationState{} = motivation) do
    drive = motivation.current_urgent_drive
    urgency = get_urgency(motivation, drive)

    base = drive_description(drive, urgency)

    if MotivationState.frustrated?(motivation) do
      blocked = motivation.blocked_drive
      "#{base} Frustrated by blocked #{blocked} drive."
    else
      base
    end
  end

  @doc """
  Maps drive to desire type for DesireEngine compatibility.
  """
  @spec drive_to_desire(drive()) :: atom()
  def drive_to_desire(:survival), do: :wants_rest
  def drive_to_desire(:safety), do: :wants_rest
  def drive_to_desire(:belonging), do: :wants_attention
  def drive_to_desire(:status), do: :wants_to_express
  def drive_to_desire(:autonomy), do: :wants_something_new
  def drive_to_desire(:transcendence), do: :wants_something_new

  # === Private Functions ===

  defp apply_biological_pressure(urgencies, bio) do
    urgencies
    |> maybe_boost_survival(bio)
    |> maybe_boost_safety(bio)
    |> maybe_boost_belonging(bio)
    |> maybe_penalize_autonomy(bio)
  end

  defp maybe_boost_survival(urgencies, %BioState{adenosine: adenosine})
       when adenosine > @high_adenosine_threshold do
    Map.update!(urgencies, :survival, &clamp(&1 + @adenosine_survival_boost))
  end

  defp maybe_boost_survival(urgencies, _), do: urgencies

  defp maybe_boost_safety(urgencies, %BioState{cortisol: cortisol})
       when cortisol > @high_cortisol_threshold do
    boost = (cortisol - @high_cortisol_threshold) * @cortisol_safety_boost * 2

    urgencies
    |> Map.update!(:safety, &clamp(&1 + boost))
    |> Map.update!(:survival, &clamp(&1 + boost * 0.5))
  end

  defp maybe_boost_safety(urgencies, _), do: urgencies

  defp maybe_boost_belonging(urgencies, %BioState{oxytocin: oxytocin})
       when oxytocin < @low_oxytocin_threshold do
    boost = (@low_oxytocin_threshold - oxytocin) * @oxytocin_belonging_boost * 2
    Map.update!(urgencies, :belonging, &clamp(&1 + boost))
  end

  defp maybe_boost_belonging(urgencies, _), do: urgencies

  defp maybe_penalize_autonomy(urgencies, %BioState{dopamine: dopamine})
       when dopamine < @low_dopamine_threshold do
    penalty = (@low_dopamine_threshold - dopamine) * @dopamine_autonomy_penalty

    urgencies
    |> Map.update!(:autonomy, &clamp(&1 - penalty))
    |> Map.update!(:status, &clamp(&1 - penalty * 0.5))
  end

  defp maybe_penalize_autonomy(urgencies, _), do: urgencies

  defp apply_emotional_pressure(urgencies, emotional) do
    urgencies
    |> apply_pleasure_effects(emotional)
    |> apply_arousal_effects(emotional)
    |> apply_dominance_effects(emotional)
  end

  defp apply_pleasure_effects(urgencies, %EmotionalState{pleasure: p}) when p < -0.5 do
    # Negative pleasure (shame/sadness) boosts status need
    boost = abs(p) * 0.2

    urgencies
    |> Map.update!(:status, &clamp(&1 + boost))
    |> Map.update!(:belonging, &clamp(&1 + boost * 0.5))
  end

  defp apply_pleasure_effects(urgencies, %EmotionalState{pleasure: p}) when p > 0.5 do
    # Positive pleasure opens up higher drives
    boost = p * 0.15
    Map.update!(urgencies, :transcendence, &clamp(&1 + boost))
  end

  defp apply_pleasure_effects(urgencies, _), do: urgencies

  defp apply_arousal_effects(urgencies, %EmotionalState{arousal: a}) when a > 0.7 do
    # High arousal + context could mean fear or excitement
    # For now, boost safety slightly
    Map.update!(urgencies, :safety, &clamp(&1 + 0.1))
  end

  defp apply_arousal_effects(urgencies, _), do: urgencies

  defp apply_dominance_effects(urgencies, %EmotionalState{dominance: d}) when d < -0.3 do
    # Low dominance = feeling controlled → autonomy need rises
    boost = abs(d) * 0.25
    Map.update!(urgencies, :autonomy, &clamp(&1 + boost))
  end

  defp apply_dominance_effects(urgencies, %EmotionalState{dominance: d}) when d > 0.5 do
    # High dominance = feeling in control → status/transcendence accessible
    Map.update!(urgencies, :transcendence, &clamp(&1 + 0.1))
  end

  defp apply_dominance_effects(urgencies, _), do: urgencies

  defp apply_personality_modulation(urgencies, personality) do
    urgencies
    |> modulate_by_extraversion(personality.extraversion)
    |> modulate_by_neuroticism(personality.neuroticism)
    |> modulate_by_openness(personality.openness)
    |> modulate_by_agreeableness(personality.agreeableness)
  end

  defp modulate_by_extraversion(urgencies, e) when e > 0.6 do
    # High extraversion → stronger belonging need
    boost = (e - 0.5) * 0.2
    Map.update!(urgencies, :belonging, &clamp(&1 + boost))
  end

  defp modulate_by_extraversion(urgencies, e) when e < 0.4 do
    # Low extraversion → reduced belonging urgency
    reduction = (0.5 - e) * 0.15
    Map.update!(urgencies, :belonging, &clamp(&1 - reduction))
  end

  defp modulate_by_extraversion(urgencies, _), do: urgencies

  defp modulate_by_neuroticism(urgencies, n) when n > 0.6 do
    # High neuroticism → stronger safety need
    boost = (n - 0.5) * 0.25
    Map.update!(urgencies, :safety, &clamp(&1 + boost))
  end

  defp modulate_by_neuroticism(urgencies, _), do: urgencies

  defp modulate_by_openness(urgencies, o) when o > 0.6 do
    # High openness → stronger autonomy and transcendence needs
    boost = (o - 0.5) * 0.2

    urgencies
    |> Map.update!(:autonomy, &clamp(&1 + boost))
    |> Map.update!(:transcendence, &clamp(&1 + boost))
  end

  defp modulate_by_openness(urgencies, _), do: urgencies

  defp modulate_by_agreeableness(urgencies, a) when a > 0.6 do
    # High agreeableness → stronger belonging need
    boost = (a - 0.5) * 0.15
    Map.update!(urgencies, :belonging, &clamp(&1 + boost))
  end

  defp modulate_by_agreeableness(urgencies, _), do: urgencies

  defp apply_frustration_boost(urgencies, motivation) do
    if motivation.blocked_drive && motivation.block_duration > @block_duration_threshold do
      blocked = motivation.blocked_drive
      boost = min(@frustration_urgency_boost * motivation.block_duration / 10, 0.4)
      Map.update!(urgencies, blocked, &clamp(&1 + boost))
    else
      urgencies
    end
  end

  defp find_most_urgent(urgencies) do
    # Apply hierarchy: lower drives win ties
    priority_order = [:survival, :safety, :belonging, :status, :autonomy, :transcendence]

    {drive, _} =
      urgencies
      |> Enum.sort_by(fn {drive, urgency} ->
        priority = Enum.find_index(priority_order, &(&1 == drive)) || 99
        # Sort by urgency descending, then by priority ascending
        {-urgency, priority}
      end)
      |> List.first()

    drive
  end

  defp get_urgency(motivation, drive) do
    case drive do
      :survival -> motivation.survival_urgency
      :safety -> motivation.safety_urgency
      :belonging -> motivation.belonging_urgency
      :status -> motivation.status_urgency
      :autonomy -> motivation.autonomy_urgency
      :transcendence -> motivation.transcendence_urgency
    end
  end

  defp drive_description(:survival, u) when u > 0.7 do
    "Desperately needs rest and recovery. Body is screaming for a break."
  end

  defp drive_description(:survival, _) do
    "Basic needs are calling - time to rest."
  end

  defp drive_description(:safety, u) when u > 0.7 do
    "Feeling unsafe and anxious. Seeking security and stability."
  end

  defp drive_description(:safety, _) do
    "Looking for a sense of security."
  end

  defp drive_description(:belonging, u) when u > 0.7 do
    "Deeply lonely. Craving connection and community."
  end

  defp drive_description(:belonging, _) do
    "Wanting to connect with others."
  end

  defp drive_description(:status, u) when u > 0.7 do
    "Burning need for recognition. Must prove worth."
  end

  defp drive_description(:status, _) do
    "Seeking acknowledgment and respect."
  end

  defp drive_description(:autonomy, u) when u > 0.7 do
    "Feeling trapped. Must break free and choose own path."
  end

  defp drive_description(:autonomy, _) do
    "Wanting freedom and self-determination."
  end

  defp drive_description(:transcendence, u) when u > 0.6 do
    "Seeking meaning beyond the everyday. Drawn to beauty and purpose."
  end

  defp drive_description(:transcendence, _) do
    "Open to moments of wonder and meaning."
  end

  defp clamp(value) do
    value
    |> max(0.0)
    |> min(1.0)
  end
end
