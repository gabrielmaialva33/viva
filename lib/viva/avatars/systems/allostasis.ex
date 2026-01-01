defmodule Viva.Avatars.Systems.Allostasis do
  @moduledoc """
  The Allostatic Load Engine.

  Tracks and applies cumulative stress effects:
  1. High cortisol over time increases allostatic load
  2. High load causes receptor downregulation (emotional blunting)
  3. Recovery capacity decreases as load increases
  4. Cognitive impairment affects decision-making

  Integration points:
  - TICK: Called in LifeProcess tick to update load based on cortisol
  - APPLY: Modifies emotional output in Psychology.calculate_emotional_state()
  """

  alias Viva.Avatars.AllostasisState
  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState

  # Cortisol threshold for "high stress"
  @high_stress_threshold 0.6

  # Hours of history to keep for cortisol trend
  @cortisol_history_size 24

  # Load accumulation rate per hour of high stress
  @load_accumulation_rate 0.01

  # Recovery rate when cortisol is low (per hour)
  @base_recovery_rate 0.02

  # === Type Definitions ===

  @type phenomenology_result :: %{
          quality: :peaceful | :pressured | :overwhelmed | :burnout,
          narrative: String.t(),
          attention_capacity: :broad | :moderate | :narrow | :fragmented,
          time_horizon: :long | :medium | :very_short | :none,
          threat_sensitivity: :low | :moderate | :very_high | :exhausted
        }

  @type impairment_result :: %{
          cognitive_penalty: float(),
          impulsivity_bonus: float(),
          risk_aversion_shift: float()
        }

  # === Public API ===

  @doc """
  Main tick function. Updates allostatic load based on current cortisol.
  Called every tick (representing 10 simulated minutes = 0.167 hours).
  """
  @spec tick(AllostasisState.t(), BioState.t(), float()) :: AllostasisState.t()
  def tick(allostasis, bio, hours_elapsed \\ 0.167) do
    # 1. Update cortisol history
    new_history = update_cortisol_history(allostasis.cortisol_history, bio.cortisol)

    # 2. Calculate if we're in high-stress state
    is_high_stress = bio.cortisol > @high_stress_threshold

    # 3. Update high stress hours
    new_high_stress_hours =
      if is_high_stress do
        allostasis.high_stress_hours + hours_elapsed
      else
        max(0.0, allostasis.high_stress_hours - hours_elapsed * 0.5)
      end

    # 4. Update load level
    new_load =
      if is_high_stress do
        accumulate_load(allostasis.load_level, hours_elapsed, bio.cortisol)
      else
        recover_load(allostasis.load_level, hours_elapsed, allostasis.recovery_capacity)
      end

    # 5. Update derived values
    new_sensitivity = calculate_receptor_sensitivity(new_load)
    new_recovery = calculate_recovery_capacity(new_load, new_high_stress_hours)
    new_impairment = calculate_cognitive_impairment(new_load)

    # 6. Check for recovery milestone
    new_recovery_at =
      if allostasis.load_level > 0.3 and new_load < 0.3 do
        DateTime.utc_now()
      else
        allostasis.last_recovery_at
      end

    %{
      allostasis
      | cortisol_history: new_history,
        high_stress_hours: new_high_stress_hours,
        load_level: new_load,
        receptor_sensitivity: new_sensitivity,
        recovery_capacity: new_recovery,
        cognitive_impairment: new_impairment,
        last_recovery_at: new_recovery_at
    }
  end

  @doc """
  Apply allostatic effects to emotional state.
  Called after Psychology.calculate_emotional_state().
  """
  @spec dampen_emotions(EmotionalState.t(), AllostasisState.t()) :: EmotionalState.t()
  def dampen_emotions(emotional, allostasis) do
    sensitivity = allostasis.receptor_sensitivity

    # ASYMMETRIC DAMPENING: Joy fades quickly, pain persists
    # Positive emotions are fully dampened by receptor sensitivity
    # Negative emotions are 50% less dampened (suffering persists)
    dampened_pleasure =
      if emotional.pleasure >= 0 do
        emotional.pleasure * sensitivity
      else
        negative_sensitivity = 0.5 + sensitivity * 0.5
        emotional.pleasure * negative_sensitivity
      end

    dampened_arousal = emotional.arousal * sensitivity

    # Update mood label if significant dampening
    new_mood =
      if sensitivity < 0.5 and emotional.mood_label not in ["numb", "exhausted"] do
        determine_burnout_mood(emotional.pleasure, sensitivity)
      else
        emotional.mood_label
      end

    %{
      emotional
      | pleasure: dampened_pleasure,
        arousal: dampened_arousal,
        mood_label: new_mood
    }
  end

  @doc """
  Get cognitive impairment modifier for decision-making.
  Returns 1.0 (no penalty) to 0.5 (50% reduced effectiveness).
  """
  @spec cognitive_penalty(AllostasisState.t()) :: float()
  def cognitive_penalty(allostasis) do
    1.0 - allostasis.cognitive_impairment * 0.5
  end

  @doc """
  Check if avatar is approaching burnout.
  """
  @spec burnout?(AllostasisState.t()) :: boolean()
  def burnout?(allostasis) do
    allostasis.load_level > 0.8 and allostasis.receptor_sensitivity < 0.4
  end

  @doc """
  Get a human-readable description of allostatic state.
  """
  @spec describe(AllostasisState.t()) :: String.t()
  def describe(allostasis) do
    cond do
      burnout?(allostasis) ->
        "Experiencing burnout - emotionally exhausted and unable to recover."

      allostasis.load_level > 0.6 ->
        "Under significant chronic stress - emotional responses are dampened."

      allostasis.load_level > 0.3 ->
        "Moderate stress accumulation - some emotional blunting occurring."

      allostasis.load_level > 0.1 ->
        "Mild stress - functioning normally with slight fatigue."

      true ->
        "Well-rested and resilient."
    end
  end

  # === Phenomenology Functions ===

  @doc """
  Returns the subjective EXPERIENCE of allostatic load.
  Not just a number - describes what the avatar FEELS phenomenologically.
  """
  @spec phenomenology(AllostasisState.t()) :: phenomenology_result()
  def phenomenology(%AllostasisState{load_level: load}) do
    cond do
      load < 0.2 ->
        %{
          quality: :peaceful,
          narrative: "I feel grounded and resilient. There's a quiet strength within me.",
          attention_capacity: :broad,
          time_horizon: :long,
          threat_sensitivity: :low
        }

      load < 0.5 ->
        %{
          quality: :pressured,
          narrative:
            "There's a low hum of tension, but I'm managing. I can feel the weight of things.",
          attention_capacity: :moderate,
          time_horizon: :medium,
          threat_sensitivity: :moderate
        }

      load < 0.75 ->
        %{
          quality: :overwhelmed,
          narrative:
            "My nerves are frayed. Everything feels urgent and threatening. I can barely think straight.",
          attention_capacity: :narrow,
          time_horizon: :very_short,
          threat_sensitivity: :very_high
        }

      true ->
        %{
          quality: :burnout,
          narrative:
            "I'm numb. Nothing matters. I can barely function. Just... existing takes everything.",
          attention_capacity: :fragmented,
          time_horizon: :none,
          threat_sensitivity: :exhausted
        }
    end
  end

  @doc """
  Returns how allostatic load impairs decision-making.
  High load leads to more impulsive, risk-averse decisions.
  """
  @spec decision_impairment(AllostasisState.t()) :: impairment_result()
  def decision_impairment(%AllostasisState{load_level: load} = state) do
    # Cognitive penalty (already exists, reused)
    cog_penalty = cognitive_penalty(state)

    # Impulsivity increases with load (less deliberation)
    impulsivity =
      cond do
        load < 0.3 -> 0.0
        load < 0.5 -> (load - 0.3) * 0.5
        load < 0.7 -> 0.1 + (load - 0.5) * 0.75
        true -> 0.25 + (load - 0.7) * 1.0
      end

    # Risk aversion shifts with load
    risk_shift =
      cond do
        load < 0.3 -> 0.0
        load < 0.6 -> -0.2
        load < 0.8 -> -0.3
        true -> 0.3
      end

    %{
      cognitive_penalty: cog_penalty,
      impulsivity_bonus: min(impulsivity, 0.5),
      risk_aversion_shift: risk_shift
    }
  end

  @doc """
  Generates recovery fantasy when under significant stress.
  The avatar imagines relief based on personality.
  """
  @spec generate_recovery_fantasy(AllostasisState.t(), map()) ::
          {:fantasy_activated, String.t()} | :no_fantasy
  def generate_recovery_fantasy(%AllostasisState{load_level: load}, personality)
      when load > 0.6 do
    fantasy =
      cond do
        personality.extraversion > 0.7 ->
          "I dream of a weekend surrounded by friends, laughing, feeling alive again."

        personality.extraversion < 0.3 ->
          "I crave solitude. A quiet place, no demands, just... peace."

        personality.openness > 0.7 ->
          "I fantasize about escaping somewhere new, discovering something that makes me forget all this."

        personality.neuroticism > 0.7 ->
          "I just want to feel safe. To know everything will be okay."

        true ->
          "I fantasize about a break. Just a few days where nothing matters, where I can breathe."
      end

    {:fantasy_activated, fantasy}
  end

  def generate_recovery_fantasy(_, _), do: :no_fantasy

  @doc """
  Returns attention constraints based on allostatic load.
  Under stress, attention narrows to immediate threats.
  """
  @spec attention_constraints(AllostasisState.t()) :: map()
  def attention_constraints(%AllostasisState{} = allostasis) do
    phenom = phenomenology(allostasis)

    case phenom.attention_capacity do
      :broad ->
        %{
          can_plan_ahead: true,
          can_consider_abstract: true,
          threat_filter: :balanced,
          memory_access: :full
        }

      :moderate ->
        %{
          can_plan_ahead: true,
          can_consider_abstract: :limited,
          threat_filter: :slightly_biased,
          memory_access: :mostly_full
        }

      :narrow ->
        %{
          can_plan_ahead: false,
          can_consider_abstract: false,
          threat_filter: :threat_focused,
          memory_access: :recent_only
        }

      :fragmented ->
        %{
          can_plan_ahead: false,
          can_consider_abstract: false,
          threat_filter: :blunted,
          memory_access: :impaired
        }
    end
  end

  # === Private Functions ===

  defp update_cortisol_history(history, cortisol) do
    [cortisol | Enum.take(history, @cortisol_history_size - 1)]
  end

  defp accumulate_load(current_load, hours, cortisol) do
    # Higher cortisol = faster accumulation
    cortisol_factor = cortisol / @high_stress_threshold
    accumulation = @load_accumulation_rate * hours * cortisol_factor

    min(current_load + accumulation, 1.0)
  end

  defp recover_load(current_load, hours, recovery_capacity) do
    recovery = @base_recovery_rate * hours * recovery_capacity

    max(current_load - recovery, 0.0)
  end

  defp calculate_receptor_sensitivity(load) do
    # Sensitivity decreases as load increases
    # At 0 load = 1.0 sensitivity
    # At 1.0 load = 0.3 sensitivity (severe blunting but not complete)
    1.0 - load * 0.7
  end

  defp calculate_recovery_capacity(load, high_stress_hours) do
    # Recovery capacity decreases with both load and sustained stress
    base_capacity = 1.0 - load * 0.5

    # Sustained high stress further impairs recovery
    sustained_penalty = min(high_stress_hours / 100, 0.3)

    max(base_capacity - sustained_penalty, 0.1)
  end

  defp calculate_cognitive_impairment(load) do
    # Cognitive impairment increases non-linearly with load
    # More impairment at higher loads
    impairment =
      cond do
        load < 0.3 -> 0.0
        load < 0.5 -> (load - 0.3) * 0.5
        load < 0.7 -> 0.1 + (load - 0.5) * 0.75
        true -> 0.25 + (load - 0.7) * 1.5
      end

    min(impairment, 0.7)
  end

  defp determine_burnout_mood(pleasure, sensitivity) do
    cond do
      sensitivity < 0.3 -> "numb"
      sensitivity < 0.5 and pleasure < 0 -> "exhausted"
      sensitivity < 0.5 -> "depleted"
      true -> "fatigued"
    end
  end
end
