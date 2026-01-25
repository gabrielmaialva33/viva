defmodule VivaCore.Personality do
  @moduledoc """
  VIVA's Personality Traits

  Based on affective personality research (Mehrabian, 1996) and
  "Emotions in Artificial Intelligence" (Borotschnig, 2025).

  Personality defines:
  - Baseline PAD (resting emotional state / attractor point)
  - Reactivity (amplification factor for emotions)
  - Volatility (speed of emotional change)
  - Traits (categorical labels for introspection)

  The personality influences how VIVA processes and expresses emotions,
  providing consistency across different situations while allowing
  adaptation based on long-term experiences.
  """

  require VivaLog

  @default_baseline %{pleasure: 0.1, arousal: 0.05, dominance: 0.1}
  @default_reactivity 1.0
  @default_volatility 1.0
  @default_traits [:curious, :calm]

  @doc "Returns the default baseline personality."
  def baseline, do: @default_baseline

  @persistence_key "viva:personality"

  defstruct [
    # Baseline emotional state (attractor point)
    # VIVA tends to return to this state over time
    baseline: @default_baseline,

    # Reactivity: how much emotions are amplified (1.0 = normal)
    # > 1.0 = more reactive, < 1.0 = dampened
    reactivity: @default_reactivity,

    # Volatility: how quickly emotions change (1.0 = normal)
    # > 1.0 = faster changes, < 1.0 = more stable
    volatility: @default_volatility,

    # Trait labels (for introspection and self-description)
    traits: @default_traits,

    # Timestamp of last adaptation
    last_adapted: nil
  ]

  @type t :: %__MODULE__{
          baseline: %{pleasure: float(), arousal: float(), dominance: float()},
          reactivity: float(),
          volatility: float(),
          traits: [atom()],
          last_adapted: DateTime.t() | nil
        }

  @type pad :: %{pleasure: float(), arousal: float(), dominance: float()}

  # ============================================================
  # Public API
  # ============================================================

  @doc """
  Load personality from persistent storage or return defaults.

  Attempts to load from Redis first, falls back to defaults.
  """
  @spec load() :: t()
  def load do
    case load_from_redis() do
      {:ok, personality} ->
        VivaLog.debug(:personality, :loaded_from_storage)
        personality

      {:error, _reason} ->
        VivaLog.info(:personality, :using_defaults)
        %__MODULE__{}
    end
  end

  @doc """
  Save personality to persistent storage.
  """
  @spec save(t()) :: :ok | {:error, term()}
  def save(%__MODULE__{} = personality) do
    case save_to_redis(personality) do
      :ok ->
        VivaLog.debug(:personality, :saved_to_storage)
        :ok

      error ->
        VivaLog.warning(:personality, :save_failed, error: inspect(error))
        error
    end
  end

  @doc """
  Adapt personality based on long-term experiences.

  This should be called periodically (e.g., during sleep/consolidation)
  to update personality traits based on accumulated experiences.

  ## Parameters
  - personality: Current personality state
  - experiences: List of emotional experiences to learn from
    Each experience: %{pad: PAD, intensity: float, valence: :positive | :negative}

  ## Returns
  Updated personality struct
  """
  @spec adapt(t(), [map()]) :: t()
  def adapt(%__MODULE__{} = personality, experiences) when is_list(experiences) do
    if Enum.empty?(experiences) do
      personality
    else
      # Calculate average emotional tendency from experiences
      avg_pad = aggregate_experiences(experiences)

      # Slowly shift baseline toward experienced average (α = 0.05)
      new_baseline = shift_baseline(personality.baseline, avg_pad, 0.05)

      # Adjust reactivity based on emotional range
      new_reactivity = adjust_reactivity(personality.reactivity, experiences)

      # Update traits based on dominant patterns
      new_traits = infer_traits(new_baseline, experiences)

      updated = %{
        personality
        | baseline: new_baseline,
          reactivity: new_reactivity,
          traits: new_traits,
          last_adapted: DateTime.utc_now()
      }

      VivaLog.info(:personality, :adapted,
        baseline_pleasure: Float.round(new_baseline.pleasure, 3),
        reactivity: Float.round(new_reactivity, 2)
      )

      updated
    end
  end

  @doc """
  Apply personality to a raw emotion (PAD vector).

  This modifies the incoming emotion based on personality traits:
  1. Blend with baseline (regression toward personality mean)
  2. Apply reactivity (amplify or dampen deviations)

  ## Parameters
  - personality: Personality struct
  - raw_pad: Raw emotional PAD from stimulus

  ## Returns
  Modified PAD incorporating personality influence
  """
  @spec apply(t(), pad()) :: pad()
  def apply(%__MODULE__{} = personality, raw_pad) when is_map(raw_pad) do
    # Personality influence weight (how much baseline affects the emotion)
    personality_weight = 0.2

    # 1. Blend raw emotion with baseline
    blended = %{
      pleasure:
        (1 - personality_weight) * get_value(raw_pad, :pleasure) +
          personality_weight * personality.baseline.pleasure,
      arousal:
        (1 - personality_weight) * get_value(raw_pad, :arousal) +
          personality_weight * personality.baseline.arousal,
      dominance:
        (1 - personality_weight) * get_value(raw_pad, :dominance) +
          personality_weight * personality.baseline.dominance
    }

    # 2. Apply reactivity to deviation from baseline
    %{
      pleasure:
        apply_reactivity_dim(
          blended.pleasure,
          personality.baseline.pleasure,
          personality.reactivity
        ),
      arousal:
        apply_reactivity_dim(
          blended.arousal,
          personality.baseline.arousal,
          personality.reactivity
        ),
      dominance:
        apply_reactivity_dim(
          blended.dominance,
          personality.baseline.dominance,
          personality.reactivity
        )
    }
    |> clamp_pad()
  end

  @doc """
  Get a description of the personality for introspection.
  """
  @spec describe(t()) :: String.t()
  def describe(%__MODULE__{} = personality) do
    trait_str =
      personality.traits
      |> Enum.map(&Atom.to_string/1)
      |> Enum.join(", ")

    valence =
      if personality.baseline.pleasure > 0,
        do: Gettext.dgettext(Viva.Gettext, "default", "personality.valence.positive"),
        else: Gettext.dgettext(Viva.Gettext, "default", "personality.valence.neutral")

    energy =
      if personality.baseline.arousal > 0.1,
        do: Gettext.dgettext(Viva.Gettext, "default", "personality.energy.energetic"),
        else: Gettext.dgettext(Viva.Gettext, "default", "personality.energy.calm")

    Gettext.dgettext(Viva.Gettext, "default", "personality.describe", %{
      traits: trait_str,
      valence: valence,
      energy: energy,
      reactivity: Float.round(personality.reactivity, 2)
    })
  end

  @doc """
  Get the neutral PAD state (0, 0, 0).
  """
  @spec neutral_pad() :: pad()
  def neutral_pad, do: %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

  # ============================================================
  # Private Functions
  # ============================================================

  defp load_from_redis do
    try do
      case Redix.command(:redix, ["GET", @persistence_key]) do
        {:ok, nil} ->
          {:error, :not_found}

        {:ok, json} ->
          data = Jason.decode!(json)

          personality = %__MODULE__{
            baseline: %{
              pleasure: data["baseline"]["pleasure"],
              arousal: data["baseline"]["arousal"],
              dominance: data["baseline"]["dominance"]
            },
            reactivity: data["reactivity"],
            volatility: data["volatility"],
            traits: Enum.map(data["traits"], &String.to_atom/1),
            last_adapted: parse_datetime(data["last_adapted"])
          }

          {:ok, personality}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e ->
        VivaLog.debug(:personality, :redis_load_error, error: inspect(e))
        {:error, e}
    end
  end

  defp save_to_redis(%__MODULE__{} = personality) do
    try do
      data = %{
        "baseline" => %{
          "pleasure" => personality.baseline.pleasure,
          "arousal" => personality.baseline.arousal,
          "dominance" => personality.baseline.dominance
        },
        "reactivity" => personality.reactivity,
        "volatility" => personality.volatility,
        "traits" => Enum.map(personality.traits, &Atom.to_string/1),
        "last_adapted" =>
          if(personality.last_adapted,
            do: DateTime.to_iso8601(personality.last_adapted),
            else: nil
          )
      }

      case Redix.command(:redix, ["SET", @persistence_key, Jason.encode!(data)]) do
        {:ok, "OK"} -> :ok
        error -> error
      end
    rescue
      e -> {:error, e}
    end
  end

  defp parse_datetime(nil), do: nil

  defp parse_datetime(iso_string) do
    case DateTime.from_iso8601(iso_string) do
      {:ok, dt, _} -> dt
      _ -> nil
    end
  end

  defp aggregate_experiences(experiences) do
    count = length(experiences)

    total =
      Enum.reduce(experiences, neutral_pad(), fn exp, acc ->
        pad = Map.get(exp, :pad, neutral_pad())
        intensity = Map.get(exp, :intensity, 1.0)

        %{
          pleasure: acc.pleasure + get_value(pad, :pleasure) * intensity,
          arousal: acc.arousal + get_value(pad, :arousal) * intensity,
          dominance: acc.dominance + get_value(pad, :dominance) * intensity
        }
      end)

    %{
      pleasure: total.pleasure / count,
      arousal: total.arousal / count,
      dominance: total.dominance / count
    }
  end

  defp shift_baseline(current, target, alpha) do
    %{
      pleasure: current.pleasure + alpha * (target.pleasure - current.pleasure),
      arousal: current.arousal + alpha * (target.arousal - current.arousal),
      dominance: current.dominance + alpha * (target.dominance - current.dominance)
    }
  end

  defp adjust_reactivity(current_reactivity, experiences) do
    # Calculate emotional variance from experiences
    pads = Enum.map(experiences, fn exp -> Map.get(exp, :pad, neutral_pad()) end)

    if length(pads) < 2 do
      current_reactivity
    else
      # High variance → increase reactivity slightly
      # Low variance → decrease reactivity slightly
      variance = calculate_pad_variance(pads)
      # Small adjustment based on variance
      adjustment = (variance - 0.1) * 0.1

      # Clamp reactivity to reasonable range [0.5, 2.0]
      (current_reactivity + adjustment)
      |> max(0.5)
      |> min(2.0)
    end
  end

  defp calculate_pad_variance(pads) do
    count = length(pads)

    mean =
      Enum.reduce(pads, neutral_pad(), fn pad, acc ->
        %{
          pleasure: acc.pleasure + get_value(pad, :pleasure) / count,
          arousal: acc.arousal + get_value(pad, :arousal) / count,
          dominance: acc.dominance + get_value(pad, :dominance) / count
        }
      end)

    variance =
      Enum.reduce(pads, 0.0, fn pad, acc ->
        dp = get_value(pad, :pleasure) - mean.pleasure
        da = get_value(pad, :arousal) - mean.arousal
        dd = get_value(pad, :dominance) - mean.dominance
        acc + (dp * dp + da * da + dd * dd) / 3.0
      end)

    variance / count
  end

  defp infer_traits(baseline, _experiences) do
    traits = []

    # Infer traits from baseline PAD
    traits = if baseline.pleasure > 0.15, do: [:optimistic | traits], else: traits
    traits = if baseline.pleasure < -0.15, do: [:melancholic | traits], else: traits
    traits = if baseline.arousal > 0.1, do: [:energetic | traits], else: traits
    traits = if baseline.arousal < -0.1, do: [:calm | traits], else: traits
    traits = if baseline.dominance > 0.15, do: [:assertive | traits], else: traits
    traits = if baseline.dominance < -0.15, do: [:submissive | traits], else: traits

    # Default traits if none inferred
    if Enum.empty?(traits), do: [:balanced], else: traits
  end

  defp apply_reactivity_dim(value, baseline, reactivity) do
    deviation = value - baseline
    baseline + deviation * reactivity
  end

  defp clamp_pad(pad) do
    %{
      pleasure: clamp(get_value(pad, :pleasure), -1.0, 1.0),
      arousal: clamp(get_value(pad, :arousal), -1.0, 1.0),
      dominance: clamp(get_value(pad, :dominance), -1.0, 1.0)
    }
  end

  defp clamp(value, min_val, max_val) do
    value |> max(min_val) |> min(max_val)
  end

  defp get_value(pad, key) when is_map(pad) do
    Map.get(pad, key) || Map.get(pad, Atom.to_string(key)) || 0.0
  end
end
