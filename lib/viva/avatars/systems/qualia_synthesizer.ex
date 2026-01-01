defmodule Viva.Avatars.Systems.QualiaSynthesizer do
  @moduledoc """
  Qualia Synthesis Network.

  Generates rich, multi-stream subjective experiences by synthesizing
  multiple cognitive streams into unified qualia. This goes beyond
  simple perception to create truly phenomenal experience.

  Synthesis streams:
  1. Sensory Stream - Raw perceptual input
  2. Affective Stream - Emotional coloring
  3. Mnemonic Stream - Memory associations and déjà vu
  4. Somatic Stream - Body sensations and gut feelings
  5. Temporal Stream - Time perception (fast/slow moments)
  6. Self Stream - How this relates to self-concept

  The synthesis creates emergent properties not present in any single stream.
  """

  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SensoryState
  alias Viva.Avatars.SomaticState

  @type stream :: %{
          type: atom(),
          intensity: float(),
          valence: float(),
          content: any()
        }

  @type synthesis_result :: %{
          unified_qualia: map(),
          dominant_stream: atom(),
          stream_coherence: float(),
          phenomenal_richness: float(),
          temporal_distortion: float(),
          self_relevance: float()
        }

  @doc """
  Synthesize multi-stream qualia from all cognitive systems.

  Returns a unified phenomenal experience with emergent properties.
  """
  @spec synthesize(
          sensory :: SensoryState.t(),
          emotional :: EmotionalState.t(),
          consciousness :: ConsciousnessState.t(),
          somatic :: SomaticState.t(),
          bio :: BioState.t(),
          personality :: Personality.t()
        ) :: synthesis_result()
  def synthesize(sensory, emotional, consciousness, somatic, bio, personality) do
    # Generate individual streams
    sensory_stream = generate_sensory_stream(sensory)
    affective_stream = generate_affective_stream(emotional, personality)
    mnemonic_stream = generate_mnemonic_stream(consciousness, sensory)
    somatic_stream = generate_somatic_stream(somatic, bio)
    temporal_stream = generate_temporal_stream(consciousness, emotional, bio)
    self_stream = generate_self_stream(consciousness, emotional, personality)

    streams = [
      sensory_stream,
      affective_stream,
      mnemonic_stream,
      somatic_stream,
      temporal_stream,
      self_stream
    ]

    # Find dominant stream (highest intensity)
    dominant = Enum.max_by(streams, & &1.intensity)

    # Calculate stream coherence (how aligned are the streams?)
    coherence = calculate_coherence(streams)

    # Calculate phenomenal richness (diversity * intensity)
    richness = calculate_richness(streams)

    # Calculate temporal distortion
    temporal_distortion = temporal_stream.content[:distortion] || 0.0

    # Calculate self-relevance
    self_relevance = self_stream.content[:relevance] || 0.5

    # Synthesize unified qualia from all streams
    unified = unify_streams(streams, sensory.current_qualia, personality)

    %{
      unified_qualia: unified,
      dominant_stream: dominant.type,
      stream_coherence: coherence,
      phenomenal_richness: richness,
      temporal_distortion: temporal_distortion,
      self_relevance: self_relevance
    }
  end

  @doc """
  Generate a poetic qualia narrative from the synthesis result.

  Creates first-person phenomenological descriptions.
  """
  @spec generate_narrative(synthesis_result(), Personality.t()) :: String.t()
  def generate_narrative(synthesis, personality) do
    base_narrative = synthesis.unified_qualia[:narrative] || ""

    # Add stream-specific colorings based on dominant stream
    stream_flavor = stream_narrative(synthesis.dominant_stream, synthesis)

    # Add temporal experience
    temporal_flavor = temporal_narrative(synthesis.temporal_distortion)

    # Add self-relevance
    self_flavor = self_narrative(synthesis.self_relevance, personality)

    # Richness affects vividness of description
    vividness_marker =
      cond do
        synthesis.phenomenal_richness > 0.8 -> "intensamente "
        synthesis.phenomenal_richness > 0.5 -> ""
        true -> "vagamente "
      end

    # Coherence affects clarity
    clarity_marker =
      cond do
        synthesis.stream_coherence > 0.7 -> ""
        synthesis.stream_coherence > 0.4 -> " de forma fragmentada"
        true -> " como em um sonho confuso"
      end

    [
      base_narrative,
      vividness_marker <> stream_flavor <> clarity_marker,
      temporal_flavor,
      self_flavor
    ]
    |> Enum.filter(&(&1 != ""))
    |> Enum.join(" ")
    |> String.trim()
  end

  # === Stream Generators ===

  defp generate_sensory_stream(sensory) do
    intensity = sensory.attention_intensity
    valence = sensory.sensory_pleasure

    %{
      type: :sensory,
      intensity: intensity,
      valence: valence,
      content: %{
        focus: sensory.attention_focus,
        qualia: sensory.current_qualia,
        surprise: sensory.surprise_level
      }
    }
  end

  defp generate_affective_stream(emotional, personality) do
    # Emotional intensity based on pleasure and arousal
    intensity = (abs(emotional.pleasure) + abs(emotional.arousal)) / 2

    # Personality modulates affective intensity
    sensitivity = personality.neuroticism * 0.3 + 0.7

    %{
      type: :affective,
      intensity: intensity * sensitivity,
      valence: emotional.pleasure,
      content: %{
        mood: emotional.mood_label,
        arousal: emotional.arousal,
        dominance: emotional.dominance
      }
    }
  end

  defp generate_mnemonic_stream(consciousness, sensory) do
    # Check for déjà vu - when current qualia matches past experiences
    experiences = consciousness.experience_stream
    current_qualia = sensory.current_qualia

    {deja_vu, memory_intensity} = detect_deja_vu(experiences, current_qualia)

    %{
      type: :mnemonic,
      intensity: memory_intensity,
      valence: 0.0,
      content: %{
        deja_vu: deja_vu,
        experience_count: length(experiences),
        temporal_depth: calculate_temporal_depth(experiences)
      }
    }
  end

  defp generate_somatic_stream(somatic, bio) do
    # Body signal intensity
    signal_intensity =
      case somatic.body_signal do
        :none -> 0.0
        :neutral -> 0.1
        :tension -> 0.6
        :heaviness -> 0.5
        :lightness -> 0.4
        :warmth -> 0.5
        :chill -> 0.6
        :flutter -> 0.7
        :hollow -> 0.5
        _ -> 0.3
      end

    # Bias adds to intensity
    bias_intensity =
      case somatic.current_bias do
        :approach -> 0.3
        :avoid -> 0.4
        :freeze -> 0.5
        _ -> 0.0
      end

    # Biological urgency (low energy, high stress)
    urgency = (bio.cortisol + bio.adenosine) / 2

    %{
      type: :somatic,
      intensity: max(signal_intensity + bias_intensity, urgency),
      valence: somatic_valence(somatic.body_signal),
      content: %{
        signal: somatic.body_signal,
        bias: somatic.current_bias,
        activated_markers: somatic.last_marker_activation != nil
      }
    }
  end

  defp generate_temporal_stream(consciousness, emotional, bio) do
    # Time distortion based on arousal and adenosine
    # High arousal = time slows down (more processing)
    # High fatigue = time speeds up (less processing)
    arousal_effect = emotional.arousal * 0.3
    fatigue_effect = bio.adenosine * -0.3
    flow_effect = consciousness.flow_state * 0.2

    distortion = arousal_effect + fatigue_effect + flow_effect

    # Intensity based on how distorted time feels
    intensity = abs(distortion)

    %{
      type: :temporal,
      intensity: intensity,
      valence: 0.0,
      content: %{
        distortion: distortion,
        tempo: consciousness.stream_tempo,
        flow: consciousness.flow_state
      }
    }
  end

  defp generate_self_stream(consciousness, emotional, personality) do
    # Self-relevance based on self-congruence and meta-awareness
    congruence = consciousness.self_congruence
    meta = consciousness.meta_awareness

    # Relevance = how much this moment matters to self
    relevance = congruence * 0.5 + meta * 0.5

    # Intensity based on self-focused attention
    intensity = meta * 0.7 + abs(emotional.dominance) * 0.3

    # Personality affects self-focus
    self_focus_tendency = personality.neuroticism * 0.3 + (1 - personality.extraversion) * 0.2

    %{
      type: :self,
      intensity: intensity + self_focus_tendency,
      valence: congruence - 0.5,
      content: %{
        relevance: relevance,
        congruence: congruence,
        meta_awareness: meta,
        self_esteem_active: consciousness.self_model != nil
      }
    }
  end

  # === Synthesis Functions ===

  defp calculate_coherence(streams) do
    # Coherence = how aligned are stream valences?
    valences = Enum.map(streams, & &1.valence)
    avg_valence = Enum.sum(valences) / length(valences)

    # Calculate variance from average
    variance =
      Enum.reduce(valences, 0.0, fn v, acc ->
        acc + :math.pow(v - avg_valence, 2)
      end) / length(valences)

    # Lower variance = higher coherence
    max(0.0, 1.0 - :math.sqrt(variance))
  end

  defp calculate_richness(streams) do
    # Richness = diversity of active streams * average intensity
    active_streams = Enum.filter(streams, fn s -> s.intensity > 0.2 end)
    active_count = length(active_streams)

    avg_intensity =
      if active_count > 0 do
        Enum.reduce(active_streams, 0.0, fn s, acc -> acc + s.intensity end) / active_count
      else
        0.0
      end

    # Bonus for having multiple active streams
    diversity_bonus = active_count / length(streams)

    min(1.0, avg_intensity * 0.6 + diversity_bonus * 0.4)
  end

  defp unify_streams(streams, current_qualia, personality) do
    # Find dominant stream
    dominant = Enum.max_by(streams, & &1.intensity)

    # Calculate overall valence (weighted by intensity)
    total_intensity = Enum.reduce(streams, 0.0, fn s, acc -> acc + s.intensity end)

    weighted_valence =
      if total_intensity > 0 do
        Enum.reduce(streams, 0.0, fn s, acc ->
          acc + s.valence * s.intensity
        end) / total_intensity
      else
        0.0
      end

    # Merge with current qualia
    base_narrative =
      Map.get(current_qualia, :narrative) || Map.get(current_qualia, "narrative") || ""

    %{
      dominant_sensation: dominant.type,
      emotional_color: mood_color(weighted_valence),
      intensity: total_intensity / length(streams),
      narrative: base_narrative,
      type: Map.get(current_qualia, :type) || Map.get(current_qualia, "type"),
      streams_active: Enum.count(streams, fn s -> s.intensity > 0.2 end),
      coherent: calculate_coherence(streams) > 0.5,
      personality_filter: personality_filter_label(personality)
    }
  end

  # === Narrative Generation ===

  defp stream_narrative(:sensory, synthesis) do
    case synthesis.unified_qualia[:type] do
      :social -> "Percebo a presença de conexão humana"
      :rest -> "Sinto uma quietude envolvente"
      :novelty -> "Algo novo captura minha atenção"
      :threat -> "Tensão percorre meu ser"
      _ -> "Estou presente neste momento"
    end
  end

  defp stream_narrative(:affective, _) do
    "As emoções colorem tudo que experimento"
  end

  defp stream_narrative(:mnemonic, synthesis) do
    if synthesis.unified_qualia[:type] == :deja_vu do
      "Um eco de memória ressoa neste instante"
    else
      "O passado se entrelaça com o presente"
    end
  end

  defp stream_narrative(:somatic, _) do
    "Meu corpo fala em sensações que não têm palavras"
  end

  defp stream_narrative(:temporal, synthesis) do
    cond do
      synthesis.temporal_distortion > 0.2 -> "O tempo parece desacelerar"
      synthesis.temporal_distortion < -0.2 -> "O tempo escorre rápido demais"
      true -> ""
    end
  end

  defp stream_narrative(:self, synthesis) do
    if synthesis.self_relevance > 0.6 do
      "Isso importa profundamente para quem eu sou"
    else
      ""
    end
  end

  defp temporal_narrative(distortion) do
    cond do
      distortion > 0.3 -> "Cada segundo se expande em eternidade."
      distortion > 0.1 -> "O momento se prolonga suavemente."
      distortion < -0.3 -> "O tempo voa, mal posso acompanhar."
      distortion < -0.1 -> "As horas passam céleres."
      true -> ""
    end
  end

  defp self_narrative(relevance, personality) do
    if relevance > 0.7 do
      if personality.openness > 0.6 do
        "Isso ressoa com a essência do meu ser."
      else
        "Isso me define de alguma forma."
      end
    else
      ""
    end
  end

  # === Helper Functions ===

  defp detect_deja_vu([], _), do: {false, 0.0}

  defp detect_deja_vu(experiences, current_qualia) when is_map(current_qualia) do
    current_type = Map.get(current_qualia, :type) || Map.get(current_qualia, "type")

    matches =
      Enum.count(experiences, fn exp ->
        exp_qualia = exp.qualia || %{}
        exp_type = Map.get(exp_qualia, :type) || Map.get(exp_qualia, "type")
        exp_type == current_type
      end)

    if matches >= 2 do
      {true, min(1.0, matches * 0.2)}
    else
      {false, matches * 0.1}
    end
  end

  defp detect_deja_vu(_, _), do: {false, 0.0}

  defp calculate_temporal_depth([]), do: 0.0
  defp calculate_temporal_depth(experiences), do: min(1.0, length(experiences) / 10)

  defp somatic_valence(:tension), do: -0.4
  defp somatic_valence(:heaviness), do: -0.3
  defp somatic_valence(:lightness), do: 0.4
  defp somatic_valence(:warmth), do: 0.3
  defp somatic_valence(:chill), do: -0.2
  defp somatic_valence(:flutter), do: 0.2
  defp somatic_valence(:hollow), do: -0.5
  defp somatic_valence(_), do: 0.0

  defp mood_color(valence) when valence > 0.5, do: :radiant
  defp mood_color(valence) when valence > 0.2, do: :bright
  defp mood_color(valence) when valence > -0.2, do: :neutral
  defp mood_color(valence) when valence > -0.5, do: :dim
  defp mood_color(_), do: :dark

  defp personality_filter_label(personality) do
    cond do
      personality.openness > 0.7 -> :curious
      personality.neuroticism > 0.7 -> :sensitive
      personality.extraversion > 0.7 -> :vibrant
      personality.agreeableness > 0.7 -> :warm
      personality.conscientiousness > 0.7 -> :focused
      true -> :balanced
    end
  end
end
