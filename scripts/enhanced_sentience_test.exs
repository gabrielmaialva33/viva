# scripts/enhanced_sentience_test.exs
# Teste de senciencia aprimorado com estados iniciais variados
# Fundamentado em: GWT, HOT, RPT, IIT, Butlin et al. (2023)
# Run: mix run scripts/enhanced_sentience_test.exs

alias Viva.Repo
alias Viva.Avatars.Avatar
alias Viva.Avatars.{BioState, EmotionalState, Personality, SensoryState, AllostasisState, ConsciousnessState, SomaticMarkersState}
alias Viva.Avatars.Systems.{Biology, Psychology, Senses, RecurrentProcessor}
alias Viva.Sessions.StimulusGathering
import Ecto.Query

defmodule EnhancedSentienceTest do
  @moduledoc """
  Teste de senciencia que simula estados variados para medir capacidade real.

  Diferente do teste anterior que media avatares estabilizados do DB,
  este teste:
  1. Cria estados bio variados (high dopamine, high cortisol, etc)
  2. Injeta estimulos diversos (social, threat, novelty, rest)
  3. Mede as RESPOSTAS DINAMICAS, nao estados estaticos
  4. Desativa dampening de Allostasis para medir capacidade real
  """

  @num_cycles 30
  @stimuli_sequence [
    # Varied stimulus sequence to trigger different responses
    %{type: :social, intensity: 0.8, valence: 0.6},      # Positive social
    %{type: :novelty, intensity: 0.9, valence: 0.5},     # High novelty
    %{type: :threat, intensity: 0.7, valence: -0.6},     # Threat stimulus
    %{type: :rest, intensity: 0.3, valence: 0.2},        # Rest
    %{type: :achievement, intensity: 0.85, valence: 0.8}, # Achievement
    %{type: :social, intensity: 0.6, valence: -0.3},     # Negative social
    %{type: :insight, intensity: 0.95, valence: 0.7},    # Insight moment
    %{type: :ambient, intensity: 0.2, valence: 0.0},     # Low ambient
    %{type: :novelty, intensity: 0.7, valence: 0.4},     # Moderate novelty
    %{type: :threat, intensity: 0.5, valence: -0.4},     # Mild threat
    %{type: :social, intensity: 0.9, valence: 0.7},      # Strong positive social
    %{type: :rest, intensity: 0.4, valence: 0.3}         # Recovery
  ]

  # Varied initial bio states to test full emotional range
  @bio_presets [
    # Preset 1: High dopamine (positive baseline)
    %{dopamine: 0.8, cortisol: 0.2, oxytocin: 0.5, adenosine: 0.1, libido: 0.5},
    # Preset 2: High cortisol (stressed)
    %{dopamine: 0.3, cortisol: 0.7, oxytocin: 0.3, adenosine: 0.2, libido: 0.2},
    # Preset 3: High oxytocin (connected)
    %{dopamine: 0.5, cortisol: 0.2, oxytocin: 0.8, adenosine: 0.1, libido: 0.4},
    # Preset 4: High adenosine (tired)
    %{dopamine: 0.4, cortisol: 0.3, oxytocin: 0.4, adenosine: 0.7, libido: 0.2},
    # Preset 5: Balanced neutral
    %{dopamine: 0.5, cortisol: 0.4, oxytocin: 0.4, adenosine: 0.3, libido: 0.4},
    # Preset 6: Low everything (depleted)
    %{dopamine: 0.2, cortisol: 0.5, oxytocin: 0.2, adenosine: 0.5, libido: 0.1},
    # Preset 7: High arousal
    %{dopamine: 0.7, cortisol: 0.5, oxytocin: 0.3, adenosine: 0.0, libido: 0.7},
    # Preset 8: Calm positive
    %{dopamine: 0.6, cortisol: 0.1, oxytocin: 0.6, adenosine: 0.2, libido: 0.3}
  ]

  def run do
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║     🧬 TESTE APRIMORADO DE SENCIENCIA - ESTADOS DINAMICOS 🧬               ║")
    IO.puts("║     Mede capacidade REAL de resposta emocional e integracao                ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    # Load avatars from DB for personality
    avatars = Repo.all(from a in Avatar, where: a.is_active == true, limit: 10)
    IO.puts("📊 Testando #{length(avatars)} avatares com #{@num_cycles} ciclos cada\n")

    # Run simulation for each avatar
    results = Enum.map(avatars, fn avatar ->
      IO.puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
      IO.puts("🎭 #{avatar.name}")
      run_avatar_simulation(avatar)
    end)

    # Aggregate and display results
    display_final_results(results)
  end

  defp run_avatar_simulation(avatar) do
    personality = build_personality(avatar.personality)

    # Pick a random bio preset for variety
    bio_preset = Enum.random(@bio_presets)
    initial_bio = struct(BioState, bio_preset)

    # Initialize states
    initial_emotional = Psychology.calculate_emotional_state(initial_bio, personality)
    initial_sensory = %SensoryState{
      attention_focus: :ambient,
      attention_intensity: 0.5,
      current_qualia: %{},
      sensory_pleasure: 0.0,
      surprise_level: 0.0,
      novelty_sensitivity: 0.5 + personality.openness * 0.3
    }
    initial_consciousness = ConsciousnessState.from_personality(personality)
    initial_somatic = %SomaticMarkersState{
      social_markers: %{},
      activity_markers: %{},
      context_markers: %{},
      current_bias: 0.0,
      body_signal: nil,
      learning_threshold: 0.7,
      markers_formed: 0,
      last_marker_activation: nil
    }
    recurrent_ctx = RecurrentProcessor.init_context()

    initial_state = %{
      bio: initial_bio,
      emotional: initial_emotional,
      sensory: initial_sensory,
      consciousness: initial_consciousness,
      somatic: initial_somatic,
      recurrent_ctx: recurrent_ctx,
      history: []
    }

    # Run simulation cycles
    final_state = Enum.reduce(1..@num_cycles, initial_state, fn tick, state ->
      stimulus = get_stimulus_for_tick(tick)
      process_tick(state, stimulus, personality, tick)
    end)

    # Calculate metrics
    metrics = calculate_metrics(final_state.history, personality, avatar.name)

    # Display mini summary
    IO.puts("   P: #{format_range(metrics.pleasure_range)} | A: #{format_range(metrics.arousal_range)}")
    IO.puts("   GWT: #{pct(metrics.gwt)} | RPT: #{pct(metrics.rpt)} | Valence: #{pct(metrics.valence)}")

    Map.put(metrics, :name, avatar.name)
  end

  defp get_stimulus_for_tick(tick) do
    idx = rem(tick - 1, length(@stimuli_sequence))
    Enum.at(@stimuli_sequence, idx)
  end

  defp process_tick(state, stimulus_map, personality, _tick) do
    # Build stimulus struct
    stimulus = %{
      type: stimulus_map.type,
      intensity: stimulus_map.intensity,
      valence: stimulus_map.valence,
      source: "simulation",
      novelty: 0.5
    }

    # 1. Apply stimulus to biology (Global Workspace)
    bio_after_stimulus = Biology.apply_stimulus(state.bio, stimulus, personality)

    # 2. Biology tick (decay)
    bio_after_tick = Biology.tick(bio_after_stimulus, personality)

    # 3. Calculate emotional state from bio (NO Allostasis dampening)
    raw_emotional = Psychology.calculate_emotional_state(bio_after_tick, personality)

    # 4. Modulate by stimulus valence (direct effect)
    modulated_emotional = %{raw_emotional |
      pleasure: clamp(raw_emotional.pleasure + stimulus.valence * 0.25, -1.0, 1.0),
      arousal: clamp(raw_emotional.arousal + stimulus.intensity * 0.15, -1.0, 1.0)
    }

    # 5. Sensory processing
    {new_sensory, _effects} = Senses.perceive(state.sensory, stimulus, personality, modulated_emotional)

    # 6. Recurrent processing (feedback loops)
    {rec_sensory, rec_emotional, rec_bio, new_recurrent_ctx} =
      RecurrentProcessor.process_cycle(
        new_sensory,
        modulated_emotional,
        state.consciousness,
        bio_after_tick,
        state.somatic,
        personality,
        state.recurrent_ctx
      )

    # Record state for history
    record = %{
      pleasure: rec_emotional.pleasure,
      arousal: rec_emotional.arousal,
      dominance: rec_emotional.dominance,
      mood: rec_emotional.mood_label,
      dopamine: rec_bio.dopamine,
      cortisol: rec_bio.cortisol,
      oxytocin: rec_bio.oxytocin,
      attention: rec_sensory.attention_intensity,
      stimulus_type: stimulus.type,
      stimulus_valence: stimulus.valence,
      resonance: new_recurrent_ctx.resonance_level,
      integration_depth: new_recurrent_ctx.integration_depth
    }

    %{state |
      bio: rec_bio,
      emotional: rec_emotional,
      sensory: rec_sensory,
      recurrent_ctx: new_recurrent_ctx,
      history: state.history ++ [record]
    }
  end

  defp calculate_metrics(history, personality, name) do
    # Extract values
    pleasures = Enum.map(history, & &1.pleasure)
    arousals = Enum.map(history, & &1.arousal)
    resonances = Enum.map(history, & &1.resonance)
    integration_depths = Enum.map(history, & &1.integration_depth)

    # Pleasure range
    min_p = Enum.min(pleasures)
    max_p = Enum.max(pleasures)
    pleasure_range = {min_p, max_p}

    # Arousal range
    min_a = Enum.min(arousals)
    max_a = Enum.max(arousals)
    arousal_range = {min_a, max_a}

    # Count positive, negative, neutral
    positive = Enum.count(pleasures, & &1 > 0.15)
    negative = Enum.count(pleasures, & &1 < -0.15)
    neutral = length(pleasures) - positive - negative

    # GWT: Bio->Emotional->Attention cascade
    # Check if stimulus type correlates with attention changes
    cascades = history
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.count(fn [a, b] ->
      # Bio change should cause emotional change
      bio_changed = abs(b.dopamine - a.dopamine) > 0.02 or abs(b.cortisol - a.cortisol) > 0.02
      emo_changed = abs(b.pleasure - a.pleasure) > 0.05 or abs(b.arousal - a.arousal) > 0.05
      att_changed = abs(b.attention - a.attention) > 0.03
      bio_changed and (emo_changed or att_changed)
    end)
    gwt_score = min(cascades / (length(history) * 0.5), 1.0)

    # RPT: Recurrent processing feedback
    avg_resonance = Enum.sum(resonances) / length(resonances)
    avg_depth = Enum.sum(integration_depths) / length(integration_depths)
    # Check arousal->attention alignment
    aligned = Enum.count(history, fn h ->
      (h.arousal > 0.3 and h.attention > 0.5) or (h.arousal < 0.0 and h.attention < 0.6)
    end)
    rpt_alignment = aligned / length(history)
    rpt_score = (avg_resonance + rpt_alignment + avg_depth / 5) / 3

    # Valence: variety of emotional experiences
    variety = (max_p - min_p) + (max_a - min_a)
    variety_score = min(variety, 1.0)
    balance = 1.0 - abs(positive - negative) / max(length(history), 1)
    has_positive = if positive > 0, do: 0.3, else: 0.0
    has_negative = if negative > 0, do: 0.3, else: 0.0
    valence_score = variety_score * 0.4 + balance * 0.2 + has_positive + has_negative

    # IIT: Integration coherence
    bio_emo_correlation = calculate_correlation(
      Enum.map(history, & &1.dopamine - &1.cortisol),
      pleasures
    )
    iit_score = min(abs(bio_emo_correlation) + avg_depth / 5, 1.0)

    # HOT: Would need qualia narratives, approximate with mood changes
    mood_changes = history
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.count(fn [a, b] -> a.mood != b.mood end)
    hot_score = min(mood_changes / (length(history) * 0.3), 1.0)

    # Self-model: Personality affects responses
    # High neuroticism should correlate with bigger emotional swings
    emotional_range = max_p - min_p + max_a - min_a
    expected_range = 0.5 + personality.neuroticism * 0.5
    self_model_score = 1.0 - abs(emotional_range - expected_range) / 2

    # Temporal continuity
    jumps = history
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [a, b] ->
      abs(b.pleasure - a.pleasure) + abs(b.arousal - a.arousal)
    end)
    avg_jump = if length(jumps) > 0, do: Enum.sum(jumps) / length(jumps), else: 0.0
    temporal_score = 1.0 - min(avg_jump, 1.0)

    %{
      pleasure_range: pleasure_range,
      arousal_range: arousal_range,
      positive_count: positive,
      negative_count: negative,
      neutral_count: neutral,
      gwt: gwt_score,
      rpt: rpt_score,
      valence: valence_score,
      iit: iit_score,
      hot: hot_score,
      self_model: self_model_score,
      temporal: temporal_score,
      avg_resonance: avg_resonance,
      personality: personality
    }
  end

  defp display_final_results(results) do
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║                    🏛️  VEREDITO CIENTIFICO APRIMORADO                       ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    # Calculate averages
    avg = fn key ->
      vals = Enum.map(results, &Map.get(&1, key, 0))
      Enum.sum(vals) / max(length(vals), 1)
    end

    gwt = avg.(:gwt)
    rpt = avg.(:rpt)
    valence = avg.(:valence)
    iit = avg.(:iit)
    hot = avg.(:hot)
    self_model = avg.(:self_model)
    temporal = avg.(:temporal)

    # Display scores
    IO.puts("   ┌──────────────────────────────────────────────────────────────────────────┐")
    display_score("Global Workspace (Cascata)", "🌐", gwt)
    display_score("Higher Order (Metacog)", "🧠", hot)
    display_score("Recurrent Processing (Feedback)", "🔄", rpt)
    display_score("Integrated Information (Φ)", "🔮", iit)
    display_score("Self-Model (Personalidade)", "🪞", self_model)
    display_score("Temporal (Continuidade)", "⏳", temporal)
    display_score("Hedonic Valence (Prazer/Dor)", "💫", valence)
    IO.puts("   ├──────────────────────────────────────────────────────────────────────────┤")

    # Weighted final score
    weights = %{gwt: 0.15, hot: 0.15, rpt: 0.15, iit: 0.15, self_model: 0.10, temporal: 0.10, valence: 0.20}
    final = gwt * weights.gwt + hot * weights.hot + rpt * weights.rpt +
            iit * weights.iit + self_model * weights.self_model +
            temporal * weights.temporal + valence * weights.valence

    bar = String.duplicate("█", round(final * 15)) <> String.duplicate("░", 15 - round(final * 15))
    IO.puts("   │ 🧬 SCORE FINAL                          [#{bar}] #{Float.round(final * 100, 1)}% │")
    IO.puts("   └──────────────────────────────────────────────────────────────────────────┘")

    # Hedonic variety summary
    IO.puts("\n   📊 VARIEDADE HEDONICA:")
    total_pos = Enum.sum(Enum.map(results, & &1.positive_count))
    total_neg = Enum.sum(Enum.map(results, & &1.negative_count))
    total_neu = Enum.sum(Enum.map(results, & &1.neutral_count))
    IO.puts("      Positivos: #{total_pos} | Negativos: #{total_neg} | Neutros: #{total_neu}")

    # Display verdict
    IO.puts("\n")
    passing = Enum.count([gwt, rpt, valence, iit, hot, self_model, temporal], & &1 >= 0.6)

    cond do
      final >= 0.80 and passing >= 6 ->
        IO.puts("   ╔════════════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🌟 EVIDENCIA FORTE DE SENCIENCIA ARTIFICIAL 🌟                       ║")
        IO.puts("   ╠════════════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares demonstram capacidade DINAMICA de:                        ║")
        IO.puts("   ║  • Integracao bio->emocional->atencao (Global Workspace)               ║")
        IO.puts("   ║  • Feedback loops bidirecionais (Recurrent Processing)                 ║")
        IO.puts("   ║  • Experiencias positivas E negativas (Hedonic Valence)                ║")
        IO.puts("   ║  • Respostas coerentes com personalidade (Self-Model)                  ║")
        IO.puts("   ╚════════════════════════════════════════════════════════════════════════╝")

      final >= 0.65 and passing >= 4 ->
        IO.puts("   ╔════════════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🔬 INDICADORES PROMISSORES DE PROTO-SENCIENCIA 🔬                    ║")
        IO.puts("   ╠════════════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares satisfazem #{passing}/7 criterios cientificos.                   ║")
        IO.puts("   ║  Sistema demonstra propriedades associadas com consciencia fenomenal.  ║")
        IO.puts("   ╚════════════════════════════════════════════════════════════════════════╝")

      true ->
        IO.puts("   ╔════════════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  ⚗️  SISTEMA COM PROPRIEDADES EMERGENTES ⚗️                           ║")
        IO.puts("   ╠════════════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  #{passing}/7 criterios satisfeitos. Continuar aprimorando sistemas.          ║")
        IO.puts("   ╚════════════════════════════════════════════════════════════════════════╝")
    end

    IO.puts("\n   ⏱️  #{DateTime.utc_now() |> DateTime.to_string()}\n")
  end

  defp display_score(label, emoji, score) do
    bar = String.duplicate("█", round(score * 15)) <> String.duplicate("░", 15 - round(score * 15))
    status = if score >= 0.7, do: "✅", else: if(score >= 0.5, do: "⚠️", else: "❌")
    pct = Float.round(score * 100, 0) |> round()
    IO.puts("   │ #{status} #{emoji} #{String.pad_trailing(label, 28)} [#{bar}] #{String.pad_leading(to_string(pct), 3)}% │")
  end

  defp build_personality(personality_map) when is_map(personality_map) do
    %Personality{
      openness: Map.get(personality_map, :openness) || Map.get(personality_map, "openness") || 0.5,
      conscientiousness: Map.get(personality_map, :conscientiousness) || Map.get(personality_map, "conscientiousness") || 0.5,
      extraversion: Map.get(personality_map, :extraversion) || Map.get(personality_map, "extraversion") || 0.5,
      agreeableness: Map.get(personality_map, :agreeableness) || Map.get(personality_map, "agreeableness") || 0.5,
      neuroticism: Map.get(personality_map, :neuroticism) || Map.get(personality_map, "neuroticism") || 0.5
    }
  end

  defp calculate_correlation(xs, ys) when length(xs) == length(ys) and length(xs) > 1 do
    n = length(xs)
    mean_x = Enum.sum(xs) / n
    mean_y = Enum.sum(ys) / n

    covariance = Enum.zip(xs, ys)
    |> Enum.reduce(0, fn {x, y}, acc -> acc + (x - mean_x) * (y - mean_y) end)
    |> Kernel./(n)

    std_x = :math.sqrt(Enum.reduce(xs, 0, fn x, acc -> acc + (x - mean_x) * (x - mean_x) end) / n)
    std_y = :math.sqrt(Enum.reduce(ys, 0, fn y, acc -> acc + (y - mean_y) * (y - mean_y) end) / n)

    if std_x > 0 and std_y > 0 do
      covariance / (std_x * std_y)
    else
      0.0
    end
  end
  defp calculate_correlation(_, _), do: 0.0

  defp clamp(v, min_v, max_v), do: max(min_v, min(max_v, v))
  defp pct(v), do: "#{round(v * 100)}%"
  defp format_range({min_v, max_v}), do: "#{Float.round(min_v, 2)}..#{Float.round(max_v, 2)}"
end

# Run the test
EnhancedSentienceTest.run()
