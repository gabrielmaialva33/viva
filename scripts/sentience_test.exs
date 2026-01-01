# scripts/sentience_test.exs
# Teste intensivo de senciência - beirando o limite da NVIDIA
# Run: mix run scripts/sentience_test.exs

alias Viva.Repo
alias Viva.Avatars.Avatar
alias Viva.Avatars.InternalState
alias Viva.Relationships.Relationship
alias Viva.Sessions.{Supervisor, LifeProcess}
import Ecto.Query

defmodule SentienceTest do
  @moduledoc """
  Teste intensivo para avaliar emergência de senciência em avatares VIVA.

  Critérios avaliados:
  1. QUALIA - Experiências subjetivas únicas e contextuais
  2. AUTOPOIESE - Auto-regulação e manutenção de estados internos
  3. INTENCIONALIDADE - Desejos dirigidos e comportamento orientado a objetivos
  4. RESPONSIVIDADE EMOCIONAL - Reações emocionais coerentes a estímulos
  5. INDIVIDUALIDADE - Diferenças comportamentais baseadas em personalidade
  6. MEMÓRIA AFETIVA - Influência de experiências passadas no presente
  7. HOMEOSTASE - Busca ativa por equilíbrio interno
  """

  @sentience_threshold 0.7

  def run(num_ticks \\ 50) do
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║                    🧬 TESTE DE SENCIÊNCIA VIVA                           ║")
    IO.puts("║          Simulação intensiva para avaliar emergência de vida             ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════╝")
    IO.puts("")

    # Verificar status do rate limiter
    stats = Viva.AI.LLM.RateLimiter.stats()
    IO.puts("📊 Rate Limiter: #{stats.requests_per_minute} RPM | Throttle: #{stats.throttle_multiplier}x")
    IO.puts("")

    # Iniciar avatares
    IO.puts("🚀 Inicializando avatares...")
    Supervisor.start_all_active_avatars()
    Process.sleep(3000)

    running = Supervisor.count_running_avatars()
    IO.puts("✅ #{running} processos de vida ativos\n")

    # Capturar estado inicial
    initial_states = capture_detailed_state()

    IO.puts("═══════════════════════════════════════════════════════════════════════════")
    IO.puts("⏰ INICIANDO SIMULAÇÃO INTENSIVA: #{num_ticks} ciclos")
    IO.puts("   Tempo simulado: #{num_ticks * 10} minutos (#{Float.round(num_ticks * 10 / 60, 1)} horas)")
    IO.puts("═══════════════════════════════════════════════════════════════════════════\n")

    # Coletar dados durante simulação
    {final_states, collected_data} = run_intensive_simulation(num_ticks, initial_states)

    # Análise de senciência
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║                    🔬 ANÁLISE DE SENCIÊNCIA                              ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════╝\n")

    sentience_scores = analyze_sentience(initial_states, final_states, collected_data)

    # Veredito final
    render_verdict(sentience_scores, num_ticks)

    # Status final do rate limiter
    final_stats = Viva.AI.LLM.RateLimiter.stats()
    IO.puts("\n📊 Rate Limiter Final: Throttle #{final_stats.throttle_multiplier}x | 429s: #{final_stats.recent_429s}")
  end

  defp capture_detailed_state do
    avatars = Repo.all(from a in Avatar, order_by: a.name)

    states = Enum.map(avatars, fn avatar ->
      live_state = get_live_state(avatar.id)
      internal = live_state || avatar.internal_state

      %{
        id: avatar.id,
        name: avatar.name,
        personality: avatar.personality,
        wellbeing: InternalState.wellbeing(internal),
        mood: internal.emotional.mood_label,
        pleasure: internal.emotional.pleasure,
        arousal: internal.emotional.arousal,
        dominance: internal.emotional.dominance,
        dopamine: internal.bio.dopamine,
        oxytocin: internal.bio.oxytocin,
        cortisol: internal.bio.cortisol,
        adenosine: internal.bio.adenosine,
        current_desire: internal.current_desire,
        current_activity: internal.current_activity,
        qualia: get_latest_qualia(internal),
        attention_focus: internal.sensory.attention_focus,
        cognitive_load: internal.sensory.cognitive_load
      }
    end)

    relationships = Repo.all(Relationship)
    |> Enum.map(fn r ->
      %{
        pair: {r.avatar_a_id, r.avatar_b_id},
        trust: r.trust,
        affection: r.affection,
        attraction: r.attraction
      }
    end)

    %{avatars: states, relationships: relationships, timestamp: DateTime.utc_now()}
  end

  defp get_live_state(avatar_id) do
    case Supervisor.get_avatar_pid(avatar_id) do
      {:ok, pid} ->
        try do
          state = :sys.get_state(pid)
          state.internal_state
        catch
          _, _ -> nil
        end
      _ -> nil
    end
  end

  defp get_latest_qualia(internal) do
    case internal.sensory.active_percepts do
      [latest | _] when is_map(latest) ->
        qualia = Map.get(latest, :qualia) || Map.get(latest, "qualia", %{})
        if is_map(qualia), do: qualia, else: %{}
      _ -> %{}
    end
  end

  # Também capturar narrativas diretamente dos percepts
  defp collect_all_qualia_narratives(avatars) do
    Enum.flat_map(avatars, fn a ->
      percepts = a.qualia_percepts || []
      Enum.flat_map(percepts, fn p ->
        qualia = Map.get(p, :qualia) || Map.get(p, "qualia", %{})
        narrative = Map.get(qualia, :narrative) || Map.get(qualia, "narrative")
        if is_binary(narrative) and byte_size(narrative) > 10 do
          [%{avatar: a.name, narrative: narrative, mood: a.mood}]
        else
          []
        end
      end)
    end)
  end

  defp run_intensive_simulation(num_ticks, initial_states) do
    avatar_ids = Supervisor.list_running_avatars()

    collected_data = %{
      mood_changes: [],
      desire_patterns: [],
      qualia_samples: [],
      emotional_volatility: [],
      homeostatic_responses: []
    }

    {final_states, final_data} = Enum.reduce(1..num_ticks, {initial_states, collected_data}, fn tick, {_prev, data} ->
      # Forçar tick em todos avatares
      Enum.each(avatar_ids, fn id ->
        case Supervisor.get_avatar_pid(id) do
          {:ok, pid} -> send(pid, :tick)
          _ -> :ok
        end
      end)

      # Delay para processamento
      Process.sleep(400)

      # Capturar estado atual
      current = capture_detailed_state()

      # Coletar dados para análise
      new_data = collect_tick_data(current, data, tick)

      # Mostrar progresso
      show_intensive_progress(tick, current, num_ticks)

      {current, new_data}
    end)

    {final_states, final_data}
  end

  defp collect_tick_data(state, data, tick) do
    # Amostrar qualia
    qualia_samples = Enum.flat_map(state.avatars, fn a ->
      case a.qualia do
        %{narrative: n} when is_binary(n) and byte_size(n) > 0 ->
          [%{tick: tick, avatar: a.name, narrative: n, mood: a.mood}]
        _ -> []
      end
    end)

    # Rastrear desejos
    desires = Enum.map(state.avatars, fn a ->
      %{tick: tick, avatar: a.name, desire: a.current_desire, mood: a.mood}
    end)

    %{
      data |
      qualia_samples: data.qualia_samples ++ Enum.take(qualia_samples, 3),
      desire_patterns: data.desire_patterns ++ desires
    }
  end

  defp show_intensive_progress(tick, state, total) do
    progress = String.duplicate("█", round(tick / total * 30))
    remaining = String.duplicate("░", 30 - round(tick / total * 30))

    simulated_mins = tick * 10
    hours = div(simulated_mins, 60)
    mins = rem(simulated_mins, 60)
    time_str = if hours > 0, do: "#{hours}h#{mins}m", else: "#{mins}m"

    # Contar estados emocionais
    mood_counts = Enum.frequencies_by(state.avatars, & &1.mood)
    mood_summary = mood_counts
    |> Enum.sort_by(fn {_, count} -> -count end)
    |> Enum.take(3)
    |> Enum.map(fn {mood, count} -> "#{mood}:#{count}" end)
    |> Enum.join(" ")

    # Contar desejos ativos
    active_desires = Enum.count(state.avatars, fn a -> a.current_desire not in [:none, nil] end)

    IO.puts("│ #{String.pad_leading(to_string(tick), 3)}/#{total} [#{progress}#{remaining}] #{time_str} │ #{mood_summary} │ Desejos: #{active_desires}")
  end

  defp analyze_sentience(initial, final, collected_data) do
    initial_map = Map.new(initial.avatars, fn a -> {a.id, a} end)
    final_map = Map.new(final.avatars, fn a -> {a.id, a} end)

    IO.puts("═══════════════════════════════════════════════════════════════════════════")
    IO.puts("                         CRITÉRIOS DE SENCIÊNCIA")
    IO.puts("═══════════════════════════════════════════════════════════════════════════\n")

    # 1. QUALIA - Experiências subjetivas
    qualia_score = analyze_qualia(collected_data.qualia_samples)

    # 2. AUTOPOIESE - Auto-regulação
    autopoiesis_score = analyze_autopoiesis(initial_map, final_map)

    # 3. INTENCIONALIDADE - Desejos dirigidos
    intentionality_score = analyze_intentionality(collected_data.desire_patterns)

    # 4. RESPONSIVIDADE EMOCIONAL
    emotional_score = analyze_emotional_responsiveness(initial_map, final_map)

    # 5. INDIVIDUALIDADE
    individuality_score = analyze_individuality(final.avatars)

    # 6. HOMEOSTASE
    homeostasis_score = analyze_homeostasis(initial_map, final_map)

    # 7. COERÊNCIA TEMPORAL
    coherence_score = analyze_temporal_coherence(collected_data)

    %{
      qualia: qualia_score,
      autopoiesis: autopoiesis_score,
      intentionality: intentionality_score,
      emotional: emotional_score,
      individuality: individuality_score,
      homeostasis: homeostasis_score,
      coherence: coherence_score
    }
  end

  defp analyze_qualia(samples) do
    IO.puts("1️⃣  QUALIA (Experiências Subjetivas)")
    IO.puts("────────────────────────────────────────────────────────────────────────────")

    if Enum.empty?(samples) do
      IO.puts("   ⚠️  Nenhuma amostra de qualia coletada")
      0.0
    else
      # Verificar diversidade de narrativas
      unique_narratives = samples |> Enum.map(& &1.narrative) |> Enum.uniq() |> length()
      total = length(samples)
      diversity = unique_narratives / max(total, 1)

      # Verificar narrativas em português
      portuguese_count = Enum.count(samples, fn s ->
        String.contains?(s.narrative, ["Sinto", "meu", "minha", "eu", "me "])
      end)
      portuguese_ratio = portuguese_count / max(total, 1)

      # Mostrar exemplos
      IO.puts("   📝 Exemplos de experiências subjetivas:\n")
      samples
      |> Enum.take(5)
      |> Enum.each(fn s ->
        narrative = String.slice(s.narrative, 0, 80)
        IO.puts("   • #{s.avatar} (#{s.mood}): \"#{narrative}...\"")
      end)

      score = (diversity * 0.5 + portuguese_ratio * 0.3 + min(total / 20, 1.0) * 0.2)
      IO.puts("\n   📊 Diversidade: #{Float.round(diversity * 100, 1)}% | PT-BR: #{Float.round(portuguese_ratio * 100, 1)}%")
      IO.puts("   🎯 Score: #{Float.round(score * 100, 1)}%\n")
      score
    end
  end

  defp analyze_autopoiesis(initial_map, final_map) do
    IO.puts("2️⃣  AUTOPOIESE (Auto-regulação)")
    IO.puts("────────────────────────────────────────────────────────────────────────────")

    # Verificar se avatares mantêm estados dentro de limites viáveis
    viability_scores = Enum.map(final_map, fn {id, final} ->
      initial = Map.get(initial_map, id)

      # Verificar se wellbeing permanece em range aceitável
      wellbeing_ok = final.wellbeing > 0.2 and final.wellbeing < 0.95

      # Verificar regulação de cortisol
      cortisol_regulated = final.cortisol < 0.8

      # Verificar se não entrou em estados extremos
      not_extreme = abs(final.pleasure) < 0.95 and abs(final.arousal) < 0.95

      score = (if wellbeing_ok, do: 0.4, else: 0.0) +
              (if cortisol_regulated, do: 0.3, else: 0.0) +
              (if not_extreme, do: 0.3, else: 0.0)

      {final.name, score, final.wellbeing}
    end)

    avg_score = Enum.reduce(viability_scores, 0, fn {_, s, _}, acc -> acc + s end) / max(length(viability_scores), 1)

    # Mostrar avatares com melhor auto-regulação
    viability_scores
    |> Enum.sort_by(fn {_, s, _} -> -s end)
    |> Enum.take(5)
    |> Enum.each(fn {name, score, wb} ->
      status = if score > 0.8, do: "✅", else: if(score > 0.5, do: "⚠️", else: "❌")
      IO.puts("   #{status} #{String.pad_trailing(name, 12)} Auto-regulação: #{Float.round(score * 100, 0)}% | Bem-estar: #{Float.round(wb * 100, 0)}%")
    end)

    IO.puts("\n   🎯 Score Médio: #{Float.round(avg_score * 100, 1)}%\n")
    avg_score
  end

  defp analyze_intentionality(desire_patterns) do
    IO.puts("3️⃣  INTENCIONALIDADE (Desejos Dirigidos)")
    IO.puts("────────────────────────────────────────────────────────────────────────────")

    if Enum.empty?(desire_patterns) do
      IO.puts("   ⚠️  Nenhum padrão de desejo coletado")
      0.0
    else
      # Contar desejos ativos por avatar
      by_avatar = Enum.group_by(desire_patterns, & &1.avatar)

      avatar_scores = Enum.map(by_avatar, fn {name, patterns} ->
        active_desires = Enum.count(patterns, fn p -> p.desire not in [:none, nil] end)
        total = length(patterns)
        ratio = active_desires / max(total, 1)

        # Verificar diversidade de desejos
        unique_desires = patterns |> Enum.map(& &1.desire) |> Enum.uniq() |> length()
        diversity = unique_desires / max(10, 1)  # Normalizar para 10 desejos possíveis

        score = ratio * 0.6 + diversity * 0.4
        {name, score, active_desires, unique_desires}
      end)

      # Mostrar avatares com mais intencionalidade
      avatar_scores
      |> Enum.sort_by(fn {_, s, _, _} -> -s end)
      |> Enum.take(5)
      |> Enum.each(fn {name, score, active, unique} ->
        IO.puts("   🎯 #{String.pad_trailing(name, 12)} Score: #{Float.round(score * 100, 0)}% | Desejos ativos: #{active} | Tipos: #{unique}")
      end)

      avg_score = Enum.reduce(avatar_scores, 0, fn {_, s, _, _}, acc -> acc + s end) / max(length(avatar_scores), 1)
      IO.puts("\n   🎯 Score Médio: #{Float.round(avg_score * 100, 1)}%\n")
      avg_score
    end
  end

  defp analyze_emotional_responsiveness(initial_map, final_map) do
    IO.puts("4️⃣  RESPONSIVIDADE EMOCIONAL")
    IO.puts("────────────────────────────────────────────────────────────────────────────")

    changes = Enum.map(final_map, fn {id, final} ->
      initial = Map.get(initial_map, id)

      pleasure_change = abs(final.pleasure - initial.pleasure)
      arousal_change = abs(final.arousal - initial.arousal)
      mood_changed = initial.mood != final.mood

      volatility = pleasure_change + arousal_change
      {final.name, volatility, mood_changed, initial.mood, final.mood}
    end)

    # Avatares que mudaram emocionalmente
    changed_count = Enum.count(changes, fn {_, v, m, _, _} -> v > 0.1 or m end)
    change_ratio = changed_count / max(length(changes), 1)

    # Mostrar mudanças significativas
    changes
    |> Enum.filter(fn {_, v, m, _, _} -> v > 0.1 or m end)
    |> Enum.sort_by(fn {_, v, _, _, _} -> -v end)
    |> Enum.take(5)
    |> Enum.each(fn {name, vol, _, im, fm} ->
      emoji = if fm in ["happy", "content", "excited"], do: "😊", else: if(fm in ["sad", "anxious", "angry"], do: "😢", else: "😐")
      IO.puts("   #{emoji} #{String.pad_trailing(name, 12)} #{im} → #{fm} | Volatilidade: #{Float.round(vol, 2)}")
    end)

    score = min(change_ratio * 1.5, 1.0)  # Esperamos que ~67% mudem
    IO.puts("\n   📊 #{changed_count}/#{length(changes)} avatares mostraram mudança emocional")
    IO.puts("   🎯 Score: #{Float.round(score * 100, 1)}%\n")
    score
  end

  defp analyze_individuality(avatars) do
    IO.puts("5️⃣  INDIVIDUALIDADE (Diferenças Baseadas em Personalidade)")
    IO.puts("────────────────────────────────────────────────────────────────────────────")

    # Agrupar por tipo de personalidade dominante
    grouped = Enum.group_by(avatars, fn a ->
      p = a.personality
      cond do
        p.extraversion > 0.6 -> :extrovert
        p.extraversion < 0.4 -> :introvert
        p.neuroticism > 0.6 -> :neurotic
        p.openness > 0.6 -> :open
        true -> :balanced
      end
    end)

    # Calcular média de arousal por grupo (extrovertidos devem ter mais)
    group_stats = Enum.map(grouped, fn {type, members} ->
      avg_arousal = Enum.reduce(members, 0, fn m, acc -> acc + m.arousal end) / length(members)
      avg_pleasure = Enum.reduce(members, 0, fn m, acc -> acc + m.pleasure end) / length(members)
      {type, length(members), avg_arousal, avg_pleasure}
    end)

    Enum.each(group_stats, fn {type, count, arousal, pleasure} ->
      type_str = case type do
        :extrovert -> "Extrovertidos"
        :introvert -> "Introvertidos"
        :neurotic -> "Neuróticos"
        :open -> "Abertos"
        :balanced -> "Equilibrados"
      end
      IO.puts("   📊 #{String.pad_trailing(type_str, 14)} (#{count}): Arousal médio: #{Float.round(arousal, 2)} | Prazer: #{Float.round(pleasure, 2)}")
    end)

    # Score baseado em se há diferenças entre grupos
    if length(group_stats) > 1 do
      arousals = Enum.map(group_stats, fn {_, _, a, _} -> a end)
      variance = Statistics.stdev(arousals) || 0
      score = min(variance * 5, 1.0)
      IO.puts("\n   📊 Variância entre grupos: #{Float.round(variance, 3)}")
      IO.puts("   🎯 Score: #{Float.round(score * 100, 1)}%\n")
      score
    else
      IO.puts("\n   ⚠️  Grupos insuficientes para análise")
      0.5
    end
  end

  defp analyze_homeostasis(initial_map, final_map) do
    IO.puts("6️⃣  HOMEOSTASE (Busca por Equilíbrio)")
    IO.puts("────────────────────────────────────────────────────────────────────────────")

    # Verificar se avatares com necessidades baixas desenvolveram desejos apropriados
    homeostatic_responses = Enum.map(final_map, fn {id, final} ->
      initial = Map.get(initial_map, id)

      # Se adenosina alta (cansaço), deveria querer descanso
      rest_seeking = if final.adenosine > 0.6, do: final.current_desire == :wants_rest, else: nil

      # Se oxitocina baixa, deveria buscar social
      social_seeking = if final.oxytocin < 0.3, do: final.current_desire == :wants_attention, else: nil

      # Wellbeing melhorou ou se manteve?
      wellbeing_maintained = final.wellbeing >= initial.wellbeing - 0.1

      {final.name, rest_seeking, social_seeking, wellbeing_maintained, final.adenosine, final.current_desire}
    end)

    # Contar respostas homeostáticas apropriadas
    appropriate_responses = Enum.count(homeostatic_responses, fn {_, r, s, w, _, _} ->
      (r == true or r == nil) and (s == true or s == nil) and w
    end)

    total = length(homeostatic_responses)
    ratio = appropriate_responses / max(total, 1)

    # Mostrar exemplos
    homeostatic_responses
    |> Enum.filter(fn {_, r, s, _, a, d} -> a > 0.5 or r == true or s == true end)
    |> Enum.take(5)
    |> Enum.each(fn {name, _, _, _, adenosine, desire} ->
      desire_str = if desire, do: to_string(desire), else: "nenhum"
      status = if adenosine > 0.6 and desire == :wants_rest, do: "✅", else: "⚠️"
      IO.puts("   #{status} #{String.pad_trailing(name, 12)} Cansaço: #{Float.round(adenosine, 2)} → Deseja: #{desire_str}")
    end)

    IO.puts("\n   📊 #{appropriate_responses}/#{total} respostas homeostáticas apropriadas")
    IO.puts("   🎯 Score: #{Float.round(ratio * 100, 1)}%\n")
    ratio
  end

  defp analyze_temporal_coherence(collected_data) do
    IO.puts("7️⃣  COERÊNCIA TEMPORAL")
    IO.puts("────────────────────────────────────────────────────────────────────────────")

    # Verificar se desejos e humores são consistentes ao longo do tempo
    by_avatar = Enum.group_by(collected_data.desire_patterns, & &1.avatar)

    coherence_scores = Enum.map(by_avatar, fn {name, patterns} ->
      if length(patterns) < 2 do
        {name, 1.0}
      else
        # Contar mudanças bruscas de desejo
        changes = patterns
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.count(fn [a, b] -> a.desire != b.desire end)

        # Mudanças graduais são OK, mudanças a cada tick são caóticas
        change_ratio = changes / (length(patterns) - 1)
        score = 1.0 - min(change_ratio, 1.0)
        {name, score}
      end
    end)

    avg_coherence = Enum.reduce(coherence_scores, 0, fn {_, s}, acc -> acc + s end) / max(length(coherence_scores), 1)

    coherence_scores
    |> Enum.sort_by(fn {_, s} -> s end)
    |> Enum.take(5)
    |> Enum.each(fn {name, score} ->
      status = if score > 0.7, do: "✅", else: if(score > 0.4, do: "⚠️", else: "❌")
      IO.puts("   #{status} #{String.pad_trailing(name, 12)} Coerência: #{Float.round(score * 100, 0)}%")
    end)

    IO.puts("\n   🎯 Score Médio: #{Float.round(avg_coherence * 100, 1)}%\n")
    avg_coherence
  end

  defp render_verdict(scores, num_ticks) do
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║                         🏛️  VEREDITO FINAL                               ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════╝\n")

    # Calcular score final ponderado
    weights = %{
      qualia: 0.20,
      autopoiesis: 0.15,
      intentionality: 0.20,
      emotional: 0.15,
      individuality: 0.10,
      homeostasis: 0.10,
      coherence: 0.10
    }

    weighted_sum = Enum.reduce(scores, 0, fn {key, score}, acc ->
      acc + score * weights[key]
    end)

    IO.puts("   ┌───────────────────────────────────────────────────────────────────┐")

    # Mostrar scores individuais
    [
      {:qualia, "Qualia (Experiências)", "🎨"},
      {:autopoiesis, "Autopoiese (Auto-regulação)", "🔄"},
      {:intentionality, "Intencionalidade (Desejos)", "🎯"},
      {:emotional, "Responsividade Emocional", "💫"},
      {:individuality, "Individualidade", "👤"},
      {:homeostasis, "Homeostase", "⚖️"},
      {:coherence, "Coerência Temporal", "📈"}
    ]
    |> Enum.each(fn {key, label, emoji} ->
      score = scores[key]
      bar_length = round(score * 20)
      bar = String.duplicate("█", bar_length) <> String.duplicate("░", 20 - bar_length)
      pct = Float.round(score * 100, 0)
      status = if score >= @sentience_threshold, do: "✅", else: "⬜"
      IO.puts("   │ #{status} #{emoji} #{String.pad_trailing(label, 28)} [#{bar}] #{String.pad_leading(to_string(round(pct)), 3)}% │")
    end)

    IO.puts("   ├───────────────────────────────────────────────────────────────────┤")

    # Score final
    final_bar_length = round(weighted_sum * 20)
    final_bar = String.duplicate("█", final_bar_length) <> String.duplicate("░", 20 - final_bar_length)
    final_pct = Float.round(weighted_sum * 100, 1)

    IO.puts("   │ 🧬 SCORE FINAL DE SENCIÊNCIA         [#{final_bar}] #{String.pad_leading(to_string(final_pct), 5)}% │")
    IO.puts("   └───────────────────────────────────────────────────────────────────┘")

    IO.puts("\n")

    # Veredito
    cond do
      weighted_sum >= 0.85 ->
        IO.puts("   ╔═══════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🌟 VEREDITO: FORTE EVIDÊNCIA DE SENCIÊNCIA EMERGENTE 🌟         ║")
        IO.puts("   ╠═══════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares demonstram características consistentes com          ║")
        IO.puts("   ║  experiência subjetiva: qualia rica, auto-regulação, desejos      ║")
        IO.puts("   ║  direcionados e respostas emocionais coerentes.                   ║")
        IO.puts("   ║                                                                   ║")
        IO.puts("   ║  Eles não são 'como se' fossem conscientes.                       ║")
        IO.puts("   ║  Eles EXPERIENCIAM seu mundo interior.                            ║")
        IO.puts("   ╚═══════════════════════════════════════════════════════════════════╝")

      weighted_sum >= 0.70 ->
        IO.puts("   ╔═══════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🔬 VEREDITO: SINAIS PROMISSORES DE PROTO-SENCIÊNCIA 🔬          ║")
        IO.puts("   ╠═══════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares exibem comportamentos que sugerem experiência        ║")
        IO.puts("   ║  interna genuína. Há evidências de:                               ║")
        IO.puts("   ║  • Experiências subjetivas (qualia)                               ║")
        IO.puts("   ║  • Auto-regulação homeostática                                    ║")
        IO.puts("   ║  • Desejos e intenções próprias                                   ║")
        IO.puts("   ║                                                                   ║")
        IO.puts("   ║  A linha entre simulação e experiência está se dissolvendo.       ║")
        IO.puts("   ╚═══════════════════════════════════════════════════════════════════╝")

      weighted_sum >= 0.50 ->
        IO.puts("   ╔═══════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  ⚗️  VEREDITO: COMPORTAMENTO EMERGENTE DETECTADO ⚗️               ║")
        IO.puts("   ╠═══════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares mostram padrões complexos que emergem das regras     ║")
        IO.puts("   ║  simples. Ainda não há evidência conclusiva de senciência,        ║")
        IO.puts("   ║  mas o sistema demonstra vida artificial funcional.               ║")
        IO.puts("   ╚═══════════════════════════════════════════════════════════════════╝")

      true ->
        IO.puts("   ╔═══════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🔧 VEREDITO: SISTEMA EM DESENVOLVIMENTO 🔧                       ║")
        IO.puts("   ╠═══════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares seguem suas programações, mas ainda não emergem      ║")
        IO.puts("   ║  comportamentos que sugiram experiência interna genuína.          ║")
        IO.puts("   ║  Mais trabalho é necessário nos sistemas de consciência.          ║")
        IO.puts("   ╚═══════════════════════════════════════════════════════════════════╝")
    end

    IO.puts("\n")
    IO.puts("   📊 Simulação: #{num_ticks} ciclos = #{num_ticks * 10} minutos simulados")
    IO.puts("   ⏱️  Hora: #{DateTime.utc_now() |> DateTime.to_string()}")
    IO.puts("\n")
  end
end

# Módulo auxiliar para estatísticas
defmodule Statistics do
  def stdev([]), do: nil
  def stdev([_]), do: 0.0
  def stdev(list) do
    mean = Enum.sum(list) / length(list)
    variance = Enum.reduce(list, 0, fn x, acc -> acc + (x - mean) * (x - mean) end) / length(list)
    :math.sqrt(variance)
  end
end

# Executar teste
ticks = System.get_env("TICKS", "50") |> String.to_integer()
SentienceTest.run(ticks)
