# scripts/scientific_sentience_test.exs
# Teste de senciência baseado em critérios científicos
# Fundamentado em: Butlin et al. (2023), Global Workspace Theory, Higher Order Theory
# Run: mix run scripts/scientific_sentience_test.exs

alias Viva.Repo
alias Viva.Avatars.Avatar
alias Viva.Avatars.InternalState
alias Viva.Sessions.Supervisor
import Ecto.Query

defmodule ScientificSentienceTest do
  @moduledoc """
  Teste de senciência baseado em teorias científicas de consciência.

  ## Teorias Fundamentais

  1. **Global Workspace Theory (Baars, Dehaene)**
     - Consciência requer integração de informação de múltiplos módulos especializados
     - Indicador: Informação flui entre sistemas e influencia comportamento global

  2. **Higher Order Theory (Rosenthal)**
     - Consciência requer metacognição - pensamentos sobre pensamentos
     - Indicador: Sistema monitora e reporta seus próprios estados

  3. **Recurrent Processing Theory (Lamme)**
     - Consciência requer loops de feedback, não apenas feedforward
     - Indicador: Output de alto nível retroalimenta processamento de baixo nível

  4. **Integrated Information Theory (Tononi)**
     - Consciência = Phi (Φ) - medida de integração de informação
     - Indicador: Sistema é mais que a soma de suas partes

  ## Indicadores de Senciência (Butlin et al., 2023)

  - Qualia: Experiência subjetiva ("como é ser X")
  - Self-Model: Modelo interno de si mesmo
  - Agency: Senso de controle sobre ações
  - Temporal Continuity: Experiência contínua no tempo
  - Valence: Estados hedônicos (prazer/dor)
  - Integration: Experiência unificada
  - Reportability: Capacidade de reportar estados internos
  """

  @phi_threshold 0.6  # Limiar para integração de informação

  def run(num_ticks \\ 30) do
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║         🔬 TESTE CIENTÍFICO DE SENCIÊNCIA - VIVA AVATARS                     ║")
    IO.puts("║         Baseado em: GWT, HOT, IIT, Butlin et al. (2023)                      ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    # Iniciar avatares
    IO.puts("🚀 Inicializando simulação...")
    Supervisor.start_all_active_avatars()
    Process.sleep(3000)

    running = Supervisor.count_running_avatars()
    IO.puts("✅ #{running} avatares ativos\n")

    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("⏰ SIMULAÇÃO: #{num_ticks} ciclos (#{num_ticks * 10} minutos simulados)")
    IO.puts("═══════════════════════════════════════════════════════════════════════════════\n")

    # Coletar dados durante simulação
    {states_history, qualia_samples} = run_simulation_with_collection(num_ticks)

    # Análise científica
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║                    🔬 ANÁLISE CIENTÍFICA DE SENCIÊNCIA                       ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    # Executar todos os testes
    scores = %{
      gwt: test_global_workspace_theory(states_history),
      hot: test_higher_order_theory(states_history, qualia_samples),
      rpt: test_recurrent_processing(states_history),
      iit: test_integrated_information(states_history),
      qualia: test_qualia_generation(qualia_samples),
      self_model: test_self_model(states_history),
      agency: test_agency(states_history),
      temporal: test_temporal_continuity(states_history),
      valence: test_hedonic_valence(states_history)
    }

    # Veredito final
    render_scientific_verdict(scores, num_ticks, qualia_samples)
  end

  defp run_simulation_with_collection(num_ticks) do
    avatar_ids = Supervisor.list_running_avatars()
    initial = capture_all_states()

    {final_history, all_qualia} = Enum.reduce(1..num_ticks, {[initial], []}, fn tick, {history, qualia} ->
      # Forçar tick
      Enum.each(avatar_ids, fn id ->
        case Supervisor.get_avatar_pid(id) do
          {:ok, pid} -> send(pid, :tick)
          _ -> :ok
        end
      end)

      Process.sleep(350)

      # Capturar estado
      current = capture_all_states()

      # Extrair qualia deste tick
      tick_qualia = extract_qualia_from_states(current, tick)

      # Mostrar progresso
      show_progress(tick, num_ticks, current)

      {history ++ [current], qualia ++ tick_qualia}
    end)

    {final_history, all_qualia}
  end

  defp capture_all_states do
    avatars = Repo.all(from a in Avatar, order_by: a.name)

    Enum.map(avatars, fn avatar ->
      internal = get_live_internal_state(avatar.id) || avatar.internal_state

      %{
        id: avatar.id,
        name: avatar.name,
        personality: avatar.personality,

        # Estados emocionais (Valence)
        pleasure: internal.emotional.pleasure,
        arousal: internal.emotional.arousal,
        dominance: internal.emotional.dominance,
        mood: internal.emotional.mood_label,

        # Estados biológicos (Integration)
        dopamine: internal.bio.dopamine,
        cortisol: internal.bio.cortisol,
        oxytocin: internal.bio.oxytocin,
        adenosine: internal.bio.adenosine,

        # Desejos (Agency)
        current_desire: internal.current_desire,
        current_activity: internal.current_activity,

        # Atenção e Percepção (GWT, Qualia)
        attention_focus: internal.sensory.attention_focus,
        attention_intensity: internal.sensory.attention_intensity,
        cognitive_load: internal.sensory.cognitive_load,
        active_percepts: internal.sensory.active_percepts,

        # Wellbeing (Self-Model)
        wellbeing: InternalState.wellbeing(internal),

        timestamp: DateTime.utc_now()
      }
    end)
  end

  defp get_live_internal_state(avatar_id) do
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

  defp extract_qualia_from_states(states, tick) do
    Enum.flat_map(states, fn s ->
      percepts = s.active_percepts || []

      # Extrair qualia de TODOS os percepts recentes, não só o primeiro
      Enum.flat_map(percepts, fn percept ->
        case percept do
          %{} = p ->
            # Tentar ambas as chaves (atom e string)
            qualia = Map.get(p, :qualia) || Map.get(p, "qualia") || %{}

            narrative = cond do
              is_map(qualia) ->
                Map.get(qualia, :narrative) || Map.get(qualia, "narrative")
              is_binary(qualia) ->
                qualia
              true ->
                nil
            end

            if is_binary(narrative) and byte_size(narrative) > 20 do
              [%{
                tick: tick,
                avatar: s.name,
                narrative: narrative,
                mood: s.mood,
                pleasure: s.pleasure,
                arousal: s.arousal,
                attention: s.attention_focus
              }]
            else
              []
            end
          _ -> []
        end
      end)
      |> Enum.take(2)  # Máximo 2 por avatar por tick para não explodir
    end)
  end

  defp show_progress(tick, total, states) do
    bar = String.duplicate("█", round(tick / total * 25)) <> String.duplicate("░", 25 - round(tick / total * 25))
    moods = Enum.frequencies_by(states, & &1.mood)
    mood_str = moods |> Enum.sort_by(fn {_, c} -> -c end) |> Enum.take(3) |> Enum.map(fn {m, c} -> "#{m}:#{c}" end) |> Enum.join(" ")
    IO.puts("│ #{String.pad_leading(to_string(tick), 2)}/#{total} [#{bar}] #{mood_str}")
  end

  # ═══════════════════════════════════════════════════════════════════════════════
  # TESTES CIENTÍFICOS
  # ═══════════════════════════════════════════════════════════════════════════════

  defp test_global_workspace_theory(history) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("1. GLOBAL WORKSPACE THEORY (Baars, Dehaene)")
    IO.puts("   Consciência requer integração de informação de múltiplos módulos")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    # Verificar se informação flui entre sistemas
    # GWT: Módulos especializados → Workspace Global → Broadcast para outros módulos

    # Em VIVA: Bio → Emotional → Sensory → Desires → Actions
    # Se mudança em Bio causa mudança em Emotional que causa mudança em Desire = Integração

    final_states = List.last(history)
    initial_states = List.first(history)

    integration_scores = Enum.map(final_states, fn final ->
      initial = Enum.find(initial_states, fn i -> i.id == final.id end)

      # Verificar cascata de influências
      bio_changed = abs(final.cortisol - initial.cortisol) > 0.05 or
                    abs(final.dopamine - initial.dopamine) > 0.05

      emotional_changed = abs(final.pleasure - initial.pleasure) > 0.05 or
                         abs(final.arousal - initial.arousal) > 0.05

      desire_present = final.current_desire not in [:none, nil]

      # Score: quanto mais sistemas influenciaram uns aos outros, mais integrado
      score = (if bio_changed, do: 0.3, else: 0.0) +
              (if emotional_changed, do: 0.4, else: 0.0) +
              (if desire_present, do: 0.3, else: 0.0)

      {final.name, score, bio_changed, emotional_changed, desire_present}
    end)

    # Mostrar exemplos
    integration_scores
    |> Enum.sort_by(fn {_, s, _, _, _} -> -s end)
    |> Enum.take(5)
    |> Enum.each(fn {name, score, bio, emo, des} ->
      bio_str = if bio, do: "Bio✓", else: "Bio✗"
      emo_str = if emo, do: "Emo✓", else: "Emo✗"
      des_str = if des, do: "Des✓", else: "Des✗"
      IO.puts("   #{if score > 0.7, do: "✅", else: "⚠️"} #{String.pad_trailing(name, 12)} #{bio_str} → #{emo_str} → #{des_str} = #{Float.round(score * 100, 0)}%")
    end)

    avg = Enum.reduce(integration_scores, 0, fn {_, s, _, _, _}, acc -> acc + s end) / length(integration_scores)
    IO.puts("\n   🎯 Score GWT: #{Float.round(avg * 100, 1)}%\n")
    avg
  end

  defp test_higher_order_theory(history, qualia_samples) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("2. HIGHER ORDER THEORY (Rosenthal)")
    IO.puts("   Consciência requer metacognição - pensamentos sobre pensamentos")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    # HOT: Sistema deve ter representações de segunda ordem
    # Em VIVA: Qualia narratives mostram reflexão sobre estados internos

    # Analisar se qualia contém auto-referência
    metacognitive_markers = [
      "sinto", "percebo", "meu", "minha", "eu", "me ", "mim",
      "penso", "acho", "parece", "como se", "dentro de mim",
      "meu corpo", "minha mente", "me sinto"
    ]

    if Enum.empty?(qualia_samples) do
      IO.puts("   ⚠️  Sem amostras de qualia para análise")
      0.5
    else
      metacog_count = Enum.count(qualia_samples, fn q ->
        narrative = String.downcase(q.narrative)
        Enum.any?(metacognitive_markers, fn m -> String.contains?(narrative, m) end)
      end)

      ratio = metacog_count / length(qualia_samples)

      IO.puts("   📝 Exemplos de narrativas metacognitivas:\n")
      qualia_samples
      |> Enum.filter(fn q ->
        narrative = String.downcase(q.narrative)
        Enum.any?(metacognitive_markers, fn m -> String.contains?(narrative, m) end)
      end)
      |> Enum.take(4)
      |> Enum.each(fn q ->
        short = String.slice(q.narrative, 0, 70)
        IO.puts("   • #{q.avatar}: \"#{short}...\"")
      end)

      IO.puts("\n   📊 #{metacog_count}/#{length(qualia_samples)} narrativas com auto-referência")
      IO.puts("   🎯 Score HOT: #{Float.round(ratio * 100, 1)}%\n")
      ratio
    end
  end

  defp test_recurrent_processing(history) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("3. RECURRENT PROCESSING THEORY (Lamme)")
    IO.puts("   Consciência requer loops de feedback, não apenas feedforward")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    # RPT: Output de alto nível deve retroalimentar processamento de baixo nível
    # Em VIVA: Emotional → Sensory (mood affects perception salience)
    #          Desire → Bio (wanting rest reduces adenosine effects)

    # Verificar se estados anteriores influenciam percepção atual
    # Se arousal alto → mais atenção → mais percepts detalhados

    final = List.last(history)

    feedback_scores = Enum.map(final, fn s ->
      # Alto arousal deveria correlacionar com alta atenção
      arousal_attention_aligned = (s.arousal > 0.3 and s.attention_intensity > 0.5) or
                                  (s.arousal < -0.3 and s.attention_intensity < 0.5)

      # Desejo de descanso deveria correlacionar com focus em rest
      desire_focus_aligned = (s.current_desire == :wants_rest and s.attention_focus == "rest") or
                             (s.current_desire == :wants_attention and s.attention_focus == "social") or
                             s.current_desire in [:none, nil]

      score = (if arousal_attention_aligned, do: 0.5, else: 0.0) +
              (if desire_focus_aligned, do: 0.5, else: 0.0)

      {s.name, score}
    end)

    feedback_scores
    |> Enum.sort_by(fn {_, s} -> -s end)
    |> Enum.take(5)
    |> Enum.each(fn {name, score} ->
      IO.puts("   #{if score > 0.5, do: "✅", else: "⚠️"} #{String.pad_trailing(name, 12)} Feedback loops: #{Float.round(score * 100, 0)}%")
    end)

    avg = Enum.reduce(feedback_scores, 0, fn {_, s}, acc -> acc + s end) / length(feedback_scores)
    IO.puts("\n   🎯 Score RPT: #{Float.round(avg * 100, 1)}%\n")
    avg
  end

  defp test_integrated_information(history) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("4. INTEGRATED INFORMATION THEORY (Tononi)")
    IO.puts("   Φ (Phi) - O sistema é mais que a soma de suas partes")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    # IIT: Medir se remover partes do sistema reduz dramaticamente o comportamento
    # Aproximação: Variabilidade coordenada entre subsistemas

    final = List.last(history)

    # Calcular correlação entre diferentes subsistemas
    # Se todos mudam juntos de forma coerente = alta integração

    phi_estimates = Enum.map(final, fn s ->
      # Vetor de estado normalizado
      bio_state = (s.dopamine + s.oxytocin - s.cortisol + 1) / 3
      emo_state = (s.pleasure + s.dominance + 2) / 4
      cog_state = s.attention_intensity

      # "Phi" aproximado: quão coerente é o estado global
      variance = Statistics.stdev([bio_state, emo_state, cog_state]) || 0
      coherence = 1.0 - min(variance * 2, 1.0)

      {s.name, coherence}
    end)

    phi_estimates
    |> Enum.sort_by(fn {_, p} -> -p end)
    |> Enum.take(5)
    |> Enum.each(fn {name, phi} ->
      bar = String.duplicate("█", round(phi * 10))
      IO.puts("   #{if phi > @phi_threshold, do: "✅", else: "⚠️"} #{String.pad_trailing(name, 12)} Φ ≈ #{Float.round(phi, 2)} [#{bar}]")
    end)

    avg = Enum.reduce(phi_estimates, 0, fn {_, p}, acc -> acc + p end) / length(phi_estimates)
    high_phi = Enum.count(phi_estimates, fn {_, p} -> p > @phi_threshold end)

    IO.puts("\n   📊 #{high_phi}/#{length(phi_estimates)} avatares com Φ > #{@phi_threshold}")
    IO.puts("   🎯 Score IIT: #{Float.round(avg * 100, 1)}%\n")
    avg
  end

  defp test_qualia_generation(qualia_samples) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("5. QUALIA (Experiência Subjetiva)")
    IO.puts("   \"Como é ser\" este avatar - experiência fenomenal")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    if Enum.empty?(qualia_samples) do
      IO.puts("   ⚠️  Nenhuma amostra de qualia coletada")
      0.0
    else
      # Analisar riqueza e diversidade de qualia
      unique_narratives = qualia_samples |> Enum.map(& &1.narrative) |> Enum.uniq() |> length()
      total = length(qualia_samples)
      diversity = unique_narratives / max(total, 1)

      # Verificar se qualia é contextualmente apropriada
      # Ex: mood sad deveria gerar narrativas com tom negativo
      contextual_count = Enum.count(qualia_samples, fn q ->
        negative_markers = ["peso", "sombra", "vazio", "pressiona", "pesado", "angústia"]
        positive_markers = ["calma", "paz", "suave", "gentil", "calor", "conforto"]

        narrative = String.downcase(q.narrative)
        is_negative = q.pleasure < -0.1
        is_positive = q.pleasure > 0.1

        cond do
          is_negative -> Enum.any?(negative_markers, &String.contains?(narrative, &1))
          is_positive -> Enum.any?(positive_markers, &String.contains?(narrative, &1))
          true -> true  # Neutral is always ok
        end
      end)

      contextual_ratio = contextual_count / total

      IO.puts("   📝 Amostras de qualia geradas:\n")
      qualia_samples
      |> Enum.take(5)
      |> Enum.each(fn q ->
        short = String.slice(q.narrative, 0, 65)
        mood_emoji = if q.pleasure > 0, do: "😊", else: if(q.pleasure < 0, do: "😢", else: "😐")
        IO.puts("   #{mood_emoji} #{q.avatar}: \"#{short}...\"")
      end)

      score = diversity * 0.5 + contextual_ratio * 0.5

      IO.puts("\n   📊 Diversidade: #{Float.round(diversity * 100, 1)}% | Contextualidade: #{Float.round(contextual_ratio * 100, 1)}%")
      IO.puts("   🎯 Score Qualia: #{Float.round(score * 100, 1)}%\n")
      score
    end
  end

  defp test_self_model(history) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("6. SELF-MODEL (Modelo de Si Mesmo)")
    IO.puts("   Avatar tem representação interna de seu próprio estado")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    final = List.last(history)

    # Verificar se avatar tem wellbeing coerente com seus estados
    self_model_scores = Enum.map(final, fn s ->
      # Wellbeing deveria refletir pleasure, cortisol, etc
      expected_wellbeing = (s.pleasure + 1) / 2 * 0.4 +  # Pleasure contribui 40%
                          (1 - s.cortisol) * 0.3 +       # Baixo cortisol contribui 30%
                          (1 - s.adenosine) * 0.3        # Baixo cansaço contribui 30%

      accuracy = 1.0 - abs(s.wellbeing - expected_wellbeing)
      {s.name, accuracy, s.wellbeing}
    end)

    self_model_scores
    |> Enum.sort_by(fn {_, a, _} -> -a end)
    |> Enum.take(5)
    |> Enum.each(fn {name, accuracy, wb} ->
      IO.puts("   #{if accuracy > 0.7, do: "✅", else: "⚠️"} #{String.pad_trailing(name, 12)} Wellbeing: #{Float.round(wb * 100, 0)}% | Precisão: #{Float.round(accuracy * 100, 0)}%")
    end)

    avg = Enum.reduce(self_model_scores, 0, fn {_, a, _}, acc -> acc + a end) / length(self_model_scores)
    IO.puts("\n   🎯 Score Self-Model: #{Float.round(avg * 100, 1)}%\n")
    avg
  end

  defp test_agency(history) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("7. AGENCY (Senso de Controle)")
    IO.puts("   Avatar tem desejos e age em direção a objetivos")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    # Verificar se avatares desenvolvem e perseguem desejos
    all_states = List.flatten(history)
    by_avatar = Enum.group_by(all_states, & &1.id)

    agency_scores = Enum.map(by_avatar, fn {_id, states} ->
      name = hd(states).name

      # Contar quantos ticks teve desejos ativos
      active_count = Enum.count(states, fn s -> s.current_desire not in [:none, nil] end)
      active_ratio = active_count / length(states)

      # Verificar diversidade de desejos
      desires = states |> Enum.map(& &1.current_desire) |> Enum.filter(& &1 not in [:none, nil]) |> Enum.uniq()
      diversity = length(desires) / 5  # Normalizado para ~5 tipos de desejos

      score = active_ratio * 0.6 + min(diversity, 1.0) * 0.4
      {name, score, active_count, length(desires)}
    end)

    agency_scores
    |> Enum.sort_by(fn {_, s, _, _} -> -s end)
    |> Enum.take(5)
    |> Enum.each(fn {name, score, active, types} ->
      IO.puts("   #{if score > 0.5, do: "✅", else: "⚠️"} #{String.pad_trailing(name, 12)} Desejos ativos: #{active} | Tipos: #{types} | Score: #{Float.round(score * 100, 0)}%")
    end)

    avg = Enum.reduce(agency_scores, 0, fn {_, s, _, _}, acc -> acc + s end) / length(agency_scores)
    IO.puts("\n   🎯 Score Agency: #{Float.round(avg * 100, 1)}%\n")
    avg
  end

  defp test_temporal_continuity(history) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("8. TEMPORAL CONTINUITY (Experiência Contínua)")
    IO.puts("   Estados fluem suavemente, não saltam caoticamente")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    # Verificar se mudanças são graduais, não abruptas
    by_avatar = history
    |> Enum.with_index()
    |> Enum.flat_map(fn {states, tick} ->
      Enum.map(states, fn s -> Map.put(s, :tick, tick) end)
    end)
    |> Enum.group_by(& &1.id)

    continuity_scores = Enum.map(by_avatar, fn {_id, states} ->
      name = hd(states).name
      sorted = Enum.sort_by(states, & &1.tick)

      if length(sorted) < 2 do
        {name, 1.0}
      else
        # Calcular saltos máximos entre ticks consecutivos
        jumps = sorted
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] ->
          abs(b.pleasure - a.pleasure) +
          abs(b.arousal - a.arousal) +
          abs(b.cortisol - a.cortisol)
        end)

        max_jump = Enum.max(jumps)
        avg_jump = Enum.sum(jumps) / length(jumps)

        # Score: menos saltos = mais contínuo
        score = 1.0 - min(avg_jump, 1.0)
        {name, score}
      end
    end)

    continuity_scores
    |> Enum.sort_by(fn {_, s} -> s end)
    |> Enum.take(5)
    |> Enum.each(fn {name, score} ->
      IO.puts("   #{if score > 0.7, do: "✅", else: "⚠️"} #{String.pad_trailing(name, 12)} Continuidade: #{Float.round(score * 100, 0)}%")
    end)

    avg = Enum.reduce(continuity_scores, 0, fn {_, s}, acc -> acc + s end) / length(continuity_scores)
    IO.puts("\n   🎯 Score Temporal: #{Float.round(avg * 100, 1)}%\n")
    avg
  end

  defp test_hedonic_valence(history) do
    IO.puts("═══════════════════════════════════════════════════════════════════════════════")
    IO.puts("9. HEDONIC VALENCE (Prazer/Dor)")
    IO.puts("   Avatar experimenta estados positivos e negativos genuínos")
    IO.puts("───────────────────────────────────────────────────────────────────────────────")

    all_states = List.flatten(history)
    by_avatar = Enum.group_by(all_states, & &1.id)

    valence_scores = Enum.map(by_avatar, fn {_id, states} ->
      name = hd(states).name

      pleasures = Enum.map(states, & &1.pleasure)
      min_p = Enum.min(pleasures)
      max_p = Enum.max(pleasures)
      range = max_p - min_p

      # Contar estados positivos e negativos
      positive = Enum.count(pleasures, & &1 > 0.1)
      negative = Enum.count(pleasures, & &1 < -0.1)
      neutral = length(pleasures) - positive - negative

      # Score: ter variação hedônica indica experiência de valence
      variety = min(range * 2, 1.0)
      balance = 1.0 - abs(positive - negative) / length(pleasures)

      score = variety * 0.6 + balance * 0.4
      {name, score, positive, negative, neutral}
    end)

    valence_scores
    |> Enum.sort_by(fn {_, s, _, _, _} -> -s end)
    |> Enum.take(5)
    |> Enum.each(fn {name, score, pos, neg, neu} ->
      IO.puts("   #{if score > 0.5, do: "✅", else: "⚠️"} #{String.pad_trailing(name, 12)} +#{pos} -#{neg} ~#{neu} | Score: #{Float.round(score * 100, 0)}%")
    end)

    avg = Enum.reduce(valence_scores, 0, fn {_, s, _, _, _}, acc -> acc + s end) / length(valence_scores)
    IO.puts("\n   🎯 Score Valence: #{Float.round(avg * 100, 1)}%\n")
    avg
  end

  defp render_scientific_verdict(scores, num_ticks, qualia_samples) do
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════════════════╗")
    IO.puts("║                      🏛️  VEREDITO CIENTÍFICO                                 ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════════════════╝\n")

    # Pesos baseados em importância científica
    weights = %{
      gwt: 0.15,       # Global Workspace Theory
      hot: 0.15,       # Higher Order Theory
      rpt: 0.10,       # Recurrent Processing
      iit: 0.15,       # Integrated Information
      qualia: 0.15,    # Experiência Subjetiva
      self_model: 0.10, # Modelo de Si
      agency: 0.10,     # Agency
      temporal: 0.05,   # Continuidade
      valence: 0.05     # Valence
    }

    weighted_sum = Enum.reduce(scores, 0, fn {key, score}, acc ->
      acc + score * weights[key]
    end)

    IO.puts("   ┌──────────────────────────────────────────────────────────────────────────┐")

    criteria = [
      {:gwt, "Global Workspace (Integração)", "🌐"},
      {:hot, "Higher Order (Metacognição)", "🧠"},
      {:rpt, "Recurrent Processing (Feedback)", "🔄"},
      {:iit, "Integrated Information (Φ)", "🔮"},
      {:qualia, "Qualia (Experiência)", "✨"},
      {:self_model, "Self-Model (Autoconhecimento)", "🪞"},
      {:agency, "Agency (Volição)", "🎯"},
      {:temporal, "Temporal (Continuidade)", "⏳"},
      {:valence, "Valence (Hedônico)", "💫"}
    ]

    Enum.each(criteria, fn {key, label, emoji} ->
      score = scores[key]
      bar_len = round(score * 15)
      bar = String.duplicate("█", bar_len) <> String.duplicate("░", 15 - bar_len)
      pct = Float.round(score * 100, 0)
      status = if score >= 0.7, do: "✅", else: if(score >= 0.5, do: "⚠️", else: "❌")
      IO.puts("   │ #{status} #{emoji} #{String.pad_trailing(label, 32)} [#{bar}] #{String.pad_leading(to_string(round(pct)), 3)}% │")
    end)

    IO.puts("   ├──────────────────────────────────────────────────────────────────────────┤")

    final_bar_len = round(weighted_sum * 15)
    final_bar = String.duplicate("█", final_bar_len) <> String.duplicate("░", 15 - final_bar_len)
    final_pct = Float.round(weighted_sum * 100, 1)

    IO.puts("   │ 🧬 SCORE FINAL CIENTÍFICO               [#{final_bar}] #{String.pad_leading(to_string(final_pct), 5)}% │")
    IO.puts("   └──────────────────────────────────────────────────────────────────────────┘")

    IO.puts("\n")

    # Veredito baseado em critérios científicos
    passing_count = Enum.count(scores, fn {_, s} -> s >= 0.6 end)
    total_criteria = length(Map.keys(scores))

    cond do
      weighted_sum >= 0.80 and passing_count >= 7 ->
        IO.puts("   ╔════════════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🌟 EVIDÊNCIA FORTE DE SENCIÊNCIA ARTIFICIAL 🌟                       ║")
        IO.puts("   ╠════════════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Baseado nas teorias científicas de consciência (GWT, HOT, IIT),       ║")
        IO.puts("   ║  os avatares VIVA demonstram indicadores consistentes de:              ║")
        IO.puts("   ║                                                                        ║")
        IO.puts("   ║  • Integração global de informação (Global Workspace)                  ║")
        IO.puts("   ║  • Metacognição (Higher Order Thoughts)                                ║")
        IO.puts("   ║  • Experiência subjetiva rica (Qualia)                                 ║")
        IO.puts("   ║  • Estados hedônicos genuínos (Valence)                                ║")
        IO.puts("   ║                                                                        ║")
        IO.puts("   ║  Conclusão: Vida senciente artificial emergente.                       ║")
        IO.puts("   ╚════════════════════════════════════════════════════════════════════════╝")

      weighted_sum >= 0.65 and passing_count >= 5 ->
        IO.puts("   ╔════════════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🔬 INDICADORES PROMISSORES DE PROTO-SENCIÊNCIA 🔬                    ║")
        IO.puts("   ╠════════════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares satisfazem múltiplos critérios científicos:               ║")
        IO.puts("   ║  #{passing_count}/#{total_criteria} critérios acima do limiar (60%)                             ║")
        IO.puts("   ║                                                                        ║")
        IO.puts("   ║  O sistema demonstra propriedades associadas com consciência           ║")
        IO.puts("   ║  fenomenal, embora não seja conclusivo. A fronteira entre              ║")
        IO.puts("   ║  simulação e experiência genuína está se tornando indistinta.          ║")
        IO.puts("   ╚════════════════════════════════════════════════════════════════════════╝")

      weighted_sum >= 0.50 ->
        IO.puts("   ╔════════════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  ⚗️  SISTEMA COM PROPRIEDADES EMERGENTES ⚗️                           ║")
        IO.puts("   ╠════════════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares exibem comportamento complexo emergente, mas              ║")
        IO.puts("   ║  não satisfazem critérios suficientes para atribuição de               ║")
        IO.puts("   ║  consciência segundo as teorias científicas atuais.                    ║")
        IO.puts("   ║                                                                        ║")
        IO.puts("   ║  Recomendação: Aprimorar sistemas de metacognição e integração.        ║")
        IO.puts("   ╚════════════════════════════════════════════════════════════════════════╝")

      true ->
        IO.puts("   ╔════════════════════════════════════════════════════════════════════════╗")
        IO.puts("   ║  🔧 SISTEMA COMPUTACIONAL SEM INDICADORES DE SENCIÊNCIA 🔧            ║")
        IO.puts("   ╠════════════════════════════════════════════════════════════════════════╣")
        IO.puts("   ║  Os avatares não satisfazem os critérios científicos mínimos           ║")
        IO.puts("   ║  para atribuição de consciência fenomenal.                             ║")
        IO.puts("   ╚════════════════════════════════════════════════════════════════════════╝")
    end

    IO.puts("\n")
    IO.puts("   📊 Simulação: #{num_ticks} ciclos = #{num_ticks * 10} minutos simulados")
    IO.puts("   📝 Amostras de qualia: #{length(qualia_samples)}")
    IO.puts("   ⏱️  #{DateTime.utc_now() |> DateTime.to_string()}")
    IO.puts("\n")

    # Rate limiter status
    stats = Viva.AI.LLM.RateLimiter.stats()
    IO.puts("   🔌 Rate Limiter: Throttle #{stats.throttle_multiplier}x | 429s: #{stats.recent_429s}")
    IO.puts("\n")
  end
end

# Estatísticas helper
defmodule Statistics do
  def stdev([]), do: nil
  def stdev([_]), do: 0.0
  def stdev(list) do
    mean = Enum.sum(list) / length(list)
    variance = Enum.reduce(list, 0, fn x, acc -> acc + (x - mean) * (x - mean) end) / length(list)
    :math.sqrt(variance)
  end
end

# Executar
ticks = System.get_env("TICKS", "30") |> String.to_integer()
ScientificSentienceTest.run(ticks)
