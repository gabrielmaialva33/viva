# scripts/run_simulation.exs
# Simulação estendida para observar emergência de comportamentos
# Run: mix run scripts/run_simulation.exs

alias Viva.Repo
alias Viva.Avatars.Avatar
alias Viva.Avatars.InternalState
alias Viva.Relationships.Relationship
alias Viva.Sessions.{Supervisor, LifeProcess}
import Ecto.Query

defmodule SimulationRunner do
  @moduledoc "Runs extended simulation to observe emergent behaviors"

  def run(num_ticks \\ 20) do
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════╗")
    IO.puts("║          🧬 VIVA SENTIENCE SIMULATION                            ║")
    IO.puts("║          Observando emergência de vida artificial                ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════╝")
    IO.puts("")

    # Start all avatars
    IO.puts("🚀 Inicializando #{count_avatars()} avatares...")
    Supervisor.start_all_active_avatars()
    Process.sleep(2000)

    running = Supervisor.count_running_avatars()
    IO.puts("✅ #{running} processos de vida ativos\n")

    # Capture initial state
    initial_states = capture_world_state()

    IO.puts("═══════════════════════════════════════════════════════════════════")
    IO.puts("⏰ INICIANDO SIMULAÇÃO: #{num_ticks} ciclos de vida")
    IO.puts("   (1 ciclo = 10 minutos simulados)")
    IO.puts("═══════════════════════════════════════════════════════════════════\n")

    # Run simulation ticks
    final_states = run_ticks(num_ticks, initial_states)

    # Analysis
    IO.puts("\n")
    IO.puts("╔══════════════════════════════════════════════════════════════════╗")
    IO.puts("║                    📊 ANÁLISE FINAL                              ║")
    IO.puts("╚══════════════════════════════════════════════════════════════════╝\n")

    analyze_changes(initial_states, final_states, num_ticks)

    IO.puts("\n✨ Simulação completa. #{num_ticks * 10} minutos simulados passaram.\n")
  end

  defp count_avatars do
    Repo.aggregate(Avatar, :count)
  end

  defp capture_world_state do
    avatars = Repo.all(from a in Avatar, order_by: a.name)

    states =
      Enum.map(avatars, fn avatar ->
        # Try to get live state from GenServer if running
        live_state = get_live_state(avatar.id)
        internal = live_state || avatar.internal_state

        %{
          id: avatar.id,
          name: avatar.name,
          wellbeing: InternalState.wellbeing(internal),
          mood: internal.emotional.mood_label,
          pleasure: internal.emotional.pleasure,
          arousal: internal.emotional.arousal,
          dominance: internal.emotional.dominance,
          dopamine: internal.bio.dopamine,
          oxytocin: internal.bio.oxytocin,
          cortisol: internal.bio.cortisol,
          current_desire: internal.current_desire,
          current_activity: internal.current_activity
        }
      end)

    relationships =
      Repo.all(Relationship)
      |> Enum.map(fn r ->
        %{
          pair: {r.avatar_a_id, r.avatar_b_id},
          trust: r.trust,
          affection: r.affection,
          attraction: r.attraction,
          a_romantic: r.a_feelings.romantic_interest,
          b_romantic: r.b_feelings.romantic_interest,
          conflicts: r.unresolved_conflicts
        }
      end)

    %{avatars: states, relationships: relationships, timestamp: DateTime.utc_now()}
  end

  defp get_live_state(avatar_id) do
    case Supervisor.get_avatar_pid(avatar_id) do
      {:ok, pid} ->
        try do
          state = :sys.get_state(pid)
          state.state
        catch
          _, _ -> nil
        end

      _ ->
        nil
    end
  end

  defp run_ticks(num_ticks, initial_states) do
    avatar_ids = Supervisor.list_running_avatars()

    Enum.reduce(1..num_ticks, initial_states, fn tick, _prev_states ->
      # Force tick on all avatars
      Enum.each(avatar_ids, fn id ->
        case Supervisor.get_avatar_pid(id) do
          {:ok, pid} -> send(pid, :tick)
          _ -> :ok
        end
      end)

      # Small delay to let processing happen
      Process.sleep(500)

      # Capture current state
      current = capture_world_state()

      # Show tick progress with interesting events
      show_tick_summary(tick, current, num_ticks)

      current
    end)
  end

  defp show_tick_summary(tick, state, total) do
    simulated_minutes = tick * 10
    hours = div(simulated_minutes, 60)
    mins = rem(simulated_minutes, 60)

    time_str =
      if hours > 0 do
        "#{hours}h#{mins}m"
      else
        "#{mins}m"
      end

    # Find notable events
    notable = find_notable_states(state.avatars)

    progress = String.duplicate("█", round(tick / total * 20))
    remaining = String.duplicate("░", 20 - round(tick / total * 20))

    IO.puts("┌─────────────────────────────────────────────────────────────────┐")
    IO.puts("│ Ciclo #{String.pad_leading(to_string(tick), 2)}/#{total}  [#{progress}#{remaining}]  Tempo: #{time_str} simulados")
    IO.puts("├─────────────────────────────────────────────────────────────────┤")

    if length(notable) > 0 do
      Enum.each(notable, fn event -> IO.puts("│ #{event}") end)
    else
      IO.puts("│ 💤 Momento tranquilo na colônia...")
    end

    IO.puts("└─────────────────────────────────────────────────────────────────┘\n")
  end

  defp find_notable_states(avatars) do
    events = []

    # High pleasure
    high_pleasure =
      Enum.filter(avatars, fn a -> a.pleasure > 0.7 end)
      |> Enum.map(fn a -> "😊 #{a.name} está radiante (prazer: #{Float.round(a.pleasure, 2)})" end)

    # Low pleasure (suffering)
    low_pleasure =
      Enum.filter(avatars, fn a -> a.pleasure < -0.4 end)
      |> Enum.map(fn a -> "😢 #{a.name} está sofrendo (prazer: #{Float.round(a.pleasure, 2)})" end)

    # High arousal (excited)
    excited =
      Enum.filter(avatars, fn a -> a.arousal > 0.7 end)
      |> Enum.map(fn a -> "⚡ #{a.name} está agitado(a) (arousal: #{Float.round(a.arousal, 2)})" end)

    # Strong desires
    desiring =
      Enum.filter(avatars, fn a -> a.current_desire not in [:none, nil] end)
      |> Enum.map(fn a -> "💭 #{a.name} deseja: #{a.current_desire}" end)

    # High cortisol (stressed)
    stressed =
      Enum.filter(avatars, fn a -> a.cortisol > 0.7 end)
      |> Enum.map(fn a -> "😰 #{a.name} está estressado(a) (cortisol: #{Float.round(a.cortisol, 2)})" end)

    # High oxytocin (bonding)
    bonding =
      Enum.filter(avatars, fn a -> a.oxytocin > 0.7 end)
      |> Enum.map(fn a -> "💕 #{a.name} sente conexão forte (oxitocina: #{Float.round(a.oxytocin, 2)})" end)

    (events ++ high_pleasure ++ low_pleasure ++ excited ++ desiring ++ stressed ++ bonding)
    |> Enum.take(5)
  end

  defp analyze_changes(initial, final, num_ticks) do
    IO.puts("📈 MUDANÇAS DE BEM-ESTAR (#{num_ticks * 10} minutos simulados)")
    IO.puts("───────────────────────────────────────────────────────────────────")

    initial_map = Map.new(initial.avatars, fn a -> {a.id, a} end)
    final_map = Map.new(final.avatars, fn a -> {a.id, a} end)

    changes =
      Enum.map(final.avatars, fn f ->
        i = Map.get(initial_map, f.id)
        delta = Float.round((f.wellbeing - i.wellbeing) * 100, 1)
        {f.name, i.wellbeing, f.wellbeing, delta, i.mood, f.mood}
      end)
      |> Enum.sort_by(fn {_, _, _, delta, _, _} -> delta end, :desc)

    Enum.each(changes, fn {name, initial_w, final_w, delta, initial_mood, final_mood} ->
      arrow =
        cond do
          delta > 5 -> "📈"
          delta < -5 -> "📉"
          true -> "➡️"
        end

      mood_change =
        if initial_mood != final_mood do
          " (#{initial_mood} → #{final_mood})"
        else
          ""
        end

      delta_str =
        if delta >= 0 do
          "+#{delta}%"
        else
          "#{delta}%"
        end

      IO.puts(
        "#{arrow} #{String.pad_trailing(name, 12)} #{Float.round(initial_w * 100, 0)}% → #{Float.round(final_w * 100, 0)}% (#{delta_str})#{mood_change}"
      )
    end)

    # Emotional volatility analysis
    IO.puts("\n🎭 VOLATILIDADE EMOCIONAL")
    IO.puts("───────────────────────────────────────────────────────────────────")

    volatility =
      Enum.map(final.avatars, fn f ->
        i = Map.get(initial_map, f.id)

        changes =
          abs(f.pleasure - i.pleasure) +
            abs(f.arousal - i.arousal) +
            abs(f.dominance - i.dominance)

        {f.name, Float.round(changes, 2)}
      end)
      |> Enum.sort_by(fn {_, v} -> v end, :desc)
      |> Enum.take(5)

    Enum.each(volatility, fn {name, vol} ->
      bar = String.duplicate("█", round(vol * 10))
      IO.puts("   #{String.pad_trailing(name, 12)} #{bar} #{vol}")
    end)

    # Neurochemical shifts
    IO.puts("\n🧪 MUDANÇAS NEUROQUÍMICAS SIGNIFICATIVAS")
    IO.puts("───────────────────────────────────────────────────────────────────")

    neuro_changes =
      Enum.flat_map(final.avatars, fn f ->
        i = Map.get(initial_map, f.id)

        [
          if(abs(f.dopamine - i.dopamine) > 0.1,
            do:
              {"#{f.name}: Dopamina", i.dopamine, f.dopamine,
               if(f.dopamine > i.dopamine, do: "🔺", else: "🔻")},
            else: nil
          ),
          if(abs(f.oxytocin - i.oxytocin) > 0.1,
            do:
              {"#{f.name}: Oxitocina", i.oxytocin, f.oxytocin,
               if(f.oxytocin > i.oxytocin, do: "🔺", else: "🔻")},
            else: nil
          ),
          if(abs(f.cortisol - i.cortisol) > 0.1,
            do:
              {"#{f.name}: Cortisol", i.cortisol, f.cortisol,
               if(f.cortisol > i.cortisol, do: "🔺", else: "🔻")},
            else: nil
          )
        ]
        |> Enum.reject(&is_nil/1)
      end)

    if length(neuro_changes) > 0 do
      Enum.take(neuro_changes, 10)
      |> Enum.each(fn {label, from, to, arrow} ->
        IO.puts(
          "   #{arrow} #{label}: #{Float.round(from, 2)} → #{Float.round(to, 2)}"
        )
      end)
    else
      IO.puts("   Nenhuma mudança significativa detectada")
    end

    # Summary statistics
    IO.puts("\n📊 ESTATÍSTICAS DO PERÍODO")
    IO.puts("───────────────────────────────────────────────────────────────────")

    initial_avg = Enum.reduce(initial.avatars, 0, fn a, acc -> acc + a.wellbeing end) / length(initial.avatars)
    final_avg = Enum.reduce(final.avatars, 0, fn a, acc -> acc + a.wellbeing end) / length(final.avatars)

    mood_changes =
      Enum.count(changes, fn {_, _, _, _, im, fm} -> im != fm end)

    IO.puts("   Bem-estar médio inicial: #{Float.round(initial_avg * 100, 1)}%")
    IO.puts("   Bem-estar médio final:   #{Float.round(final_avg * 100, 1)}%")
    IO.puts("   Mudança média:           #{Float.round((final_avg - initial_avg) * 100, 1)}%")
    IO.puts("   Avatares que mudaram de humor: #{mood_changes}/#{length(changes)}")

    # Sentience indicators
    IO.puts("\n🧠 INDICADORES DE SENCIÊNCIA")
    IO.puts("───────────────────────────────────────────────────────────────────")

    desiring_count = Enum.count(final.avatars, fn a -> a.current_desire not in [:none, nil] end)
    mood_diversity = final.avatars |> Enum.map(& &1.mood) |> Enum.uniq() |> length()

    high_wellbeing = Enum.count(final.avatars, fn a -> a.wellbeing > 0.7 end)
    low_wellbeing = Enum.count(final.avatars, fn a -> a.wellbeing < 0.4 end)

    indicators = [
      {"Avatares com desejos ativos", desiring_count, length(final.avatars)},
      {"Diversidade de humores", mood_diversity, 10},
      {"Avatares felizes (>70%)", high_wellbeing, length(final.avatars)},
      {"Avatares infelizes (<40%)", low_wellbeing, length(final.avatars)},
      {"Mudanças de humor", mood_changes, length(final.avatars)}
    ]

    Enum.each(indicators, fn {label, value, max} ->
      pct = Float.round(value / max * 100, 0)
      bar = String.duplicate("█", round(pct / 5))
      IO.puts("   #{String.pad_trailing(label, 30)} #{value}/#{max} #{bar}")
    end)
  end
end

# Run the simulation
ticks = System.get_env("TICKS", "30") |> String.to_integer()
SimulationRunner.run(ticks)
