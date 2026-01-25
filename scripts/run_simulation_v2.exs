# VIVA Simulation v2 - Extended with more variation
# Run: ERL_LIBS=build/dev/erlang mix run scripts/run_simulation_v2.exs

defmodule VivaSimulationV2 do
  @moduledoc """
  Extended VIVA simulation with individual variation and complex stimuli.
  """

  # Configuration
  @num_vivas 100
  @total_ticks 5000
  @dt 0.016
  @sample_interval 25
  @output_dir "data/simulations"

  def run do
    IO.puts("""

    ╔════════════════════════════════════════════════════════════╗
    ║           VIVA SIMULATION v2                               ║
    ║           Extended run with variation                      ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    File.mkdir_p!(@output_dir)
    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    run_id = "sim_v2_#{timestamp}"

    IO.puts("Run ID: #{run_id}")
    IO.puts("VIVAs: #{@num_vivas}")
    IO.puts("Ticks: #{@total_ticks}")
    IO.puts("Sample interval: #{@sample_interval}")
    IO.puts("")

    {:ok, pool} = :viva@soul_pool.start()

    # Spawn VIVAs
    IO.puts("Spawning #{@num_vivas} VIVAs...")
    ids = :viva@soul_pool.spawn_many(pool, @num_vivas)

    # Assign personality types to each VIVA
    personalities = assign_personalities(ids)
    IO.puts("Personality distribution:")
    personalities |> Enum.frequencies_by(fn {_, p} -> p end) |> Enum.each(fn {p, c} ->
      IO.puts("  #{p}: #{c}")
    end)

    # Add initial variation based on personality
    IO.puts("\nInitializing with personality-based variation...")
    Enum.each(personalities, fn {id, personality} ->
      {dp, da, dd} = initial_state_for_personality(personality)
      :viva@soul_pool.apply_delta(pool, id, dp, da, dd)
    end)
    :viva@soul_pool.tick_all(pool, @dt)

    data = %{
      run_id: run_id,
      config: %{
        num_vivas: @num_vivas,
        total_ticks: @total_ticks,
        dt: @dt,
        sample_interval: @sample_interval
      },
      personalities: Map.new(personalities),
      snapshots: [],
      events: [],
      stats: []
    }

    IO.puts("\nRunning simulation...")
    start_time = System.monotonic_time(:millisecond)

    final_data = run_simulation(pool, ids, personalities, data, 0)

    elapsed = System.monotonic_time(:millisecond) - start_time
    IO.puts("\n\nSimulation complete in #{elapsed}ms")

    save_data(final_data, run_id, personalities)
    print_summary(final_data)

    final_data
  end

  defp assign_personalities(ids) do
    # 5 personality types based on Big Five tendencies
    types = [:optimist, :neurotic, :calm, :energetic, :balanced]

    Enum.map(ids, fn id ->
      # Weighted distribution
      type = case :rand.uniform(100) do
        n when n <= 20 -> :optimist    # 20%
        n when n <= 35 -> :neurotic    # 15%
        n when n <= 55 -> :calm        # 20%
        n when n <= 75 -> :energetic   # 20%
        _ -> :balanced                  # 25%
      end
      {id, type}
    end)
  end

  defp initial_state_for_personality(personality) do
    noise = fn -> :rand.uniform() * 0.1 - 0.05 end

    case personality do
      :optimist ->   {0.3 + noise.(), 0.1 + noise.(), 0.1 + noise.()}
      :neurotic ->   {-0.1 + noise.(), 0.3 + noise.(), -0.2 + noise.()}
      :calm ->       {0.1 + noise.(), -0.2 + noise.(), 0.1 + noise.()}
      :energetic ->  {0.1 + noise.(), 0.3 + noise.(), 0.2 + noise.()}
      :balanced ->   {0.0 + noise.(), 0.0 + noise.(), 0.0 + noise.()}
    end
  end

  defp run_simulation(_pool, _ids, _personalities, data, tick) when tick >= @total_ticks do
    data
  end

  defp run_simulation(pool, ids, personalities, data, tick) do
    # Global events (affect everyone)
    data = if rem(tick, 500) == 0 and tick > 0 do
      event = generate_global_event(tick)
      :viva@soul_pool.apply_delta_all(pool, event.dp, event.da, event.dd)
      %{data | events: [%{tick: tick, type: :global, event: event} | data.events]}
    else
      data
    end

    # Individual random stimuli (personality-influenced)
    if rem(tick, 20) == 0 do
      Enum.each(personalities, fn {id, personality} ->
        if :rand.uniform() < 0.3 do  # 30% chance each tick
          {dp, da, dd} = random_stimulus_for_personality(personality)
          :viva@soul_pool.apply_delta(pool, id, dp, da, dd)
        end
      end)
    end

    # Environmental noise (small random perturbations)
    if rem(tick, 5) == 0 do
      Enum.each(ids, fn id ->
        dp = (:rand.uniform() - 0.5) * 0.02
        da = (:rand.uniform() - 0.5) * 0.02
        dd = (:rand.uniform() - 0.5) * 0.02
        :viva@soul_pool.apply_delta(pool, id, dp, da, dd)
      end)
    end

    # Tick
    :viva@soul_pool.tick_all(pool, @dt)

    # Collect data
    data = if rem(tick, @sample_interval) == 0 do
      collect_snapshot(pool, personalities, data, tick)
    else
      data
    end

    # Progress
    if rem(tick, 500) == 0 do
      pct = Float.round(tick / @total_ticks * 100, 1)
      IO.write("\r  Tick #{tick}/#{@total_ticks} (#{pct}%)")
    end

    run_simulation(pool, ids, personalities, data, tick + 1)
  end

  defp generate_global_event(tick) do
    # Cycle through different scenarios
    phase = rem(div(tick, 500), 10)

    case phase do
      0 -> %{dp: 0.2, da: 0.1, dd: 0.0, label: :good_news}
      1 -> %{dp: -0.15, da: 0.3, dd: -0.1, label: :crisis}
      2 -> %{dp: 0.1, da: -0.2, dd: 0.1, label: :resolution}
      3 -> %{dp: 0.0, da: 0.2, dd: 0.2, label: :challenge}
      4 -> %{dp: 0.15, da: -0.1, dd: 0.0, label: :relaxation}
      5 -> %{dp: -0.1, da: 0.1, dd: -0.2, label: :uncertainty}
      6 -> %{dp: 0.2, da: 0.2, dd: 0.1, label: :celebration}
      7 -> %{dp: -0.2, da: -0.1, dd: 0.0, label: :disappointment}
      8 -> %{dp: 0.0, da: -0.2, dd: 0.2, label: :empowerment}
      9 -> %{dp: 0.1, da: 0.0, dd: -0.1, label: :reflection}
    end
  end

  defp random_stimulus_for_personality(personality) do
    # Personality affects how stimuli are processed
    base = (:rand.uniform() - 0.5) * 0.2

    case personality do
      :optimist ->   {base + 0.05, base * 0.5, base * 0.3}
      :neurotic ->   {base - 0.03, base + 0.1, base - 0.05}
      :calm ->       {base * 0.5, base * 0.3, base * 0.5}
      :energetic ->  {base, base + 0.05, base + 0.03}
      :balanced ->   {base, base, base}
    end
  end

  defp collect_snapshot(pool, personalities, data, tick) do
    pads = :viva@soul_pool.get_all_pads(pool)

    pad_list = pads
      |> Map.to_list()
      |> Enum.map(fn {id, pad_tuple} ->
        {p, a, d} = case pad_tuple do
          {:Pad, p, a, d} -> {p, a, d}
          {:pad, p, a, d} -> {p, a, d}
          {p, a, d} -> {p, a, d}
          _ -> {0.0, 0.0, 0.0}
        end
        personality = Map.get(Map.new(personalities), id, :unknown)
        %{id: id, pleasure: p, arousal: a, dominance: d, personality: personality}
      end)

    # Overall stats
    overall_stats = calculate_stats(pad_list)

    # Stats by personality
    by_personality = pad_list
      |> Enum.group_by(& &1.personality)
      |> Enum.map(fn {pers, pads} ->
        stats = calculate_stats(pads)
        {pers, stats}
      end)
      |> Map.new()

    snapshot = %{
      tick: tick,
      pads: pad_list,
      overall: overall_stats,
      by_personality: by_personality
    }

    %{data |
      snapshots: [snapshot | data.snapshots],
      stats: [%{tick: tick} |> Map.merge(overall_stats) | data.stats]
    }
  end

  defp calculate_stats(pads) do
    n = length(pads)
    if n == 0 do
      %{mean_p: 0, mean_a: 0, mean_d: 0, std_p: 0, std_a: 0, std_d: 0, n: 0}
    else
      mean_p = Enum.sum(Enum.map(pads, & &1.pleasure)) / n
      mean_a = Enum.sum(Enum.map(pads, & &1.arousal)) / n
      mean_d = Enum.sum(Enum.map(pads, & &1.dominance)) / n

      std_p = :math.sqrt(Enum.sum(Enum.map(pads, fn p -> :math.pow(p.pleasure - mean_p, 2) end)) / n)
      std_a = :math.sqrt(Enum.sum(Enum.map(pads, fn p -> :math.pow(p.arousal - mean_a, 2) end)) / n)
      std_d = :math.sqrt(Enum.sum(Enum.map(pads, fn p -> :math.pow(p.dominance - mean_d, 2) end)) / n)

      quadrants = Enum.frequencies_by(pads, fn p ->
        cond do
          p.pleasure > 0 and p.arousal > 0 -> :excited_happy
          p.pleasure > 0 and p.arousal <= 0 -> :calm_happy
          p.pleasure <= 0 and p.arousal > 0 -> :stressed
          true -> :depressed
        end
      end)

      %{
        mean_p: Float.round(mean_p, 4),
        mean_a: Float.round(mean_a, 4),
        mean_d: Float.round(mean_d, 4),
        std_p: Float.round(std_p, 4),
        std_a: Float.round(std_a, 4),
        std_d: Float.round(std_d, 4),
        n: n,
        quadrants: quadrants
      }
    end
  end

  defp save_data(data, run_id, personalities) do
    data = %{data |
      snapshots: Enum.reverse(data.snapshots),
      events: Enum.reverse(data.events),
      stats: Enum.reverse(data.stats)
    }

    # Full JSON
    json_path = "#{@output_dir}/#{run_id}.json"
    File.write!(json_path, Jason.encode!(data, pretty: true))
    IO.puts("\nSaved: #{json_path}")

    # Stats CSV
    csv_path = "#{@output_dir}/#{run_id}_stats.csv"
    File.write!(csv_path, stats_to_csv(data.stats))
    IO.puts("Saved: #{csv_path}")

    # PADs CSV with personality
    pads_csv_path = "#{@output_dir}/#{run_id}_pads.csv"
    File.write!(pads_csv_path, pads_to_csv(data.snapshots))
    IO.puts("Saved: #{pads_csv_path}")

    # Personality mapping CSV
    pers_csv_path = "#{@output_dir}/#{run_id}_personalities.csv"
    pers_csv = "viva_id,personality\n" <>
      (personalities |> Enum.map(fn {id, p} -> "#{id},#{p}" end) |> Enum.join("\n"))
    File.write!(pers_csv_path, pers_csv)
    IO.puts("Saved: #{pers_csv_path}")

    # Events CSV
    events_csv_path = "#{@output_dir}/#{run_id}_events.csv"
    events_csv = "tick,type,label,dp,da,dd\n" <>
      (data.events |> Enum.map(fn e ->
        ev = e.event
        "#{e.tick},#{e.type},#{ev.label},#{ev.dp},#{ev.da},#{ev.dd}"
      end) |> Enum.join("\n"))
    File.write!(events_csv_path, events_csv)
    IO.puts("Saved: #{events_csv_path}")
  end

  defp stats_to_csv(stats) do
    header = "tick,mean_p,mean_a,mean_d,std_p,std_a,std_d\n"
    rows = Enum.map(stats, fn s ->
      "#{s.tick},#{s.mean_p},#{s.mean_a},#{s.mean_d},#{s.std_p},#{s.std_a},#{s.std_d}"
    end) |> Enum.join("\n")
    header <> rows
  end

  defp pads_to_csv(snapshots) do
    header = "tick,viva_id,pleasure,arousal,dominance,personality\n"
    rows = Enum.flat_map(snapshots, fn snap ->
      Enum.map(snap.pads, fn pad ->
        "#{snap.tick},#{pad.id},#{pad.pleasure},#{pad.arousal},#{pad.dominance},#{pad.personality}"
      end)
    end) |> Enum.join("\n")
    header <> rows
  end

  defp print_summary(data) do
    IO.puts("""

    ════════════════════════════════════════════════════════════
    SIMULATION SUMMARY
    ════════════════════════════════════════════════════════════
    """)

    first_stats = List.first(data.stats)
    last_stats = List.last(data.stats)

    IO.puts("Initial state:")
    IO.puts("  P: #{first_stats.mean_p} (std: #{first_stats.std_p})")
    IO.puts("  A: #{first_stats.mean_a} (std: #{first_stats.std_a})")
    IO.puts("  D: #{first_stats.mean_d} (std: #{first_stats.std_d})")

    IO.puts("\nFinal state:")
    IO.puts("  P: #{last_stats.mean_p} (std: #{last_stats.std_p})")
    IO.puts("  A: #{last_stats.mean_a} (std: #{last_stats.std_a})")
    IO.puts("  D: #{last_stats.mean_d} (std: #{last_stats.std_d})")

    if last_stats[:quadrants] do
      IO.puts("\nFinal emotional distribution:")
      Enum.each(last_stats.quadrants, fn {quad, count} ->
        pct = Float.round(count / data.config.num_vivas * 100, 1)
        IO.puts("  #{quad}: #{count} (#{pct}%)")
      end)
    end

    IO.puts("\nGlobal events: #{length(data.events)}")
    IO.puts("Snapshots: #{length(data.snapshots)}")
    IO.puts("════════════════════════════════════════════════════════════")
  end
end

VivaSimulationV2.run()
