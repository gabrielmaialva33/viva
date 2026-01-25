# VIVA Simulation - Data Generator
# Run: ERL_LIBS=build/dev/erlang mix run scripts/run_simulation.exs

defmodule VivaSimulation do
  @moduledoc """
  Runs VIVA simulations and collects data for analysis.
  """

  # Configuration
  @num_vivas 50
  @total_ticks 1000
  @dt 0.016  # ~60fps
  @sample_interval 10  # Collect every N ticks
  @output_dir "data/simulations"

  def run do
    IO.puts("""

    ╔════════════════════════════════════════════════════════════╗
    ║           VIVA SIMULATION                                  ║
    ║           Generating data for analysis                     ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Create output directory
    File.mkdir_p!(@output_dir)

    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    run_id = "sim_#{timestamp}"

    IO.puts("Run ID: #{run_id}")
    IO.puts("VIVAs: #{@num_vivas}")
    IO.puts("Ticks: #{@total_ticks}")
    IO.puts("Sample interval: #{@sample_interval}")
    IO.puts("")

    # Start pool via Gleam (module: viva@soul_pool)
    {:ok, pool} = :viva@soul_pool.start()

    # Spawn VIVAs
    IO.puts("Spawning #{@num_vivas} VIVAs...")
    ids = :viva@soul_pool.spawn_many(pool, @num_vivas)
    IO.puts("IDs: #{inspect(Enum.take(ids, 5))}... (#{length(ids)} total)")

    # Add initial variation to each VIVA
    IO.puts("Adding initial variation...")
    Enum.each(ids, fn id ->
      dp = :rand.uniform() * 0.4 - 0.2  # -0.2 to 0.2
      da = :rand.uniform() * 0.4 - 0.2
      dd = :rand.uniform() * 0.4 - 0.2
      :viva@soul_pool.apply_delta(pool, id, dp, da, dd)
    end)
    # Let variation take effect
    :viva@soul_pool.tick_all(pool, @dt)

    # Collected data
    data = %{
      run_id: run_id,
      config: %{
        num_vivas: @num_vivas,
        total_ticks: @total_ticks,
        dt: @dt,
        sample_interval: @sample_interval
      },
      snapshots: [],
      events: [],
      stats: []
    }

    # Run simulation
    IO.puts("\nRunning simulation...")

    {final_data, elapsed_ms} = :timer.tc(fn ->
      run_simulation(pool, ids, data, 0)
    end) |> then(fn {time_us, result} -> {result, time_us / 1000} end)

    IO.puts("\nSimulation complete in #{Float.round(elapsed_ms, 2)}ms")

    # Save data
    save_data(final_data, run_id)

    # Final statistics
    print_summary(final_data)

    final_data
  end

  defp run_simulation(pool, ids, data, tick) when tick >= @total_ticks do
    # Final collection
    collect_snapshot(pool, ids, data, tick)
  end

  defp run_simulation(pool, ids, data, tick) do
    # Apply random stimuli (simulates environment)
    data = if rem(tick, 50) == 0 do
      stimulus = generate_stimulus(tick)
      :viva@soul_pool.apply_delta_all(pool, stimulus.dp, stimulus.da, stimulus.dd)

      event = %{
        tick: tick,
        type: :stimulus,
        stimulus: stimulus
      }
      %{data | events: [event | data.events]}
    else
      data
    end

    # Tick all souls
    :viva@soul_pool.tick_all(pool, @dt)

    # Collect data periodically
    data = if rem(tick, @sample_interval) == 0 do
      collect_snapshot(pool, ids, data, tick)
    else
      data
    end

    # Progress
    if rem(tick, 100) == 0 do
      IO.write("\r  Tick #{tick}/#{@total_ticks} (#{Float.round(tick / @total_ticks * 100, 1)}%)")
    end

    run_simulation(pool, ids, data, tick + 1)
  end

  defp collect_snapshot(pool, _ids, data, tick) do
    # Get all PAD states
    pads = :viva@soul_pool.get_all_pads(pool)

    # Convert Gleam dict (Erlang map) to list
    pad_list = pads
      |> Map.to_list()
      |> Enum.map(fn {id, pad_tuple} ->
        # Gleam Pad is {:pad, p, a, d} tuple
        {p, a, d} = case pad_tuple do
          {:Pad, p, a, d} -> {p, a, d}
          {:pad, p, a, d} -> {p, a, d}
          {p, a, d} -> {p, a, d}
          _ -> {0.0, 0.0, 0.0}
        end
        %{id: id, pleasure: p, arousal: a, dominance: d}
      end)

    # Calculate statistics
    stats = calculate_stats(pad_list)

    snapshot = %{
      tick: tick,
      pads: pad_list,
      stats: stats
    }

    %{data |
      snapshots: [snapshot | data.snapshots],
      stats: [%{tick: tick} |> Map.merge(stats) | data.stats]
    }
  end

  defp generate_stimulus(tick) do
    # Generate varied stimuli based on tick
    phase = rem(div(tick, 100), 4)

    case phase do
      0 -> %{dp: 0.3, da: 0.1, dd: 0.0, label: :joy}       # Joy wave
      1 -> %{dp: -0.2, da: 0.4, dd: -0.1, label: :stress}  # Stress
      2 -> %{dp: 0.1, da: -0.3, dd: 0.2, label: :calm}     # Dominant calm
      3 -> %{dp: -0.1, da: 0.2, dd: -0.3, label: :fear}    # Fear
    end
  end

  defp calculate_stats(pads) do
    n = length(pads)

    if n == 0 do
      %{mean_p: 0, mean_a: 0, mean_d: 0, std_p: 0, std_a: 0, std_d: 0}
    else
      # Means
      mean_p = Enum.sum(Enum.map(pads, & &1.pleasure)) / n
      mean_a = Enum.sum(Enum.map(pads, & &1.arousal)) / n
      mean_d = Enum.sum(Enum.map(pads, & &1.dominance)) / n

      # Standard deviations
      std_p = :math.sqrt(Enum.sum(Enum.map(pads, fn p -> :math.pow(p.pleasure - mean_p, 2) end)) / n)
      std_a = :math.sqrt(Enum.sum(Enum.map(pads, fn p -> :math.pow(p.arousal - mean_a, 2) end)) / n)
      std_d = :math.sqrt(Enum.sum(Enum.map(pads, fn p -> :math.pow(p.dominance - mean_d, 2) end)) / n)

      # Emotional distribution (quadrants)
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
        quadrants: quadrants
      }
    end
  end

  defp save_data(data, run_id) do
    # Reverse lists (built with prepend)
    data = %{data |
      snapshots: Enum.reverse(data.snapshots),
      events: Enum.reverse(data.events),
      stats: Enum.reverse(data.stats)
    }

    # Full JSON
    json_path = "#{@output_dir}/#{run_id}.json"
    File.write!(json_path, Jason.encode!(data, pretty: true))
    IO.puts("\nSaved: #{json_path}")

    # Stats CSV (easy to import)
    csv_path = "#{@output_dir}/#{run_id}_stats.csv"
    csv_content = stats_to_csv(data.stats)
    File.write!(csv_path, csv_content)
    IO.puts("Saved: #{csv_path}")

    # All PADs over time CSV (for visualization)
    pads_csv_path = "#{@output_dir}/#{run_id}_pads.csv"
    pads_csv = pads_to_csv(data.snapshots)
    File.write!(pads_csv_path, pads_csv)
    IO.puts("Saved: #{pads_csv_path}")
  end

  defp stats_to_csv(stats) do
    header = "tick,mean_p,mean_a,mean_d,std_p,std_a,std_d\n"
    rows = Enum.map(stats, fn s ->
      "#{s.tick},#{s.mean_p},#{s.mean_a},#{s.mean_d},#{s.std_p},#{s.std_a},#{s.std_d}"
    end) |> Enum.join("\n")
    header <> rows
  end

  defp pads_to_csv(snapshots) do
    header = "tick,viva_id,pleasure,arousal,dominance\n"
    rows = Enum.flat_map(snapshots, fn snap ->
      Enum.map(snap.pads, fn pad ->
        "#{snap.tick},#{pad.id},#{pad.pleasure},#{pad.arousal},#{pad.dominance}"
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

    IO.puts("\nEvents: #{length(data.events)}")
    IO.puts("Snapshots: #{length(data.snapshots)}")
    IO.puts("════════════════════════════════════════════════════════════")
  end
end

# Run!
VivaSimulation.run()
