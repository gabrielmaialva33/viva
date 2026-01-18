defmodule VivaBridge.Firmware.MetaLearner do
  @moduledoc """
  Autonomous firmware evolution trigger.

  Monitors VIVA's prediction accuracy (Free Energy) and triggers
  MAP-Elites evolution when performance degrades below threshold.

  ## Philosophy

  VIVA observes her own performance. When predictions become inaccurate
  (high Free Energy), she knows her body needs to adapt. The MetaLearner
  triggers this self-modification autonomously.

  ## Safety Constraints

  - Rate limited: max 1 evolution per hour
  - Minimum samples: needs 10+ readings before deciding
  - Cooldown after evolution: 5 minutes observation
  - Manual override: can be paused/resumed

  ## Metrics Tracked

  - Free Energy (lower = better predictions)
  - Model accuracy (0-1)
  - Prediction error variance
  - Evolution success rate
  """

  use GenServer
  require Logger

  alias VivaBridge.Firmware.Evolution
  alias VivaBridge.Music

  # Configuration
  @check_interval 60_000          # Check every 60 seconds
  @evolution_threshold 0.3        # Evolve if accuracy < 30%
  @free_energy_threshold 0.7      # Evolve if Free Energy > 0.7
  @min_samples 10                 # Need at least 10 samples
  @cooldown_after_evolution 300_000  # 5 min cooldown after evolution
  @rate_limit_ms 3_600_000        # Max 1 evolution per hour
  @evolution_iterations 50        # MAP-Elites iterations per run

  # ============================================================================
  # Public API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Get current MetaLearner status.
  """
  def status do
    GenServer.call(__MODULE__, :status)
  end

  @doc """
  Pause autonomous evolution (manual override).
  """
  def pause do
    GenServer.call(__MODULE__, :pause)
  end

  @doc """
  Resume autonomous evolution.
  """
  def resume do
    GenServer.call(__MODULE__, :resume)
  end

  @doc """
  Force an evolution cycle (bypasses thresholds, respects rate limit).
  """
  def force_evolution do
    GenServer.call(__MODULE__, :force_evolution, 120_000)
  end

  @doc """
  Add a performance sample manually.
  """
  def add_sample(free_energy, accuracy) do
    GenServer.cast(__MODULE__, {:add_sample, free_energy, accuracy})
  end

  @doc """
  Get evolution history.
  """
  def history do
    GenServer.call(__MODULE__, :history)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    # Optional: start paused
    paused = Keyword.get(opts, :paused, false)

    state = %{
      paused: paused,
      last_evolution: nil,
      evolution_count: 0,
      samples: [],
      max_samples: 100,
      history: [],
      current_archive: nil,
      best_fitness: 0.0,
      last_check: nil
    }

    # Schedule first check
    unless paused do
      schedule_check()
    end

    Logger.info("[MetaLearner] Started (paused=#{paused})")
    {:ok, state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    stats = compute_stats(state.samples)

    status = %{
      paused: state.paused,
      evolution_count: state.evolution_count,
      last_evolution: state.last_evolution,
      sample_count: length(state.samples),
      stats: stats,
      best_fitness: state.best_fitness,
      can_evolve: can_evolve?(state),
      time_until_next_evolution: time_until_can_evolve(state)
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call(:pause, _from, state) do
    Logger.info("[MetaLearner] Paused by user")
    {:reply, :ok, %{state | paused: true}}
  end

  @impl true
  def handle_call(:resume, _from, state) do
    Logger.info("[MetaLearner] Resumed by user")
    schedule_check()
    {:reply, :ok, %{state | paused: false}}
  end

  @impl true
  def handle_call(:force_evolution, _from, state) do
    if rate_limited?(state) do
      remaining = time_until_can_evolve(state)
      {:reply, {:error, {:rate_limited, remaining}}, state}
    else
      Logger.info("[MetaLearner] Forced evolution triggered")
      new_state = run_evolution(state, :forced)
      {:reply, {:ok, new_state.best_fitness}, new_state}
    end
  end

  @impl true
  def handle_call(:history, _from, state) do
    {:reply, state.history, state}
  end

  @impl true
  def handle_cast({:add_sample, free_energy, accuracy}, state) do
    sample = %{
      free_energy: free_energy,
      accuracy: accuracy,
      timestamp: System.system_time(:millisecond)
    }

    samples = [sample | state.samples] |> Enum.take(state.max_samples)
    {:noreply, %{state | samples: samples}}
  end

  @impl true
  def handle_info(:check, %{paused: true} = state) do
    # Don't check if paused
    {:noreply, state}
  end

  @impl true
  def handle_info(:check, state) do
    new_state = perform_check(state)
    schedule_check()
    {:noreply, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp schedule_check do
    Process.send_after(self(), :check, @check_interval)
  end

  defp perform_check(state) do
    # Collect current interoception data
    state = collect_sample(state)

    # Check if we should evolve
    stats = compute_stats(state.samples)

    state = %{state | last_check: DateTime.utc_now()}

    cond do
      # Not enough samples yet
      length(state.samples) < @min_samples ->
        Logger.debug("[MetaLearner] Collecting samples (#{length(state.samples)}/#{@min_samples})")
        state

      # Rate limited
      rate_limited?(state) ->
        Logger.debug("[MetaLearner] Rate limited, skipping check")
        state

      # In cooldown after evolution
      in_cooldown?(state) ->
        Logger.debug("[MetaLearner] In cooldown, skipping check")
        state

      # Performance is poor - trigger evolution
      should_evolve?(stats) ->
        Logger.info("[MetaLearner] Poor performance detected: accuracy=#{Float.round(stats.avg_accuracy, 3)}, F=#{Float.round(stats.avg_free_energy, 3)}")
        run_evolution(state, :automatic)

      # Everything is fine
      true ->
        Logger.debug("[MetaLearner] Performance OK: accuracy=#{Float.round(stats.avg_accuracy, 3)}")
        state
    end
  end

  defp collect_sample(state) do
    # Try to get interoception state from Music module
    case get_interoception_state() do
      {:ok, intero} ->
        sample = %{
          free_energy: intero.free_energy || 0.5,
          accuracy: intero.model_accuracy || 0.5,
          timestamp: System.system_time(:millisecond)
        }

        samples = [sample | state.samples] |> Enum.take(state.max_samples)
        %{state | samples: samples}

      {:error, _reason} ->
        # No data available, keep current samples
        state
    end
  end

  defp get_interoception_state do
    try do
      case Music.interoception_state() do
        %{} = state -> {:ok, state}
        _ -> {:error, :no_state}
      end
    rescue
      _ -> {:error, :not_available}
    catch
      :exit, _ -> {:error, :not_running}
    end
  end

  defp compute_stats([]), do: %{avg_accuracy: 0.5, avg_free_energy: 0.5, variance: 0.0}
  defp compute_stats(samples) do
    n = length(samples)

    avg_accuracy = Enum.sum(Enum.map(samples, & &1.accuracy)) / n
    avg_free_energy = Enum.sum(Enum.map(samples, & &1.free_energy)) / n

    # Compute variance of free energy
    variance = if n > 1 do
      squared_diffs = Enum.map(samples, fn s ->
        diff = s.free_energy - avg_free_energy
        diff * diff
      end)
      Enum.sum(squared_diffs) / (n - 1)
    else
      0.0
    end

    %{
      avg_accuracy: avg_accuracy,
      avg_free_energy: avg_free_energy,
      variance: variance,
      sample_count: n
    }
  end

  defp should_evolve?(stats) do
    stats.avg_accuracy < @evolution_threshold or
    stats.avg_free_energy > @free_energy_threshold
  end

  defp can_evolve?(state) do
    not rate_limited?(state) and not in_cooldown?(state)
  end

  defp rate_limited?(state) do
    case state.last_evolution do
      nil -> false
      last ->
        elapsed = System.system_time(:millisecond) - last
        elapsed < @rate_limit_ms
    end
  end

  defp in_cooldown?(state) do
    case state.last_evolution do
      nil -> false
      last ->
        elapsed = System.system_time(:millisecond) - last
        elapsed < @cooldown_after_evolution
    end
  end

  defp time_until_can_evolve(state) do
    case state.last_evolution do
      nil -> 0
      last ->
        elapsed = System.system_time(:millisecond) - last
        remaining = @rate_limit_ms - elapsed
        max(0, remaining)
    end
  end

  defp run_evolution(state, trigger) do
    Logger.info("[MetaLearner] Starting MAP-Elites evolution (trigger=#{trigger})")
    start_time = System.system_time(:millisecond)

    # Initialize or continue from existing archive
    archive = state.current_archive || Evolution.initialize()

    # Run evolution
    {:ok, final_archive, stats} = Evolution.run(
      archive: archive,
      iterations: @evolution_iterations,
      evaluate_live: false  # Use simulated for safety; enable evaluate_live: true for real hardware
    )

    elapsed = System.system_time(:millisecond) - start_time

    # Find best elite from exported data
    export_data = Evolution.export(final_archive)
    best = Enum.max_by(export_data.elites, & &1.fitness, fn -> nil end)
    best_fitness = if best, do: best.fitness, else: 0.0

    # Record in history
    history_entry = %{
      timestamp: DateTime.utc_now(),
      trigger: trigger,
      duration_ms: elapsed,
      iterations: @evolution_iterations,
      coverage: stats.coverage,
      best_fitness: best_fitness,
      filled_cells: stats.filled_cells
    }

    # Attempt to deploy best elite (if hardware available)
    deploy_result = maybe_deploy_best(best)

    # Save to Qdrant
    save_elites_to_memory(final_archive)

    Logger.info("[MetaLearner] Evolution complete: fitness=#{Float.round(best_fitness, 4)}, coverage=#{stats.coverage}%, deploy=#{inspect(deploy_result)}")

    %{state |
      last_evolution: System.system_time(:millisecond),
      evolution_count: state.evolution_count + 1,
      current_archive: final_archive,
      best_fitness: best_fitness,
      history: [history_entry | Enum.take(state.history, 99)],
      samples: []  # Clear samples after evolution
    }
  end

  defp maybe_deploy_best(nil), do: :no_elite

  defp maybe_deploy_best(%{genotype: genotype} = elite) do
    # Only deploy if Music module is connected
    if Music.connected?() do
      Logger.info("[MetaLearner] Deploying best elite (gen=#{genotype.generation}, fitness=#{Float.round(elite.fitness, 4)})")

      alias VivaBridge.Firmware.{Codegen, Uploader}

      ino_code = Codegen.generate(genotype)

      case Uploader.deploy(ino_code, generation: genotype.generation) do
        {:ok, result} ->
          Logger.info("[MetaLearner] Deploy successful: #{inspect(result)}")
          {:ok, result}

        {:error, reason} ->
          Logger.error("[MetaLearner] Deploy failed: #{inspect(reason)}")
          {:error, reason}
      end
    else
      Logger.debug("[MetaLearner] Skipping deploy (not connected)")
      :not_connected
    end
  end

  defp save_elites_to_memory(archive) do
    export_data = Evolution.export(archive)
    elites = export_data.elites

    Enum.each(elites, fn elite ->
      # Save to Qdrant with metadata
      info = """
      VIVA Evolved Firmware Elite
      Generation: #{elite.genotype.generation}
      Fitness: #{Float.round(elite.fitness, 4)}
      Energy Level: #{Float.round(elem(elite.descriptors, 0), 2)}
      Melody Complexity: #{Float.round(elem(elite.descriptors, 1), 2)}
      Emotions: #{inspect(Map.keys(elite.genotype.emotions))}
      Harmony Ratio: #{elite.genotype.harmony_ratio}
      """

      try do
        # Use Qdrant MCP if available
        save_to_qdrant(info, elite)
      rescue
        _ -> Logger.debug("[MetaLearner] Qdrant save skipped (not available)")
      end
    end)

    Logger.info("[MetaLearner] Saved #{length(elites)} elites to memory")
  end

  defp save_to_qdrant(info, elite) do
    # This would use the qdrant-memory MCP tool
    # For now, just log it
    metadata = %{
      type: "firmware_elite",
      project: "viva",
      fitness: elite.fitness,
      generation: elite.genotype.generation,
      energy: elem(elite.descriptors, 0),
      complexity: elem(elite.descriptors, 1)
    }

    Logger.debug("[MetaLearner] Would save to Qdrant: #{String.slice(info, 0, 100)}... metadata=#{inspect(metadata)}")
  end
end
