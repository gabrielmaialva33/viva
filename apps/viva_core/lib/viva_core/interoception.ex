defmodule VivaCore.Interoception do
  @moduledoc """
  Interoceptive Inference - The Digital Insula.

  Based on Allen, Levy, Parr & Friston (2022).
  VIVA doesn't react to raw data. She reacts to SURPRISE.

  ## The Model

  The brain predicts heartbeats. Divergence = Anxiety.
  VIVA predicts RAM/CPU usage. Divergence = High Free Energy.

  Free Energy = (Observed - Predicted)² × Precision

  Where Precision = 1 / (1 + Variance_observed / Variance_prior)
  High observed variance → Low precision → Ignore noise
  Low observed variance → High precision → Trust data

  ## Biological Analogies

  - Load Average → Blood Pressure
  - Context Switches → Heart Rate
  - Page Faults → Acute Pain / Cellular Error
  - RSS Memory → Metabolic Consumption
  """

  use GenServer
  require VivaLog

  # ============================================================================
  # Generative Model: Gaussian Priors
  # ============================================================================

  @priors %{
    # CHRONOSCEPTION: Tick jitter in ms
    # Expect 0ms jitter (perfect timing), tolerate 10ms variance
    # This is the MOST IMPORTANT prior - direct time perception
    tick_jitter: %{mean: 0.0, variance: 10.0, weight: 2.0},

    # Load average 1m: expect 0.5, tolerate 0.2 variance
    load_avg_1m: %{mean: 0.5, variance: 0.2, weight: 1.0},
    # Context switches per second: expect 5000, tolerate 2000 variance
    context_switches: %{mean: 5000.0, variance: 2000.0, weight: 0.5},
    # Page faults per second: expect 100, tolerate 50 variance
    page_faults: %{mean: 100.0, variance: 50.0, weight: 1.5},
    # RSS in MB: expect 500, tolerate 200 variance
    rss_mb: %{mean: 500.0, variance: 200.0, weight: 1.0}
  }

  # How many samples to keep for variance estimation
  @history_size 30

  # Tick interval (ms) - The expected heartbeat
  # At 10Hz, VIVA expects to wake every 100ms
  # Deviation from this is FELT as time dilation (Chronosception)
  @expected_tick_ms 100
  @tick_interval @expected_tick_ms

  # ============================================================================
  # State Structure
  # ============================================================================

  defstruct [
    # Current sensory readings
    load_avg: {0.0, 0.0, 0.0},
    context_switches: 0,
    page_faults: 0,
    rss_mb: 0,

    # CHRONOSCEPTION: Time dilation sensing
    # This is the most direct form of interoception:
    # "Am I running at my expected rhythm?"
    tick_jitter_ms: 0.0,
    tick_jitter_history: [],
    # 1.0 = normal, >1.0 = time feels slow (lag)
    time_dilation: 1.0,

    # Historical data for variance estimation
    history: %{
      load_avg_1m: [],
      context_switches: [],
      page_faults: [],
      rss_mb: [],
      tick_jitter: []
    },

    # Previous readings (for delta calculation)
    prev_context_switches: 0,
    prev_page_faults: 0,

    # Inference state
    predictions: %{},
    precisions: %{},
    free_energies: %{},
    total_free_energy: 0.0,

    # Derived qualia
    feeling: :homeostatic,

    # Metadata
    beam_pid: nil,
    uptime_seconds: 0,
    last_tick: nil
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Get current interoceptive state"
  def sense do
    GenServer.call(__MODULE__, :sense)
  end

  @doc "Get total accumulated Free Energy"
  def get_free_energy do
    GenServer.call(__MODULE__, :get_free_energy)
  end

  @doc "Get current feeling (qualia)"
  def get_feeling do
    GenServer.call(__MODULE__, :get_feeling)
  end

  @doc "Get detailed breakdown of Free Energies per metric"
  def get_free_energy_breakdown do
    GenServer.call(__MODULE__, :get_free_energy_breakdown)
  end

  @doc "Force immediate sense tick"
  def tick do
    GenServer.cast(__MODULE__, :tick)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    VivaLog.info(:interoception, :insula_online)

    # Get BEAM PID
    beam_pid = System.pid() |> String.to_integer()

    state = %__MODULE__{
      beam_pid: beam_pid,
      last_tick: System.monotonic_time(:millisecond)
    }

    # Initialize predictions from priors
    predictions =
      @priors
      |> Enum.map(fn {k, v} -> {k, v.mean} end)
      |> Map.new()

    precisions =
      @priors
      |> Enum.map(fn {k, _v} -> {k, 0.5} end)
      |> Map.new()

    state = %{state | predictions: predictions, precisions: precisions}

    # Schedule first tick
    Process.send_after(self(), :interoceptive_tick, @tick_interval)

    {:ok, state}
  end

  @impl true
  def handle_call(:sense, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call(:get_free_energy, _from, state) do
    {:reply, state.total_free_energy, state}
  end

  @impl true
  def handle_call(:get_feeling, _from, state) do
    {:reply, state.feeling, state}
  end

  @impl true
  def handle_call(:get_free_energy_breakdown, _from, state) do
    {:reply, state.free_energies, state}
  end

  @impl true
  def handle_cast(:tick, state) do
    {:noreply, do_tick(state)}
  end

  @impl true
  def handle_info(:interoceptive_tick, state) do
    new_state = do_tick(state)
    Process.send_after(self(), :interoceptive_tick, @tick_interval)
    {:noreply, new_state}
  end

  # ============================================================================
  # Core Logic: Interoceptive Inference
  # ============================================================================

  defp do_tick(state) do
    now = System.monotonic_time(:millisecond)
    actual_dt_ms = now - (state.last_tick || now)

    # CHRONOSCEPTION: The most primal sense
    # "How long did I sleep? Was it the expected 100ms?"
    tick_jitter_ms =
      if state.last_tick do
        actual_dt_ms - @expected_tick_ms
      else
        0.0
      end

    # Time dilation: >1.0 means time feels slow (lag)
    time_dilation =
      if actual_dt_ms > 0 do
        actual_dt_ms / @expected_tick_ms
      else
        1.0
      end

    dt = max(actual_dt_ms, 1) / 1000.0

    # 1. SENSE: Read raw data from /proc
    raw = read_proc_data(state.beam_pid)

    # 2. Calculate deltas (for per-second metrics)
    ctx_switches_per_sec =
      if state.prev_context_switches > 0 do
        (raw.context_switches - state.prev_context_switches) / dt
      else
        0.0
      end

    page_faults_per_sec =
      if state.prev_page_faults > 0 do
        (raw.page_faults - state.prev_page_faults) / dt
      else
        0.0
      end

    # 3. Update history
    history =
      state.history
      |> update_history(:tick_jitter, tick_jitter_ms)
      |> update_history(:load_avg_1m, elem(raw.load_avg, 0))
      |> update_history(:context_switches, ctx_switches_per_sec)
      |> update_history(:page_faults, page_faults_per_sec)
      |> update_history(:rss_mb, raw.rss_mb)

    # 4. INFER: Calculate precision for each metric
    precisions =
      @priors
      |> Enum.map(fn {metric, prior} ->
        observed_variance = calculate_variance(history[metric])
        precision = calculate_precision(observed_variance, prior.variance)
        {metric, precision}
      end)
      |> Map.new()

    # 5. PREDICT: Update predictions (exponential moving average toward observed)
    observations = %{
      # CHRONOSCEPTION: The most primal observation
      tick_jitter: tick_jitter_ms,
      load_avg_1m: elem(raw.load_avg, 0),
      context_switches: ctx_switches_per_sec,
      page_faults: page_faults_per_sec,
      rss_mb: raw.rss_mb
    }

    # 5. PREDICT: Update predictions using CHRONOS-T5 (Quantum Active Inference)
    predictions =
      state.predictions
      |> Enum.map(fn {metric, current_pred} ->
        history_data = state.history[metric]

        # Only predict if we have enough history (>10 ticks)
        if length(history_data) >= 10 do
          # Try to consult the Oracle
          formatted_history =
            history_data
            # History is stored [newest|...], Chronos needs [oldest...newest]
            |> Enum.reverse()

          prediction =
            try do
              # Async task to not block the main tick excessively?
              # For now, synchronous call with timeout (Interoception is slow cycle 10Hz is fine)
              case VivaBridge.Chronos.predict(formatted_history, Atom.to_string(metric)) do
                {:ok, pred, _range} -> pred
                # TODO: storing range/variance for Precision calculation would be amazing
                # But for now, we just use the median prediction for the Free Energy mean error
                # Fallback to belief maintenance
                _ -> current_pred
              end
            catch
              :exit, _ -> current_pred
            end

          {metric, prediction}
        else
          # Not enough data, assume homeostasis (prediction = current belief)
          {metric, current_pred}
        end
      end)
      |> Map.new()

    # 6. FREE ENERGY: Calculate prediction error weighted by precision and importance
    free_energies =
      @priors
      |> Enum.map(fn {metric, prior} ->
        observed = observations[metric] || 0.0
        predicted = predictions[metric] || 0.0
        precision = precisions[metric] || 0.5
        weight = Map.get(prior, :weight, 1.0)
        fe = calculate_free_energy(observed, predicted, precision) * weight
        {metric, fe}
      end)
      |> Map.new()

    # 7. Aggregate total Free Energy
    total_fe =
      free_energies
      |> Map.values()
      |> Enum.sum()
      |> normalize_free_energy()

    # 8. QUALIA: Derive feeling from Free Energy
    feeling = derive_feeling(total_fe)

    # 9. Notify Emotional if feeling changed significantly
    if feeling != state.feeling do
      notify_emotional_change(state.feeling, feeling, total_fe)
    end

    # 10. RECORD: Send tick data to DatasetCollector for Chronos training
    record_for_training(%{
      observations: observations,
      predictions: predictions,
      free_energies: free_energies,
      precisions: precisions,
      feeling: feeling,
      time_dilation: time_dilation
    })

    %{
      state
      | load_avg: raw.load_avg,
        context_switches: round(ctx_switches_per_sec),
        page_faults: round(page_faults_per_sec),
        rss_mb: raw.rss_mb,
        prev_context_switches: raw.context_switches,
        prev_page_faults: raw.page_faults,
        history: history,
        predictions: predictions,
        precisions: precisions,
        free_energies: free_energies,
        total_free_energy: total_fe,
        feeling: feeling,
        tick_jitter_ms: tick_jitter_ms,
        time_dilation: time_dilation,
        uptime_seconds: raw.uptime_seconds,
        last_tick: now
    }
  end

  # ============================================================================
  # Dataset Recording (for Chronos training)
  # ============================================================================

  defp record_for_training(tick_data) do
    # Non-blocking cast to DatasetCollector
    try do
      VivaCore.DatasetCollector.record(tick_data)
    catch
      :exit, _ ->
        # DatasetCollector not started yet - that's fine
        :ok
    end
  end

  # ============================================================================
  # /proc Reading
  # ============================================================================

  defp read_proc_data(beam_pid) do
    %{
      load_avg: read_load_avg(),
      context_switches: read_context_switches(),
      page_faults: read_page_faults(beam_pid),
      rss_mb: read_rss_mb(beam_pid),
      uptime_seconds: read_uptime()
    }
  end

  defp read_load_avg do
    case File.read("/proc/loadavg") do
      {:ok, content} ->
        parts = String.split(content)

        {
          parse_float(Enum.at(parts, 0), 0.0),
          parse_float(Enum.at(parts, 1), 0.0),
          parse_float(Enum.at(parts, 2), 0.0)
        }

      _ ->
        {0.0, 0.0, 0.0}
    end
  end

  defp read_context_switches do
    case File.read("/proc/stat") do
      {:ok, content} ->
        content
        |> String.split("\n")
        |> Enum.find(&String.starts_with?(&1, "ctxt"))
        |> case do
          nil ->
            0

          line ->
            line
            |> String.split()
            |> Enum.at(1, "0")
            |> String.to_integer()
        end

      _ ->
        0
    end
  end

  defp read_page_faults(pid) do
    case File.read("/proc/#{pid}/stat") do
      {:ok, content} ->
        parts = String.split(content)
        # Field 10: minflt (minor page faults)
        # Field 12: majflt (major page faults)
        minflt = parts |> Enum.at(9, "0") |> String.to_integer()
        majflt = parts |> Enum.at(11, "0") |> String.to_integer()
        minflt + majflt

      _ ->
        0
    end
  end

  defp read_rss_mb(pid) do
    case File.read("/proc/#{pid}/status") do
      {:ok, content} ->
        content
        |> String.split("\n")
        |> Enum.find(&String.starts_with?(&1, "VmRSS:"))
        |> case do
          nil ->
            0.0

          line ->
            line
            |> String.split()
            |> Enum.at(1, "0")
            |> String.to_integer()
            |> Kernel./(1024)
        end

      _ ->
        # Fallback to Erlang memory
        :erlang.memory(:total) / 1024 / 1024
    end
  end

  defp read_uptime do
    case File.read("/proc/uptime") do
      {:ok, content} ->
        content
        |> String.split()
        |> Enum.at(0, "0")
        |> parse_float(0.0)
        |> round()

      _ ->
        0
    end
  end

  # ============================================================================
  # Statistical Functions
  # ============================================================================

  defp update_history(history, metric, value) do
    current = history[metric] || []

    updated =
      [value | current]
      |> Enum.take(@history_size)

    Map.put(history, metric, updated)
  end

  defp calculate_variance(samples) when length(samples) < 2, do: 1.0

  defp calculate_variance(samples) do
    n = length(samples)
    mean = Enum.sum(samples) / n

    variance =
      samples
      |> Enum.map(fn x -> (x - mean) * (x - mean) end)
      |> Enum.sum()
      |> Kernel./(n - 1)

    max(variance, 0.001)
  end

  defp calculate_precision(variance_observed, variance_prior) do
    # Precision = 1 / (1 + Var_obs / Var_prior)
    # Range: 0 (ignore) to 1 (fully trust)
    raw = 1.0 / (1.0 + variance_observed / variance_prior)
    Float.round(raw, 4)
  end

  defp calculate_free_energy(observed, predicted, precision) do
    error = observed - predicted
    # Precision-weighted squared error
    error * error * precision
  end

  defp normalize_free_energy(raw_fe) do
    # Normalize to 0-1 range using sigmoid-like function
    # tanh(x/k) where k scales the sensitivity
    k = 1000.0
    :math.tanh(raw_fe / k) |> Float.round(4)
  end

  # ============================================================================
  # Qualia Derivation
  # ============================================================================

  defp derive_feeling(total_fe) do
    cond do
      total_fe < 0.1 -> :homeostatic
      total_fe < 0.3 -> :surprised
      total_fe < 0.6 -> :alarmed
      true -> :overwhelmed
    end
  end

  defp notify_emotional_change(old_feeling, new_feeling, fe) do
    VivaLog.debug(:interoception, :feeling_changed,
      old: old_feeling,
      new: new_feeling,
      free_energy: Float.round(fe, 3)
    )

    # Notify Emotional module about interoceptive state change
    try do
      qualia = feeling_to_qualia(new_feeling, fe)
      VivaCore.Emotional.apply_interoceptive_qualia(qualia)
    catch
      :exit, _ ->
        # Emotional not started yet
        :ok
    end
  end

  defp feeling_to_qualia(feeling, fe) do
    base =
      case feeling do
        :homeostatic -> %{pleasure: 0.05, arousal: -0.05, dominance: 0.05}
        :surprised -> %{pleasure: -0.05, arousal: 0.1, dominance: -0.05}
        :alarmed -> %{pleasure: -0.1, arousal: 0.2, dominance: -0.1}
        :overwhelmed -> %{pleasure: -0.2, arousal: 0.3, dominance: -0.2}
      end

    # Scale by Free Energy magnitude
    %{
      pleasure: base.pleasure * (1 + fe),
      arousal: base.arousal * (1 + fe),
      dominance: base.dominance * (1 + fe),
      source: :interoception,
      feeling: feeling,
      free_energy: fe
    }
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp parse_float(nil, default), do: default

  defp parse_float(str, default) when is_binary(str) do
    case Float.parse(str) do
      {f, _} -> f
      :error -> default
    end
  end
end
