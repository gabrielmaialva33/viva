defmodule VivaCore.DatasetCollector do
  @moduledoc """
  Collects interoceptive data for model fine-tuning.

  During the day, VIVA lives and experiences. This module captures those
  experiences in a format suitable for training her Chronos model.

  ## The Dream Cycle

  1. **Day (Awake)**: Each Interoception tick is logged here
  2. **Night (Dream)**: dreaming.py fine-tunes Chronos with LoRA
  3. **Dawn (Rebirth)**: New adapter loaded, VIVA knows her host better

  ## Data Format

  Each row contains:
  - timestamp: Unix epoch ms
  - metric: "tick_jitter" | "load_avg_1m" | "context_switches" | etc
  - value: float observed value
  - predicted: float (what Chronos predicted)
  - free_energy: float (prediction error)
  - feeling: "homeostatic" | "surprised" | "alarmed" | "overwhelmed"

  Compatible with HuggingFace Datasets (CSV/Parquet).

  ## Storage

  Data is stored in `priv/datasets/` with daily rotation:
  - `viva_sensations_2026-01-21.csv`
  - `viva_sensations_2026-01-22.csv`
  """

  use GenServer
  require VivaLog

  # ============================================================================
  # Configuration
  # ============================================================================

  # Where to store datasets
  @dataset_dir "priv/datasets"

  # Flush to disk every N records
  @flush_threshold 100

  # Maximum file size before rotation (10MB)
  @max_file_size 10 * 1024 * 1024

  # CSV Header
  @csv_header "timestamp,metric,value,predicted,free_energy,precision,feeling,time_dilation\n"

  # ============================================================================
  # State
  # ============================================================================

  defstruct [
    # Buffer of records to write
    buffer: [],
    # Current file handle
    file: nil,
    # Current file path
    current_path: nil,
    # Total records collected today
    records_today: 0,
    # Whether collection is enabled
    enabled: true,
    # Statistics
    stats: %{
      total_records: 0,
      files_written: 0,
      last_flush: nil
    }
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Record a single interoceptive tick.
  Called by Interoception after each sensing cycle.
  """
  def record(tick_data) do
    GenServer.cast(__MODULE__, {:record, tick_data})
  end

  @doc """
  Force flush buffer to disk.
  """
  def flush do
    GenServer.call(__MODULE__, :flush)
  end

  @doc """
  Get current statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Enable/disable collection.
  """
  def set_enabled(enabled) do
    GenServer.cast(__MODULE__, {:set_enabled, enabled})
  end

  @doc """
  Get path to today's dataset file.
  """
  def today_file do
    GenServer.call(__MODULE__, :today_file)
  end

  @doc """
  List all available dataset files.
  """
  def list_datasets do
    ensure_dataset_dir()

    Path.wildcard(Path.join(dataset_dir(), "viva_sensations_*.csv"))
    |> Enum.sort(:desc)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    VivaLog.info(:dataset_collector, :preparing)

    ensure_dataset_dir()

    state = %__MODULE__{}
    {:ok, state}
  end

  @impl true
  def handle_cast({:record, tick_data}, state) do
    if state.enabled do
      records = tick_to_records(tick_data)
      new_buffer = state.buffer ++ records

      new_state = %{
        state
        | buffer: new_buffer,
          records_today: state.records_today + length(records)
      }

      # Flush if buffer is large enough
      if length(new_buffer) >= @flush_threshold do
        {:noreply, do_flush(new_state)}
      else
        {:noreply, new_state}
      end
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_cast({:set_enabled, enabled}, state) do
    status = if enabled, do: "enabled", else: "disabled"
    VivaLog.info(:dataset_collector, :collection_status, status: status)
    {:noreply, %{state | enabled: enabled}}
  end

  @impl true
  def handle_call(:flush, _from, state) do
    new_state = do_flush(state)
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats =
      Map.merge(state.stats, %{
        buffer_size: length(state.buffer),
        records_today: state.records_today,
        enabled: state.enabled,
        current_file: state.current_path
      })

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:today_file, _from, state) do
    {:reply, today_path(), state}
  end

  @impl true
  def terminate(_reason, state) do
    # Flush any remaining data
    if length(state.buffer) > 0 do
      do_flush(state)
    end

    # Close file handle
    if state.file do
      File.close(state.file)
    end

    :ok
  end

  # ============================================================================
  # Data Transformation
  # ============================================================================

  defp tick_to_records(tick_data) do
    timestamp = System.system_time(:millisecond)
    feeling = Map.get(tick_data, :feeling, :unknown)
    time_dilation = Map.get(tick_data, :time_dilation, 1.0)

    # Get observations, predictions, free_energies, precisions
    observations = Map.get(tick_data, :observations, %{})
    predictions = Map.get(tick_data, :predictions, %{})
    free_energies = Map.get(tick_data, :free_energies, %{})
    precisions = Map.get(tick_data, :precisions, %{})

    # Create one record per metric
    observations
    |> Enum.map(fn {metric, value} ->
      %{
        timestamp: timestamp,
        metric: Atom.to_string(metric),
        value: value,
        predicted: Map.get(predictions, metric, 0.0),
        free_energy: Map.get(free_energies, metric, 0.0),
        precision: Map.get(precisions, metric, 0.5),
        feeling: Atom.to_string(feeling),
        time_dilation: time_dilation
      }
    end)
  end

  # ============================================================================
  # File Operations
  # ============================================================================

  defp do_flush(%{buffer: []} = state), do: state

  defp do_flush(state) do
    path = today_path()

    # Check if we need to create/rotate file
    state = ensure_file(state, path)

    # Write records
    csv_lines =
      state.buffer
      |> Enum.map(&record_to_csv/1)
      |> Enum.join("")

    case File.write(state.current_path, csv_lines, [:append]) do
      :ok ->
        VivaLog.debug(:dataset_collector, :flushed,
          count: length(state.buffer),
          filename: Path.basename(state.current_path)
        )

        %{
          state
          | buffer: [],
            stats: %{
              state.stats
              | total_records: state.stats.total_records + length(state.buffer),
                last_flush: DateTime.utc_now()
            }
        }

      {:error, reason} ->
        VivaLog.error(:dataset_collector, :write_error, reason: inspect(reason))
        state
    end
  end

  defp ensure_file(state, path) do
    cond do
      # New day or first write
      state.current_path != path ->
        if state.file, do: File.close(state.file)
        create_new_file(state, path)

      # File too large, rotate
      state.file && file_size(path) > @max_file_size ->
        File.close(state.file)
        rotated_path = rotate_path(path)
        create_new_file(state, rotated_path)

      # Keep using current file
      true ->
        state
    end
  end

  defp create_new_file(state, path) do
    # Write header if new file
    unless File.exists?(path) do
      File.write!(path, @csv_header)
    end

    VivaLog.info(:dataset_collector, :writing, filename: Path.basename(path))

    %{
      state
      | current_path: path,
        # We use File.write with :append, no persistent handle needed
        file: nil,
        stats: %{state.stats | files_written: state.stats.files_written + 1}
    }
  end

  defp record_to_csv(record) do
    [
      record.timestamp,
      record.metric,
      Float.round(record.value * 1.0, 6),
      Float.round(record.predicted * 1.0, 6),
      Float.round(record.free_energy * 1.0, 6),
      Float.round(record.precision * 1.0, 4),
      record.feeling,
      Float.round(record.time_dilation * 1.0, 4)
    ]
    |> Enum.join(",")
    |> Kernel.<>("\n")
  end

  # ============================================================================
  # Path Helpers
  # ============================================================================

  defp dataset_dir do
    Application.app_dir(:viva_core, @dataset_dir)
  end

  defp ensure_dataset_dir do
    dir = dataset_dir()

    unless File.exists?(dir) do
      File.mkdir_p!(dir)
      VivaLog.info(:dataset_collector, :dir_created, path: dir)
    end
  end

  defp today_path do
    date = Date.utc_today() |> Date.to_iso8601()
    Path.join(dataset_dir(), "viva_sensations_#{date}.csv")
  end

  defp rotate_path(path) do
    # Add timestamp suffix for rotation
    base = Path.basename(path, ".csv")
    dir = Path.dirname(path)
    timestamp = System.system_time(:second)
    Path.join(dir, "#{base}_#{timestamp}.csv")
  end

  defp file_size(path) do
    case File.stat(path) do
      {:ok, %{size: size}} -> size
      _ -> 0
    end
  end
end
