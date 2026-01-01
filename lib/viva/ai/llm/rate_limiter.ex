defmodule Viva.AI.LLM.RateLimiter do
  @moduledoc """
  Token bucket rate limiter for NIM API calls.

  Prevents overwhelming the NIM API with too many requests.
  Uses a token bucket algorithm with configurable:
  - :requests_per_minute - Max requests per minute (default: 40 for NVIDIA trial)
  - :burst_size - Max burst size (default: 5)

  Tokens are refilled continuously based on the rate.

  ## NVIDIA NIM Rate Limits

  The NVIDIA trial API has a limit of 40 requests per minute.
  See: https://forums.developer.nvidia.com/t/model-limits/331075

  ## Adaptive Throttling

  When 429 responses are received, the rate limiter automatically
  reduces throughput temporarily to respect server capacity.
  """
  use GenServer
  require Logger

  @type rate_check_result :: :ok | {:error, {:rate_limited, pos_integer()}}
  @type acquire_result :: :ok | {:error, :rate_limit_timeout}
  @type stats :: %{
          tokens_available: float(),
          burst_size: pos_integer(),
          requests_per_minute: pos_integer(),
          refill_rate_per_second: float(),
          throttle_multiplier: float(),
          recent_429s: non_neg_integer()
        }

  @table :nim_rate_limiter
  # NVIDIA trial API limit - conservative default
  @default_requests_per_minute 40
  @default_burst_size 5
  @default_wait_ms 10_000
  @refill_interval_ms 1_000

  # Adaptive throttling constants
  @throttle_decay_interval_ms 60_000
  @max_throttle_multiplier 4.0
  @throttle_increase_per_429 0.5

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if a request is allowed and consume a token.
  Returns :ok if allowed, {:error, :rate_limited} if not.

  Takes into account the current throttle multiplier from adaptive throttling.
  When throttled, the effective rate is reduced to allow the API to recover.
  """
  @spec check_rate() :: rate_check_result()
  def check_rate do
    case :ets.lookup(@table, :bucket) do
      [{:bucket, tokens, last_refill}] ->
        now = System.monotonic_time(:millisecond)
        config = get_config()
        multiplier = throttle_multiplier()

        elapsed_ms = now - last_refill
        # Reduce effective rate when throttled
        effective_rpm = config.requests_per_minute / multiplier
        refill_rate = effective_rpm / 60_000
        new_tokens = min(config.burst_size * 1.0, tokens + elapsed_ms * refill_rate)

        if new_tokens >= 1.0 do
          :ets.insert(@table, {:bucket, new_tokens - 1.0, now})
          :ok
        else
          wait_time = ceil((1.0 - new_tokens) / refill_rate)
          {:error, {:rate_limited, wait_time}}
        end

      [] ->
        :ok
    end
  end

  @doc """
  Acquire a token, waiting if necessary.
  Returns :ok when token acquired, or {:error, :timeout} after max_wait_ms.

  Default wait time is configurable via `:rate_limit_wait_ms` in config.
  """
  @spec acquire(pos_integer()) :: acquire_result()
  def acquire(max_wait_ms \\ nil) do
    wait_ms = max_wait_ms || get_wait_ms()
    acquire_loop(wait_ms, System.monotonic_time(:millisecond))
  end

  defp get_wait_ms do
    nim_config = Application.get_env(:viva, :nim, [])
    Keyword.get(nim_config, :rate_limit_wait_ms, @default_wait_ms)
  end

  @doc "Get current rate limiter stats including throttle information"
  @spec stats() :: stats() | %{tokens_available: 0, error: :not_initialized}
  def stats do
    case :ets.lookup(@table, :bucket) do
      [{:bucket, tokens, last_refill}] ->
        config = get_config()
        now = System.monotonic_time(:millisecond)
        elapsed_ms = now - last_refill

        {multiplier, recent_429s} =
          case :ets.lookup(@table, :throttle) do
            [{:throttle, m, count, _}] -> {m, count}
            [] -> {1.0, 0}
          end

        effective_rpm = config.requests_per_minute / multiplier
        refill_rate = effective_rpm / 60_000
        current_tokens = min(config.burst_size * 1.0, tokens + elapsed_ms * refill_rate)

        %{
          tokens_available: Float.round(current_tokens, 2),
          burst_size: config.burst_size,
          requests_per_minute: config.requests_per_minute,
          effective_rpm: Float.round(effective_rpm, 1),
          refill_rate_per_second: Float.round(refill_rate * 1000, 2),
          throttle_multiplier: Float.round(multiplier, 2),
          recent_429s: recent_429s
        }

      [] ->
        %{tokens_available: 0, error: :not_initialized}
    end
  end

  @doc "Reset rate limiter to full capacity"
  @spec reset() :: :ok
  def reset do
    GenServer.call(__MODULE__, :reset)
  end

  @doc """
  Record a 429 response from the API.
  Triggers adaptive throttling to slow down requests.
  """
  @spec record_429() :: :ok
  def record_429 do
    GenServer.cast(__MODULE__, :record_429)
  end

  @doc """
  Get the current throttle multiplier.
  Returns 1.0 when not throttled, higher values when backing off.
  """
  @spec throttle_multiplier() :: float()
  def throttle_multiplier do
    case :ets.lookup(@table, :throttle) do
      [{:throttle, multiplier, _count, _last_429}] -> multiplier
      [] -> 1.0
    end
  end

  @impl GenServer
  def init(opts) do
    table =
      :ets.new(@table, [:named_table, :public, read_concurrency: true, write_concurrency: true])

    # Read from application config, with opts as override
    nim_config = Application.get_env(:viva, :nim, [])

    requests_per_minute =
      Keyword.get(opts, :requests_per_minute) ||
        Keyword.get(nim_config, :requests_per_minute, @default_requests_per_minute)

    burst_size =
      Keyword.get(opts, :burst_size) ||
        Keyword.get(nim_config, :burst_size, @default_burst_size)

    state = %{
      table: table,
      requests_per_minute: requests_per_minute,
      burst_size: burst_size
    }

    :ets.insert(@table, {:bucket, burst_size * 1.0, System.monotonic_time(:millisecond)})
    :ets.insert(@table, {:config, requests_per_minute, burst_size})
    # Initialize throttle state: multiplier=1.0, count=0, last_429=0
    :ets.insert(@table, {:throttle, 1.0, 0, 0})

    schedule_maintenance()
    schedule_throttle_decay()

    Logger.info(
      "RateLimiter started: #{requests_per_minute} RPM, burst=#{burst_size} " <>
        "(NVIDIA trial limit: 40 RPM)"
    )

    {:ok, state}
  end

  @impl GenServer
  def handle_call(:reset, _, state) do
    :ets.insert(@table, {:bucket, state.burst_size * 1.0, System.monotonic_time(:millisecond)})
    :ets.insert(@table, {:throttle, 1.0, 0, 0})
    {:reply, :ok, state}
  end

  @impl GenServer
  def handle_cast(:record_429, state) do
    now = System.monotonic_time(:millisecond)

    {new_multiplier, new_count} =
      case :ets.lookup(@table, :throttle) do
        [{:throttle, multiplier, count, _last}] ->
          # Increase throttle on each 429
          increased = min(multiplier + @throttle_increase_per_429, @max_throttle_multiplier)
          {increased, count + 1}

        [] ->
          {1.0 + @throttle_increase_per_429, 1}
      end

    :ets.insert(@table, {:throttle, new_multiplier, new_count, now})

    Logger.warning(
      "RateLimiter: 429 received, throttle increased to #{Float.round(new_multiplier, 2)}x " <>
        "(#{new_count} recent 429s)"
    )

    {:noreply, state}
  end

  @impl GenServer
  def handle_info(:maintenance, state) do
    case :ets.lookup(@table, :bucket) do
      [{:bucket, tokens, _}] when tokens > state.burst_size ->
        :ets.insert(@table, {:bucket, state.burst_size * 1.0, System.monotonic_time(:millisecond)})

      _ ->
        :ok
    end

    schedule_maintenance()
    {:noreply, state}
  end

  @impl GenServer
  def handle_info(:throttle_decay, state) do
    # Gradually reduce throttle multiplier over time
    case :ets.lookup(@table, :throttle) do
      [{:throttle, multiplier, count, last_429}] when multiplier > 1.0 ->
        now = System.monotonic_time(:millisecond)
        time_since_last_429 = now - last_429

        # Decay faster if no recent 429s
        decay_rate = if time_since_last_429 > 30_000, do: 0.3, else: 0.1
        new_multiplier = max(1.0, multiplier - decay_rate)

        # Reset count if multiplier returns to 1.0
        new_count = if new_multiplier == 1.0, do: 0, else: count

        :ets.insert(@table, {:throttle, new_multiplier, new_count, last_429})

        if new_multiplier < multiplier do
          Logger.debug("RateLimiter: throttle decayed to #{Float.round(new_multiplier, 2)}x")
        end

      _ ->
        :ok
    end

    schedule_throttle_decay()
    {:noreply, state}
  end

  defp schedule_maintenance do
    Process.send_after(self(), :maintenance, @refill_interval_ms * 60)
  end

  defp schedule_throttle_decay do
    Process.send_after(self(), :throttle_decay, @throttle_decay_interval_ms)
  end

  defp get_config do
    case :ets.lookup(@table, :config) do
      [{:config, rpm, burst}] ->
        %{requests_per_minute: rpm, burst_size: burst}

      [] ->
        %{requests_per_minute: @default_requests_per_minute, burst_size: @default_burst_size}
    end
  end

  defp acquire_loop(max_wait_ms, start_time) do
    case check_rate() do
      :ok ->
        :ok

      {:error, {:rate_limited, wait_time}} ->
        elapsed = System.monotonic_time(:millisecond) - start_time

        if elapsed + wait_time > max_wait_ms do
          {:error, :rate_limit_timeout}
        else
          sleep_time = min(wait_time, 100)
          Process.sleep(sleep_time)
          acquire_loop(max_wait_ms, start_time)
        end
    end
  end
end
