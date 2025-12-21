defmodule Viva.Nim.RateLimiter do
  @moduledoc """
  Token bucket rate limiter for NIM API calls.

  Prevents overwhelming the NIM API with too many requests.
  Uses a token bucket algorithm with configurable:
  - :requests_per_minute - Max requests per minute (default: 60)
  - :burst_size - Max burst size (default: 10)

  Tokens are refilled continuously based on the rate.
  """
  use GenServer
  require Logger

  @type rate_check_result :: :ok | {:error, {:rate_limited, pos_integer()}}
  @type acquire_result :: :ok | {:error, :rate_limit_timeout}
  @type stats :: %{
          tokens_available: float(),
          burst_size: pos_integer(),
          requests_per_minute: pos_integer(),
          refill_rate_per_second: float()
        }

  @table :nim_rate_limiter
  @default_requests_per_minute 60
  @default_burst_size 10
  @refill_interval_ms 1_000

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if a request is allowed and consume a token.
  Returns :ok if allowed, {:error, :rate_limited} if not.
  """
  @spec check_rate() :: rate_check_result()
  def check_rate do
    case :ets.lookup(@table, :bucket) do
      [{:bucket, tokens, last_refill}] ->
        now = System.monotonic_time(:millisecond)
        config = get_config()

        elapsed_ms = now - last_refill
        refill_rate = config.requests_per_minute / 60_000
        new_tokens = min(config.burst_size, tokens + elapsed_ms * refill_rate)

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
  """
  @spec acquire(pos_integer()) :: acquire_result()
  def acquire(max_wait_ms \\ 5_000) do
    acquire_loop(max_wait_ms, System.monotonic_time(:millisecond))
  end

  @doc "Get current rate limiter stats"
  @spec stats() :: stats() | %{tokens_available: 0, error: :not_initialized}
  def stats do
    case :ets.lookup(@table, :bucket) do
      [{:bucket, tokens, last_refill}] ->
        config = get_config()
        now = System.monotonic_time(:millisecond)
        elapsed_ms = now - last_refill
        refill_rate = config.requests_per_minute / 60_000
        current_tokens = min(config.burst_size, tokens + elapsed_ms * refill_rate)

        %{
          tokens_available: Float.round(current_tokens, 2),
          burst_size: config.burst_size,
          requests_per_minute: config.requests_per_minute,
          refill_rate_per_second: Float.round(refill_rate * 1000, 2)
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

  @impl GenServer
  def init(opts) do
    table =
      :ets.new(@table, [:named_table, :public, read_concurrency: true, write_concurrency: true])

    requests_per_minute = Keyword.get(opts, :requests_per_minute, @default_requests_per_minute)
    burst_size = Keyword.get(opts, :burst_size, @default_burst_size)

    state = %{
      table: table,
      requests_per_minute: requests_per_minute,
      burst_size: burst_size
    }

    :ets.insert(@table, {:bucket, burst_size * 1.0, System.monotonic_time(:millisecond)})
    :ets.insert(@table, {:config, requests_per_minute, burst_size})
    schedule_maintenance()

    {:ok, state}
  end

  @impl GenServer
  def handle_call(:reset, _, state) do
    :ets.insert(@table, {:bucket, state.burst_size * 1.0, System.monotonic_time(:millisecond)})
    {:reply, :ok, state}
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

  defp schedule_maintenance do
    Process.send_after(self(), :maintenance, @refill_interval_ms * 60)
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
