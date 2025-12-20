defmodule Viva.Nim.CircuitBreaker do
  @moduledoc """
  Circuit breaker for NIM API calls.

  States:
  - :closed - Normal operation, requests allowed
  - :open - Too many failures, requests rejected
  - :half_open - Testing if service recovered

  Configuration via Application env:
  - :failure_threshold - Failures before opening (default: 5)
  - :reset_timeout_ms - Time before trying again (default: 30_000)
  - :success_threshold - Successes to close from half_open (default: 2)
  """
  use GenServer
  require Logger

  @table :nim_circuit_breaker
  @default_failure_threshold 5
  @default_reset_timeout_ms 30_000
  @default_success_threshold 2

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Check if requests are allowed"
  def allow_request? do
    case get_state() do
      :closed -> true
      :half_open -> true
      :open -> false
    end
  end

  @doc "Get current circuit state"
  def get_state do
    case :ets.lookup(@table, :state) do
      [{:state, state, opened_at}] ->
        maybe_transition_to_half_open(state, opened_at)

      [] ->
        :closed
    end
  end

  @doc "Record a successful request"
  def record_success do
    GenServer.cast(__MODULE__, :success)
  end

  @doc "Record a failed request"
  def record_failure do
    GenServer.cast(__MODULE__, :failure)
  end

  @doc "Execute a function with circuit breaker protection"
  def call(fun) when is_function(fun, 0) do
    if allow_request?() do
      try do
        case fun.() do
          {:ok, _} = result ->
            record_success()
            result

          {:error, reason} = error ->
            if retryable_error?(reason) do
              record_failure()
            end

            error
        end
      rescue
        e ->
          record_failure()
          {:error, {:exception, e}}
      end
    else
      {:error, :circuit_open}
    end
  end

  @doc "Reset circuit breaker to closed state"
  def reset do
    GenServer.call(__MODULE__, :reset)
  end

  @doc "Get circuit breaker stats"
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    table = :ets.new(@table, [:named_table, :public, read_concurrency: true])

    state = %{
      table: table,
      failure_count: 0,
      success_count: 0,
      failure_threshold: Keyword.get(opts, :failure_threshold, @default_failure_threshold),
      reset_timeout_ms: Keyword.get(opts, :reset_timeout_ms, @default_reset_timeout_ms),
      success_threshold: Keyword.get(opts, :success_threshold, @default_success_threshold)
    }

    :ets.insert(@table, {:state, :closed, nil})

    {:ok, state}
  end

  @impl true
  def handle_cast(:success, state) do
    current_state = get_state()

    new_state =
      case current_state do
        :half_open ->
          new_success_count = state.success_count + 1

          if new_success_count >= state.success_threshold do
            Logger.info("NIM circuit breaker: CLOSED (service recovered)")
            :ets.insert(@table, {:state, :closed, nil})
            %{state | failure_count: 0, success_count: 0}
          else
            %{state | success_count: new_success_count}
          end

        :closed ->
          %{state | failure_count: 0}

        :open ->
          state
      end

    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:failure, state) do
    current_state = get_state()

    new_state =
      case current_state do
        :closed ->
          new_failure_count = state.failure_count + 1

          if new_failure_count >= state.failure_threshold do
            Logger.warning(
              "NIM circuit breaker: OPEN (#{new_failure_count} failures, " <>
                "will retry in #{state.reset_timeout_ms}ms)"
            )

            :ets.insert(@table, {:state, :open, System.monotonic_time(:millisecond)})
            %{state | failure_count: new_failure_count, success_count: 0}
          else
            %{state | failure_count: new_failure_count}
          end

        :half_open ->
          Logger.warning("NIM circuit breaker: OPEN (test request failed)")
          :ets.insert(@table, {:state, :open, System.monotonic_time(:millisecond)})
          %{state | failure_count: state.failure_threshold, success_count: 0}

        :open ->
          state
      end

    {:noreply, new_state}
  end

  @impl true
  def handle_call(:reset, _, state) do
    :ets.insert(@table, {:state, :closed, nil})
    new_state = %{state | failure_count: 0, success_count: 0}
    Logger.info("NIM circuit breaker: manually reset to CLOSED")
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:stats, _, state) do
    stats = %{
      state: get_state(),
      failure_count: state.failure_count,
      success_count: state.success_count,
      failure_threshold: state.failure_threshold,
      reset_timeout_ms: state.reset_timeout_ms
    }

    {:reply, stats, state}
  end

  # Private Functions

  defp maybe_transition_to_half_open(:open, opened_at) do
    config = Application.get_env(:viva, :nim, [])
    reset_timeout = Keyword.get(config, :circuit_reset_timeout_ms, @default_reset_timeout_ms)
    elapsed = System.monotonic_time(:millisecond) - opened_at

    if elapsed >= reset_timeout do
      Logger.info("NIM circuit breaker: HALF_OPEN (testing recovery)")
      :ets.insert(@table, {:state, :half_open, nil})
      :half_open
    else
      :open
    end
  end

  defp maybe_transition_to_half_open(state, _), do: state

  defp retryable_error?({:http_error, status, _}) when status in [429, 500, 502, 503, 504], do: true
  defp retryable_error?(%Req.TransportError{}), do: true
  defp retryable_error?(:timeout), do: true
  defp retryable_error?({:timeout, _}), do: true
  defp retryable_error?(:econnrefused), do: true
  defp retryable_error?(:closed), do: true
  defp retryable_error?(_), do: false
end
