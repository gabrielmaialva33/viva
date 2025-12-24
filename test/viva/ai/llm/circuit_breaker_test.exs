defmodule Viva.AI.LLM.CircuitBreakerTest do
  use ExUnit.Case, async: false

  alias Viva.AI.LLM.CircuitBreaker

  setup do
    # Reset the circuit breaker before each test
    # The CircuitBreaker is already started by the application supervisor
    CircuitBreaker.reset()
    # Give time for the reset to process
    Process.sleep(10)
    :ok
  end

  describe "allow_request?/0" do
    test "allows requests when circuit is closed" do
      assert CircuitBreaker.allow_request?() == true
    end

    test "blocks requests when circuit is open" do
      # Get threshold from stats
      stats = CircuitBreaker.stats()
      threshold = stats.failure_threshold

      # Cause enough failures to open the circuit
      Enum.each(1..threshold, fn _ ->
        CircuitBreaker.record_failure()
      end)

      # Wait for GenServer to process the casts
      Process.sleep(50)

      assert CircuitBreaker.allow_request?() == false
    end
  end

  describe "get_state/0" do
    test "starts in closed state after reset" do
      assert CircuitBreaker.get_state() == :closed
    end

    test "transitions to open after threshold failures" do
      stats = CircuitBreaker.stats()
      threshold = stats.failure_threshold

      Enum.each(1..threshold, fn _ ->
        CircuitBreaker.record_failure()
      end)

      Process.sleep(50)
      assert CircuitBreaker.get_state() == :open
    end
  end

  describe "record_success/0" do
    test "keeps circuit closed" do
      CircuitBreaker.record_success()
      Process.sleep(20)
      assert CircuitBreaker.get_state() == :closed
    end
  end

  describe "record_failure/0" do
    test "increments failure count" do
      stats_before = CircuitBreaker.stats()
      CircuitBreaker.record_failure()
      Process.sleep(20)
      stats_after = CircuitBreaker.stats()

      assert stats_after.failure_count == stats_before.failure_count + 1
    end

    test "opens circuit after threshold" do
      assert CircuitBreaker.get_state() == :closed

      stats = CircuitBreaker.stats()
      threshold = stats.failure_threshold

      Enum.each(1..threshold, fn _ ->
        CircuitBreaker.record_failure()
      end)

      Process.sleep(50)
      assert CircuitBreaker.get_state() == :open
    end
  end

  describe "reset/0" do
    test "resets circuit to closed state" do
      stats = CircuitBreaker.stats()
      threshold = stats.failure_threshold

      # Open the circuit
      Enum.each(1..threshold, fn _ ->
        CircuitBreaker.record_failure()
      end)

      Process.sleep(50)
      assert CircuitBreaker.get_state() == :open

      # Reset
      assert :ok = CircuitBreaker.reset()

      assert CircuitBreaker.get_state() == :closed
    end

    test "resets failure count to zero" do
      CircuitBreaker.record_failure()
      CircuitBreaker.record_failure()
      Process.sleep(20)

      stats_before = CircuitBreaker.stats()
      assert stats_before.failure_count == 2

      CircuitBreaker.reset()

      stats_after = CircuitBreaker.stats()
      assert stats_after.failure_count == 0
    end
  end

  describe "stats/0" do
    test "returns circuit breaker stats" do
      stats = CircuitBreaker.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :state)
      assert Map.has_key?(stats, :failure_count)
      assert Map.has_key?(stats, :success_count)
      assert Map.has_key?(stats, :failure_threshold)
      assert Map.has_key?(stats, :reset_timeout_ms)
    end
  end

  describe "call/1" do
    test "executes function when circuit is closed" do
      result = CircuitBreaker.call(fn -> {:ok, :success} end)
      assert result == {:ok, :success}
    end

    test "returns circuit_open error when circuit is open" do
      stats = CircuitBreaker.stats()
      threshold = stats.failure_threshold

      # Open the circuit
      Enum.each(1..threshold, fn _ ->
        CircuitBreaker.record_failure()
      end)

      Process.sleep(50)

      result = CircuitBreaker.call(fn -> {:ok, :success} end)
      assert result == {:error, :circuit_open}
    end

    test "records success on successful function call" do
      CircuitBreaker.call(fn -> {:ok, :success} end)
      Process.sleep(20)

      stats_after = CircuitBreaker.stats()
      # Success resets failure count
      assert stats_after.failure_count == 0
    end

    test "records failure on retryable error" do
      stats_before = CircuitBreaker.stats()

      CircuitBreaker.call(fn -> {:error, {:http_error, 500, %{}}} end)
      Process.sleep(20)

      stats_after = CircuitBreaker.stats()
      assert stats_after.failure_count == stats_before.failure_count + 1
    end

    test "does not record failure on non-retryable error" do
      stats_before = CircuitBreaker.stats()

      CircuitBreaker.call(fn -> {:error, {:http_error, 400, %{}}} end)
      Process.sleep(20)

      stats_after = CircuitBreaker.stats()
      assert stats_after.failure_count == stats_before.failure_count
    end

    test "records failure and returns error on exception" do
      result = CircuitBreaker.call(fn -> raise "test error" end)
      Process.sleep(20)

      assert {:error, {:exception, %RuntimeError{}}} = result

      stats = CircuitBreaker.stats()
      assert stats.failure_count == 1
    end
  end
end
