defmodule Viva.AI.LLM.RateLimiterTest do
  use ExUnit.Case, async: false

  alias Viva.AI.LLM.RateLimiter

  setup do
    # Reset the rate limiter before each test
    # The RateLimiter is already started by the application supervisor
    RateLimiter.reset()
    :ok
  end

  describe "check_rate/0" do
    test "allows requests when tokens available" do
      assert :ok = RateLimiter.check_rate()
    end

    test "rate limits after consuming all burst tokens" do
      # Get current burst size from stats
      stats = RateLimiter.stats()
      burst_size = stats.burst_size

      # Consume all burst tokens
      Enum.each(1..burst_size, fn _ ->
        RateLimiter.check_rate()
      end)

      # Next request should be rate limited
      assert {:error, {:rate_limited, _wait_time}} = RateLimiter.check_rate()
    end
  end

  describe "acquire/1" do
    test "acquires token immediately when available" do
      assert :ok = RateLimiter.acquire(1_000)
    end

    test "returns timeout error when tokens exhausted and timeout exceeded" do
      # Get current burst size from stats
      stats = RateLimiter.stats()
      burst_size = stats.burst_size

      # Consume all tokens
      Enum.each(1..burst_size, fn _ ->
        RateLimiter.check_rate()
      end)

      # Try to acquire with very short timeout
      assert {:error, :rate_limit_timeout} = RateLimiter.acquire(10)
    end
  end

  describe "stats/0" do
    test "returns current rate limiter stats" do
      stats = RateLimiter.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :tokens_available)
      assert Map.has_key?(stats, :burst_size)
      assert Map.has_key?(stats, :requests_per_minute)
      assert Map.has_key?(stats, :refill_rate_per_second)
    end

    test "tokens decrease after check_rate" do
      initial_stats = RateLimiter.stats()
      RateLimiter.check_rate()
      new_stats = RateLimiter.stats()

      assert new_stats.tokens_available < initial_stats.tokens_available
    end
  end

  describe "reset/0" do
    test "resets tokens to full capacity" do
      stats = RateLimiter.stats()
      burst_size = stats.burst_size

      # Consume some tokens
      Enum.each(1..3, fn _ ->
        RateLimiter.check_rate()
      end)

      stats_before = RateLimiter.stats()
      assert stats_before.tokens_available < burst_size

      # Reset
      assert :ok = RateLimiter.reset()

      stats_after = RateLimiter.stats()
      assert stats_after.tokens_available == burst_size * 1.0
    end
  end

  describe "token refill" do
    test "tokens refill over time" do
      stats = RateLimiter.stats()
      burst_size = stats.burst_size

      # Consume all tokens
      Enum.each(1..burst_size, fn _ ->
        RateLimiter.check_rate()
      end)

      stats_empty = RateLimiter.stats()
      assert stats_empty.tokens_available < 1.0

      # Wait a bit for refill (60 RPM = 1 token per second)
      Process.sleep(1_100)

      stats_refilled = RateLimiter.stats()
      assert stats_refilled.tokens_available > stats_empty.tokens_available
    end
  end
end
