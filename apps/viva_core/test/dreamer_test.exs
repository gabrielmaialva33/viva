defmodule VivaCore.DreamerTest do
  use ExUnit.Case, async: false

  alias VivaCore.Dreamer
  alias VivaCore.Memory
  alias VivaCore.Emotional

  @moduletag :dreamer

  setup do
    # Start dependencies
    {:ok, emotional} =
      Emotional.start_link(name: :"emotional_#{:erlang.unique_integer([:positive])}")

    {:ok, memory} =
      Memory.start_link(
        name: :"memory_#{:erlang.unique_integer([:positive])}",
        backend: :in_memory
      )

    {:ok, dreamer} =
      Dreamer.start_link(
        name: :"dreamer_#{:erlang.unique_integer([:positive])}",
        emotional: emotional,
        memory: memory
      )

    on_exit(fn ->
      if Process.alive?(dreamer), do: GenServer.stop(dreamer)
      if Process.alive?(memory), do: GenServer.stop(memory)
      if Process.alive?(emotional), do: GenServer.stop(emotional)
    end)

    %{dreamer: dreamer, memory: memory, emotional: emotional}
  end

  describe "start_link/1" do
    test "starts with default state", %{dreamer: dreamer} do
      status = Dreamer.status(dreamer)

      assert status.importance_accumulator == 0.0
      assert status.threshold == 15.0
      assert status.reflection_count == 0
      assert status.thoughts_count == 0
    end
  end

  describe "on_memory_stored/3" do
    test "accumulates importance", %{dreamer: dreamer} do
      before = Dreamer.status(dreamer)
      assert before.importance_accumulator == 0.0

      Dreamer.on_memory_stored("mem_test_1", 0.5, dreamer)
      :timer.sleep(50)

      after_status = Dreamer.status(dreamer)
      assert after_status.importance_accumulator == 0.5

      Dreamer.on_memory_stored("mem_test_2", 0.7, dreamer)
      :timer.sleep(50)

      final_status = Dreamer.status(dreamer)
      assert final_status.importance_accumulator == 1.2
    end

    test "calculates progress correctly", %{dreamer: dreamer} do
      Dreamer.on_memory_stored("mem_test", 7.5, dreamer)
      :timer.sleep(50)

      status = Dreamer.status(dreamer)
      assert status.progress_percent == 50.0
    end
  end

  describe "status/1" do
    test "returns comprehensive status", %{dreamer: dreamer} do
      status = Dreamer.status(dreamer)

      assert Map.has_key?(status, :importance_accumulator)
      assert Map.has_key?(status, :threshold)
      assert Map.has_key?(status, :progress_percent)
      assert Map.has_key?(status, :last_reflection)
      assert Map.has_key?(status, :seconds_since_reflection)
      assert Map.has_key?(status, :reflection_count)
      assert Map.has_key?(status, :thoughts_count)
      assert Map.has_key?(status, :total_insights_generated)
      assert Map.has_key?(status, :uptime_seconds)
    end
  end

  describe "reflect_now/1" do
    test "triggers reflection and resets accumulator", %{dreamer: dreamer} do
      # Accumulate some importance
      Dreamer.on_memory_stored("mem_1", 5.0, dreamer)
      :timer.sleep(50)

      before = Dreamer.status(dreamer)
      assert before.importance_accumulator == 5.0

      # Trigger manual reflection
      result = Dreamer.reflect_now(dreamer)

      assert Map.has_key?(result, :focal_points)
      assert Map.has_key?(result, :insights)
      assert result.trigger == :manual

      # Accumulator should be reset
      after_status = Dreamer.status(dreamer)
      assert after_status.importance_accumulator == 0.0
      assert after_status.reflection_count == 1
    end
  end

  describe "recent_thoughts/2" do
    test "returns empty list initially", %{dreamer: dreamer} do
      thoughts = Dreamer.recent_thoughts(10, dreamer)
      assert thoughts == []
    end
  end

  describe "retrieve_with_scoring/3" do
    test "returns memories with composite score", %{dreamer: dreamer, memory: memory} do
      # Store some memories
      Memory.store("Test memory about programming", %{importance: 0.7}, memory)
      Memory.store("Another memory about coding", %{importance: 0.5}, memory)
      :timer.sleep(100)

      # Retrieve with scoring
      results = Dreamer.retrieve_with_scoring("programming", [limit: 5], dreamer)

      assert is_list(results)

      if length(results) > 0 do
        first = hd(results)
        assert Map.has_key?(first, :composite_score)
      end
    end
  end

  describe "scoring formula" do
    test "recency component calculation" do
      # Test the mathematical properties
      # Recent memory should have higher recency score
      now = DateTime.utc_now() |> DateTime.to_unix()
      one_week_ago = now - 604_800

      recent_decay = :math.exp(-0 / 604_800) * (1 + :math.log(1 + 0) / 10)
      old_decay = :math.exp(-604_800 / 604_800) * (1 + :math.log(1 + 0) / 10)

      assert recent_decay > old_decay
      assert_in_delta recent_decay, 1.0, 0.01
      assert_in_delta old_decay, 0.368, 0.01
    end

    test "spaced repetition boost" do
      # Higher access count should boost recency
      base = :math.exp(0) * (1 + :math.log(1 + 0) / 10)
      boosted = :math.exp(0) * (1 + :math.log(1 + 10) / 10)

      assert boosted > base
      assert_in_delta base, 1.0, 0.01
      assert boosted > 1.0
    end

    test "emotional resonance calculation" do
      # Same emotion should have resonance = 1
      # Opposite emotion should have resonance close to 0

      pad_diagonal = :math.sqrt(12)

      # Same state
      same_distance = 0
      same_resonance = 1 - same_distance / pad_diagonal
      assert_in_delta same_resonance, 1.0, 0.01

      # Opposite corners of PAD cube: (1,1,1) vs (-1,-1,-1)
      # sqrt(12)
      opposite_distance = :math.sqrt(4 + 4 + 4)
      opposite_resonance = 1 - opposite_distance / pad_diagonal
      assert_in_delta opposite_resonance, 0.0, 0.01
    end
  end

  describe "reflection trigger threshold" do
    test "triggers reflection when threshold reached", %{dreamer: dreamer} do
      # Send enough importance to trigger threshold (15.0)
      for i <- 1..16 do
        Dreamer.on_memory_stored("mem_#{i}", 1.0, dreamer)
        :timer.sleep(10)
      end

      # Give time for reflection to complete
      :timer.sleep(200)

      status = Dreamer.status(dreamer)

      # Should have triggered at least one reflection
      assert status.reflection_count >= 1
      # Accumulator should be reset (or low)
      assert status.importance_accumulator < 15.0
    end
  end
end
