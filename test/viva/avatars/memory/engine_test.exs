defmodule Viva.Avatars.Memory.EngineTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.Memory.Engine

  describe "module structure" do
    test "exports expected functions" do
      functions = Engine.__info__(:functions)

      assert {:retrieve_relevant, 3} in functions
      assert {:calculate_score, 3} in functions
    end
  end

  describe "calculate_score/3" do
    test "returns a positive score" do
      memory = %{
        importance: 0.7,
        inserted_at: DateTime.utc_now()
      }

      query_vector = [0.1, 0.2, 0.3]
      current_time = DateTime.utc_now()

      score = Engine.calculate_score(memory, query_vector, current_time)

      assert is_float(score)
      assert score > 0
    end

    test "higher importance increases score" do
      current_time = DateTime.utc_now()
      query_vector = [0.1, 0.2, 0.3]

      low_importance = %{importance: 0.2, inserted_at: current_time}
      high_importance = %{importance: 0.9, inserted_at: current_time}

      low_score = Engine.calculate_score(low_importance, query_vector, current_time)
      high_score = Engine.calculate_score(high_importance, query_vector, current_time)

      assert high_score > low_score
    end

    test "older memories have lower recency score" do
      query_vector = [0.1, 0.2, 0.3]
      current_time = DateTime.utc_now()

      recent = %{importance: 0.5, inserted_at: current_time}
      old = %{importance: 0.5, inserted_at: DateTime.add(current_time, -24 * 7, :hour)}

      recent_score = Engine.calculate_score(recent, query_vector, current_time)
      old_score = Engine.calculate_score(old, query_vector, current_time)

      assert recent_score > old_score
    end
  end
end
