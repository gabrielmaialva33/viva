defmodule Viva.Matchmaker.EngineTest do
  use ExUnit.Case, async: true

  alias Viva.Matchmaker.Engine

  describe "module structure" do
    test "exports expected functions" do
      functions = Engine.__info__(:functions)

      assert {:start_link, 0} in functions or {:start_link, 1} in functions
      assert {:find_matches, 1} in functions or {:find_matches, 2} in functions
      assert {:calculate_compatibility, 2} in functions
      assert {:refresh_matches, 1} in functions
      assert {:invalidate, 1} in functions
      assert {:clear_cache, 0} in functions
      assert {:stats, 0} in functions
    end

    test "implements GenServer behaviour" do
      behaviours = Engine.__info__(:attributes)[:behaviour] || []
      assert GenServer in behaviours
    end
  end

  describe "invalidate/1" do
    test "returns expected structure" do
      # Test with a random UUID (will likely miss cache but should not error)
      result = Engine.invalidate(Ecto.UUID.generate())

      # Should return {:ok, boolean} (true if key existed, false otherwise)
      assert match?({:ok, _}, result)
    end
  end
end
