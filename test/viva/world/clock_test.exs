defmodule Viva.World.ClockTest do
  use ExUnit.Case, async: true

  alias Viva.World.Clock

  describe "time_scale/0" do
    test "returns the time scale constant" do
      scale = Clock.time_scale()

      assert is_integer(scale)
      assert scale == 10
    end
  end

  describe "module structure" do
    test "exports expected functions" do
      functions = Clock.__info__(:functions)

      assert {:start_link, 1} in functions
      assert {:now, 0} in functions
      assert {:time_scale, 0} in functions
      assert {:pause, 0} in functions
      assert {:resume, 0} in functions
    end

    test "implements GenServer behaviour" do
      behaviours = Clock.__info__(:attributes)[:behaviour] || []
      assert GenServer in behaviours
    end
  end

  describe "struct" do
    test "has expected fields" do
      clock = %Clock{}

      assert Map.has_key?(clock, :world_time)
      assert Map.has_key?(clock, :real_start_time)
      assert Map.has_key?(clock, :is_running)
    end

    test "fields default to nil" do
      clock = %Clock{}

      assert is_nil(clock.world_time)
      assert is_nil(clock.real_start_time)
      assert is_nil(clock.is_running)
    end
  end
end
