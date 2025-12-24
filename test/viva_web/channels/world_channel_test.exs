defmodule VivaWeb.WorldChannelTest do
  use ExUnit.Case, async: true

  alias VivaWeb.WorldChannel

  describe "module structure" do
    test "exports expected callback functions" do
      functions = WorldChannel.__info__(:functions)

      assert {:join, 3} in functions
      assert {:handle_in, 3} in functions
    end

    test "implements Phoenix.Channel behaviour" do
      behaviours = WorldChannel.__info__(:attributes)[:behaviour] || []
      assert Phoenix.Channel in behaviours
    end
  end
end
