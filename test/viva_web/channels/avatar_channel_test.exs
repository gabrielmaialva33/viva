defmodule VivaWeb.AvatarChannelTest do
  use ExUnit.Case, async: true

  alias VivaWeb.AvatarChannel

  describe "module structure" do
    test "exports expected callback functions" do
      functions = AvatarChannel.__info__(:functions)

      assert {:join, 3} in functions
      assert {:handle_in, 3} in functions
      assert {:handle_info, 2} in functions
      assert {:terminate, 2} in functions
    end

    test "implements Phoenix.Channel behaviour" do
      behaviours = AvatarChannel.__info__(:attributes)[:behaviour] || []
      assert Phoenix.Channel in behaviours
    end
  end
end
