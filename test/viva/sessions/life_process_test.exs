defmodule Viva.Sessions.LifeProcessTest do
  use ExUnit.Case, async: true

  alias Viva.Sessions.LifeProcess

  describe "module structure" do
    test "exports expected functions" do
      functions = LifeProcess.__info__(:functions)

      assert {:start_link, 1} in functions
      assert {:get_state, 1} in functions
      assert {:owner_connected, 1} in functions
      assert {:owner_disconnected, 1} in functions
      assert {:trigger_thought, 1} in functions
      assert {:start_interaction, 2} in functions
      assert {:end_interaction, 1} in functions
      assert {:set_thought, 2} in functions
    end

    test "implements LifeProcessBehaviour" do
      behaviours = LifeProcess.__info__(:attributes)[:behaviour] || []
      assert Viva.Sessions.LifeProcessBehaviour in behaviours
    end
  end

  describe "struct" do
    test "has expected fields" do
      lp = %LifeProcess{}

      assert Map.has_key?(lp, :avatar_id)
      assert Map.has_key?(lp, :avatar)
      assert Map.has_key?(lp, :state)
      assert Map.has_key?(lp, :last_tick_at)
      assert Map.has_key?(lp, :owner_online?)
      assert Map.has_key?(lp, :current_conversation)
      assert Map.has_key?(lp, :last_thought)
      assert Map.has_key?(lp, :tick_count)
    end

    test "tick_count defaults to 0" do
      lp = %LifeProcess{}
      assert lp.tick_count == 0
    end
  end
end
