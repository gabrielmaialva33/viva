defmodule Viva.Sessions.LifeProcessBehaviourTest do
  use ExUnit.Case, async: true

  alias Viva.Sessions.LifeProcessBehaviour

  describe "module structure" do
    test "defines set_thought callback" do
      callbacks = LifeProcessBehaviour.behaviour_info(:callbacks)

      assert {:set_thought, 2} in callbacks
    end
  end
end
