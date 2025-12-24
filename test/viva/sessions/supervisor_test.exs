defmodule Viva.Sessions.SupervisorTest do
  use ExUnit.Case, async: true

  alias Viva.Sessions.Supervisor, as: SessionsSupervisor

  describe "module structure" do
    test "exports expected functions" do
      functions = SessionsSupervisor.__info__(:functions)

      assert {:start_link, 1} in functions
      assert {:start_avatar, 1} in functions
      assert {:stop_avatar, 1} in functions
      assert {:avatar_alive?, 1} in functions
      assert {:get_avatar_pid, 1} in functions
      assert {:list_running_avatars, 0} in functions
      assert {:count_running_avatars, 0} in functions
      assert {:start_all_active_avatars, 0} in functions
    end

    test "implements Supervisor behaviour" do
      behaviours = SessionsSupervisor.__info__(:attributes)[:behaviour] || []
      assert Supervisor in behaviours
    end
  end
end
