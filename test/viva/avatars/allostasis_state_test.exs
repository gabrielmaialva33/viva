defmodule Viva.Avatars.AllostasisStateTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.AllostasisState

  describe "new/0" do
    test "returns default state" do
      state = AllostasisState.new()
      assert state.load_level == 0.0
      assert state.receptor_sensitivity == 1.0
    end
  end

  describe "changeset/2" do
    test "validates ranges" do
      params = %{load_level: 1.5, receptor_sensitivity: -0.1}
      changeset = AllostasisState.changeset(%AllostasisState{}, params)

      refute changeset.valid?
      assert %{load_level: ["must be less than or equal to 1.0"]} = errors_on(changeset)

      assert %{receptor_sensitivity: ["must be greater than or equal to 0.0"]} =
               errors_on(changeset)
    end

    test "validates valid params" do
      params = %{
        load_level: 0.5,
        receptor_sensitivity: 0.5,
        recovery_capacity: 0.5,
        cognitive_impairment: 0.5,
        high_stress_hours: 10.0
      }

      changeset = AllostasisState.changeset(%AllostasisState{}, params)
      assert changeset.valid?
    end
  end
end
