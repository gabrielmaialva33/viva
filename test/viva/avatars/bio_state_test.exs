defmodule Viva.Avatars.BioStateTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.BioState

  describe "changeset/2" do
    test "validates numeric ranges for dopamine and cortisol" do
      params = %{dopamine: 1.5, cortisol: -0.1}
      changeset = BioState.changeset(%BioState{}, params)

      refute changeset.valid?
      assert %{dopamine: ["must be less than or equal to 1.0"]} = errors_on(changeset)
      assert %{cortisol: ["must be greater than or equal to 0.0"]} = errors_on(changeset)
    end

    test "accepts valid attributes" do
      params = %{
        dopamine: 0.5,
        cortisol: 0.5,
        oxytocin: 0.5,
        adenosine: 0.5,
        libido: 0.5,
        chronotype: :owl,
        sleep_start_hour: 2,
        wake_start_hour: 10
      }

      changeset = BioState.changeset(%BioState{}, params)
      assert changeset.valid?
    end
  end
end
