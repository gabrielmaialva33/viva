defmodule Viva.Avatars.SomaticMarkersStateTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.SomaticMarkersState

  describe "new/0" do
    test "returns default state" do
      s = SomaticMarkersState.new()
      assert s.current_bias == 0.0
      assert s.social_markers == %{}
    end
  end

  describe "changeset/2" do
    test "validates bias range" do
      params = %{current_bias: 1.5}
      changeset = SomaticMarkersState.changeset(%SomaticMarkersState{}, params)
      refute changeset.valid?
      assert %{current_bias: ["must be less than or equal to 1.0"]} = errors_on(changeset)
    end
  end

  describe "query functions" do
    test "total_markers/1 counts all maps" do
      s = %SomaticMarkersState{
        social_markers: %{"a" => %{}},
        activity_markers: %{b: %{}},
        context_markers: %{"c" => %{}}
      }

      assert SomaticMarkersState.total_markers(s) == 3
    end

    test "has_body_memory?/1" do
      assert SomaticMarkersState.has_body_memory?(%SomaticMarkersState{markers_formed: 3})
      refute SomaticMarkersState.has_body_memory?(%SomaticMarkersState{markers_formed: 2})
    end

    test "strongest_marker/1 returns the one with highest abs impact" do
      # 0.25
      m1 = %{valence: 0.5, strength: 0.5}
      # 0.64
      m2 = %{valence: -0.8, strength: 0.8}

      s = %SomaticMarkersState{
        social_markers: %{"friend" => m1},
        activity_markers: %{social: m2}
      }

      {type, key, marker} = SomaticMarkersState.strongest_marker(s)
      assert type == :activity
      assert key == "social"
      assert marker == m2
    end
  end
end
