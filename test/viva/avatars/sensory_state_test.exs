defmodule Viva.Avatars.SensoryStateTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.SensoryState

  describe "new/0" do
    test "returns default state" do
      s = SensoryState.new()
      assert s.attention_intensity == 0.5
      assert s.current_qualia.intensity == 0.0
    end
  end

  describe "changeset/2" do
    test "validates ranges" do
      params = %{attention_intensity: 1.5}
      changeset = SensoryState.changeset(%SensoryState{}, params)
      refute changeset.valid?
      assert %{attention_intensity: ["must be less than or equal to 1.0"]} = errors_on(changeset)
    end

    test "validates sensory pleasure range (-1.0 to 1.0)" do
      params = %{sensory_pleasure: -1.5}
      changeset = SensoryState.changeset(%SensoryState{}, params)
      refute changeset.valid?
      assert %{sensory_pleasure: ["must be greater than or equal to -1.0"]} = errors_on(changeset)
    end

    test "accepts valid params" do
      params = %{
        attention_focus: "Bird",
        attention_intensity: 0.8,
        active_percepts: [%{stimulus: "tweet"}],
        sensory_pleasure: 0.5
      }

      changeset = SensoryState.changeset(%SensoryState{}, params)
      assert changeset.valid?
    end
  end

  describe "helpers" do
    setup do
      state = %SensoryState{
        current_qualia: %{dominant_sensation: "Warmth", narrative: "Feeling warm sun."},
        surprise_level: 0.8,
        sensory_pleasure: 0.9,
        sensory_pain: 0.9,
        attention_intensity: 0.9,
        cognitive_load: 0.9
      }

      {:ok, state: state}
    end

    test "dominant_sensation/1", %{state: s} do
      assert SensoryState.dominant_sensation(s) == "Warmth"
    end

    test "experience_narrative/1", %{state: s} do
      assert SensoryState.experience_narrative(s) == "Feeling warm sun."
    end

    test "surprised?/1", %{state: s} do
      assert SensoryState.surprised?(s)
      refute SensoryState.surprised?(%{s | surprise_level: 0.1})
    end

    test "experiencing_pleasure?/1", %{state: s} do
      assert SensoryState.experiencing_pleasure?(s)
      refute SensoryState.experiencing_pleasure?(%{s | sensory_pleasure: 0.1})
    end

    test "experiencing_pain?/1", %{state: s} do
      assert SensoryState.experiencing_pain?(s)
      refute SensoryState.experiencing_pain?(%{s | sensory_pain: 0.1})
    end

    test "focused?/1", %{state: s} do
      assert SensoryState.focused?(s)
      refute SensoryState.focused?(%{s | attention_intensity: 0.1})
    end

    test "overwhelmed?/1", %{state: s} do
      assert SensoryState.overwhelmed?(s)
      refute SensoryState.overwhelmed?(%{s | cognitive_load: 0.1})
    end
  end
end
