defmodule Viva.Avatars.SensoryStateTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.SensoryState

  describe "new/0" do
    test "creates a new sensory state with defaults" do
      state = SensoryState.new()

      assert state.attention_focus == nil
      assert state.attention_intensity == 0.5
      assert state.cognitive_load == 0.3
      assert state.active_percepts == []
      assert state.expectations == []
      assert state.surprise_level == 0.0
      assert state.last_prediction_error == nil
      assert state.sensory_pleasure == 0.0
      assert state.sensory_pain == 0.0
    end

    test "creates default qualia structure" do
      state = SensoryState.new()

      assert state.current_qualia.dominant_sensation == nil
      assert state.current_qualia.emotional_color == nil
      assert state.current_qualia.intensity == 0.0
      assert state.current_qualia.narrative == nil
    end
  end

  describe "changeset/2" do
    test "validates attention_intensity range" do
      state = SensoryState.new()

      invalid_high = SensoryState.changeset(state, %{attention_intensity: 1.5})
      refute invalid_high.valid?
      assert "must be less than or equal to 1.0" in errors_on(invalid_high).attention_intensity

      invalid_low = SensoryState.changeset(state, %{attention_intensity: -0.5})
      refute invalid_low.valid?
      assert "must be greater than or equal to 0.0" in errors_on(invalid_low).attention_intensity

      valid_changeset = SensoryState.changeset(state, %{attention_intensity: 0.8})
      assert valid_changeset.valid?
    end

    test "validates cognitive_load range" do
      state = SensoryState.new()

      invalid_changeset = SensoryState.changeset(state, %{cognitive_load: 1.5})
      refute invalid_changeset.valid?

      valid_changeset = SensoryState.changeset(state, %{cognitive_load: 0.6})
      assert valid_changeset.valid?
    end

    test "validates surprise_level range" do
      state = SensoryState.new()

      invalid_changeset = SensoryState.changeset(state, %{surprise_level: 2.0})
      refute invalid_changeset.valid?

      valid_changeset = SensoryState.changeset(state, %{surprise_level: 0.9})
      assert valid_changeset.valid?
    end

    test "validates sensory_pleasure range" do
      state = SensoryState.new()

      invalid_changeset = SensoryState.changeset(state, %{sensory_pleasure: -1.5})
      refute invalid_changeset.valid?

      valid_changeset = SensoryState.changeset(state, %{sensory_pleasure: -0.8})
      assert valid_changeset.valid?
    end

    test "validates sensory_pain range" do
      state = SensoryState.new()

      invalid_changeset = SensoryState.changeset(state, %{sensory_pain: -0.5})
      refute invalid_changeset.valid?

      valid_changeset = SensoryState.changeset(state, %{sensory_pain: 0.5})
      assert valid_changeset.valid?
    end
  end

  describe "query functions" do
    test "dominant_sensation/1 returns the dominant sensation" do
      state = %SensoryState{
        current_qualia: %{
          dominant_sensation: "warmth",
          emotional_color: nil,
          intensity: 0.5,
          narrative: nil
        }
      }

      assert SensoryState.dominant_sensation(state) == "warmth"
    end

    test "experience_narrative/1 returns the narrative" do
      state = %SensoryState{
        current_qualia: %{
          dominant_sensation: nil,
          emotional_color: nil,
          intensity: 0.5,
          narrative: "A warm feeling"
        }
      }

      assert SensoryState.experience_narrative(state) == "A warm feeling"
    end

    test "surprised?/1 returns true when surprise level is high" do
      high_surprise = %SensoryState{surprise_level: 0.6}
      assert SensoryState.surprised?(high_surprise)

      low_surprise = %SensoryState{surprise_level: 0.4}
      refute SensoryState.surprised?(low_surprise)
    end

    test "experiencing_pleasure?/1 returns true when pleasure is high" do
      high_pleasure = %SensoryState{sensory_pleasure: 0.5}
      assert SensoryState.experiencing_pleasure?(high_pleasure)

      low_pleasure = %SensoryState{sensory_pleasure: 0.2}
      refute SensoryState.experiencing_pleasure?(low_pleasure)
    end

    test "experiencing_pain?/1 returns true when pain is high" do
      high_pain = %SensoryState{sensory_pain: 0.5}
      assert SensoryState.experiencing_pain?(high_pain)

      low_pain = %SensoryState{sensory_pain: 0.2}
      refute SensoryState.experiencing_pain?(low_pain)
    end

    test "focused?/1 returns true when attention is intense" do
      focused_state = %SensoryState{attention_intensity: 0.8}
      assert SensoryState.focused?(focused_state)

      unfocused_state = %SensoryState{attention_intensity: 0.5}
      refute SensoryState.focused?(unfocused_state)
    end

    test "overwhelmed?/1 returns true when cognitive load is high" do
      overwhelmed_state = %SensoryState{cognitive_load: 0.9}
      assert SensoryState.overwhelmed?(overwhelmed_state)

      normal_state = %SensoryState{cognitive_load: 0.5}
      refute SensoryState.overwhelmed?(normal_state)
    end
  end

  defp errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {msg, opts} ->
      Regex.replace(~r"%{(\w+)}", msg, fn _, key ->
        opts
        |> Keyword.get(String.to_existing_atom(key), key)
        |> to_string()
      end)
    end)
  end
end
