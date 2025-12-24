defmodule Viva.Avatars.SomaticMarkersStateTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.SomaticMarkersState

  describe "new/0" do
    test "creates state with default values" do
      state = SomaticMarkersState.new()

      assert state.social_markers == %{}
      assert state.activity_markers == %{}
      assert state.context_markers == %{}
      assert state.current_bias == 0.0
      assert state.body_signal == nil
      assert state.learning_threshold == 0.7
      assert state.markers_formed == 0
      assert state.last_marker_activation == nil
    end
  end

  describe "changeset/2" do
    test "validates current_bias range" do
      state = SomaticMarkersState.new()

      changeset = SomaticMarkersState.changeset(state, %{current_bias: 1.5})
      refute changeset.valid?
      assert %{current_bias: ["must be less than or equal to 1.0"]} = errors_on(changeset)

      changeset = SomaticMarkersState.changeset(state, %{current_bias: -1.5})
      refute changeset.valid?
      assert %{current_bias: ["must be greater than or equal to -1.0"]} = errors_on(changeset)
    end

    test "validates learning_threshold range" do
      state = SomaticMarkersState.new()

      changeset = SomaticMarkersState.changeset(state, %{learning_threshold: 1.5})
      refute changeset.valid?
      assert %{learning_threshold: ["must be less than or equal to 1.0"]} = errors_on(changeset)

      changeset = SomaticMarkersState.changeset(state, %{learning_threshold: -0.1})
      refute changeset.valid?
      assert %{learning_threshold: ["must be greater than or equal to 0.0"]} = errors_on(changeset)
    end

    test "accepts valid changes" do
      state = SomaticMarkersState.new()

      changeset =
        SomaticMarkersState.changeset(state, %{
          current_bias: 0.5,
          learning_threshold: 0.8,
          markers_formed: 5
        })

      assert changeset.valid?
    end
  end

  describe "total_markers/1" do
    test "returns zero for empty state" do
      state = SomaticMarkersState.new()
      assert SomaticMarkersState.total_markers(state) == 0
    end

    test "counts markers across all categories" do
      state = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"a" => %{}, "b" => %{}},
          activity_markers: %{social: %{}},
          context_markers: %{"conversation" => %{}}
      }

      assert SomaticMarkersState.total_markers(state) == 4
    end
  end

  describe "has_body_memory?/1" do
    test "returns false when fewer than 3 markers formed" do
      state = %SomaticMarkersState{SomaticMarkersState.new() | markers_formed: 2}
      refute SomaticMarkersState.has_body_memory?(state)
    end

    test "returns true when 3 or more markers formed" do
      state = %SomaticMarkersState{SomaticMarkersState.new() | markers_formed: 3}
      assert SomaticMarkersState.has_body_memory?(state)

      state = %SomaticMarkersState{SomaticMarkersState.new() | markers_formed: 10}
      assert SomaticMarkersState.has_body_memory?(state)
    end
  end

  describe "strongest_marker/1" do
    test "returns nil when no markers exist" do
      state = SomaticMarkersState.new()
      assert SomaticMarkersState.strongest_marker(state) == nil
    end

    test "finds strongest marker by valence * strength" do
      weak_marker = %{valence: 0.5, strength: 0.3, last_activated: nil, context: nil}
      strong_marker = %{valence: 0.8, strength: 0.9, last_activated: nil, context: nil}

      state = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"weak" => weak_marker, "strong" => strong_marker}
      }

      {category, key, marker} = SomaticMarkersState.strongest_marker(state)
      assert category == :social
      assert key == "strong"
      assert marker == strong_marker
    end

    test "considers absolute valence for negative markers" do
      positive_marker = %{valence: 0.5, strength: 0.5, last_activated: nil, context: nil}
      negative_marker = %{valence: -0.9, strength: 0.9, last_activated: nil, context: nil}

      state = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"positive" => positive_marker},
          activity_markers: %{social: negative_marker}
      }

      {category, _key, marker} = SomaticMarkersState.strongest_marker(state)
      assert category == :activity
      assert marker == negative_marker
    end

    test "searches across all marker categories" do
      social_marker = %{valence: 0.3, strength: 0.3, last_activated: nil, context: nil}
      activity_marker = %{valence: 0.5, strength: 0.5, last_activated: nil, context: nil}
      context_marker = %{valence: 0.9, strength: 0.9, last_activated: nil, context: nil}

      state = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"friend" => social_marker},
          activity_markers: %{social: activity_marker},
          context_markers: %{"conversation" => context_marker}
      }

      {category, key, _marker} = SomaticMarkersState.strongest_marker(state)
      assert category == :context
      assert key == "conversation"
    end
  end

  defp errors_on(changeset) do
    Ecto.Changeset.traverse_errors(changeset, fn {message, opts} ->
      Regex.replace(~r"%{(\w+)}", message, fn _, key ->
        opts |> Keyword.get(String.to_existing_atom(key), key) |> to_string()
      end)
    end)
  end
end
