defmodule Viva.Avatars.ConsciousnessStateTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.ConsciousnessState

  describe "new/0" do
    test "creates a new consciousness state with defaults" do
      state = ConsciousnessState.new()

      assert state.experience_stream == []
      assert state.stream_tempo == :normal
      assert state.self_congruence == 0.8
      assert state.meta_awareness == 0.3
      assert state.meta_observation == nil
      assert state.peripheral_content == []
      assert state.temporal_focus == :present
      assert state.presence_level == 0.7
      assert state.experience_intensity == 0.5
      assert state.flow_state == 0.3
    end

    test "creates default focal content structure" do
      state = ConsciousnessState.new()

      assert state.focal_content.type == nil
      assert state.focal_content.content == nil
      assert state.focal_content.source == nil
    end

    test "embeds a new self-model" do
      state = ConsciousnessState.new()

      assert state.self_model != nil
      assert state.self_model.identity_narrative == "I am still discovering who I am."
    end
  end

  describe "changeset/2" do
    test "validates self_congruence range" do
      state = ConsciousnessState.new()

      invalid_changeset = ConsciousnessState.changeset(state, %{self_congruence: 1.5})
      refute invalid_changeset.valid?

      valid_changeset = ConsciousnessState.changeset(state, %{self_congruence: 0.9})
      assert valid_changeset.valid?
    end

    test "validates meta_awareness range" do
      state = ConsciousnessState.new()

      invalid_changeset = ConsciousnessState.changeset(state, %{meta_awareness: -0.5})
      refute invalid_changeset.valid?

      valid_changeset = ConsciousnessState.changeset(state, %{meta_awareness: 0.5})
      assert valid_changeset.valid?
    end

    test "validates presence_level range" do
      state = ConsciousnessState.new()

      invalid_changeset = ConsciousnessState.changeset(state, %{presence_level: 2.0})
      refute invalid_changeset.valid?

      valid_changeset = ConsciousnessState.changeset(state, %{presence_level: 0.8})
      assert valid_changeset.valid?
    end

    test "validates experience_intensity range" do
      state = ConsciousnessState.new()

      invalid_changeset = ConsciousnessState.changeset(state, %{experience_intensity: 1.5})
      refute invalid_changeset.valid?

      valid_changeset = ConsciousnessState.changeset(state, %{experience_intensity: 0.7})
      assert valid_changeset.valid?
    end

    test "validates flow_state range" do
      state = ConsciousnessState.new()

      invalid_changeset = ConsciousnessState.changeset(state, %{flow_state: -0.1})
      refute invalid_changeset.valid?

      valid_changeset = ConsciousnessState.changeset(state, %{flow_state: 0.9})
      assert valid_changeset.valid?
    end

    test "validates stream_tempo enum" do
      state = ConsciousnessState.new()

      slow_changeset = ConsciousnessState.changeset(state, %{stream_tempo: :slow})
      assert slow_changeset.valid?

      racing_changeset = ConsciousnessState.changeset(state, %{stream_tempo: :racing})
      assert racing_changeset.valid?
    end

    test "validates temporal_focus enum" do
      state = ConsciousnessState.new()

      past_changeset = ConsciousnessState.changeset(state, %{temporal_focus: :past})
      assert past_changeset.valid?

      future_changeset = ConsciousnessState.changeset(state, %{temporal_focus: :future})
      assert future_changeset.valid?
    end
  end

  describe "query functions" do
    test "describe_state/1 returns state description" do
      state = %ConsciousnessState{
        presence_level: 0.8,
        stream_tempo: :normal,
        temporal_focus: :present
      }

      description = ConsciousnessState.describe_state(state)
      assert String.contains?(description, "present")
      assert String.contains?(description, "naturally")
    end

    test "describe_state/1 with racing tempo and low presence" do
      state = %ConsciousnessState{
        presence_level: 0.3,
        stream_tempo: :racing,
        temporal_focus: :future
      }

      description = ConsciousnessState.describe_state(state)
      assert String.contains?(description, "distant")
      assert String.contains?(description, "racing")
    end

    test "dissociated?/1 returns true when presence is low" do
      low_presence = %ConsciousnessState{presence_level: 0.2}
      assert ConsciousnessState.dissociated?(low_presence)

      normal_presence = %ConsciousnessState{presence_level: 0.5}
      refute ConsciousnessState.dissociated?(normal_presence)
    end

    test "in_flow?/1 returns true when flow state is high" do
      high_flow = %ConsciousnessState{flow_state: 0.8}
      assert ConsciousnessState.in_flow?(high_flow)

      normal_flow = %ConsciousnessState{flow_state: 0.5}
      refute ConsciousnessState.in_flow?(normal_flow)
    end

    test "current_focus/1 returns focal content map" do
      state = %ConsciousnessState{
        focal_content: %{type: :sensation, content: "warmth", source: :perception}
      }

      focus = ConsciousnessState.current_focus(state)
      assert focus.type == :sensation
      assert focus.content == "warmth"
    end

    test "current_focus/1 returns map with nil type for empty focus" do
      state = %ConsciousnessState{
        focal_content: %{type: nil, content: nil, source: nil}
      }

      focus = ConsciousnessState.current_focus(state)
      assert focus.type == nil
    end

    test "latest_experience/1 returns most recent experience" do
      exp1 = %{content: "old", timestamp: DateTime.utc_now()}
      exp2 = %{content: "newest", timestamp: DateTime.utc_now()}

      state = %ConsciousnessState{experience_stream: [exp2, exp1]}

      latest = ConsciousnessState.latest_experience(state)
      assert latest.content == "newest"
    end

    test "latest_experience/1 returns nil for empty stream" do
      state = %ConsciousnessState{experience_stream: []}
      assert ConsciousnessState.latest_experience(state) == nil
    end
  end

  describe "from_personality/1" do
    test "initializes consciousness from personality" do
      personality = %Viva.Avatars.Personality{
        openness: 0.8,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5,
        attachment_style: :secure,
        values: ["creativity"]
      }

      state = ConsciousnessState.from_personality(personality)

      assert state.self_model != nil
      assert "creative thinking" in state.self_model.perceived_strengths
    end

    test "meta_awareness is influenced by openness" do
      high_openness = %Viva.Avatars.Personality{
        openness: 0.9,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5
      }

      low_openness = %Viva.Avatars.Personality{
        openness: 0.2,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5
      }

      high_state = ConsciousnessState.from_personality(high_openness)
      low_state = ConsciousnessState.from_personality(low_openness)

      assert high_state.meta_awareness > low_state.meta_awareness
    end
  end
end
