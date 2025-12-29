defmodule Viva.Avatars.ConsciousnessStateTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SelfModel

  describe "new/0" do
    test "returns default state" do
      state = ConsciousnessState.new()
      assert state.stream_tempo == :normal
      assert state.presence_level == 0.7
      assert %SelfModel{} = state.self_model
    end
  end

  describe "from_personality/1" do
    test "derives meta_awareness from openness" do
      p = %Personality{openness: 0.8, enneagram_type: :type_5}
      state = ConsciousnessState.from_personality(p)

      # 0.8 * 0.4 + 0.2 = 0.32 + 0.2 = 0.52
      assert_in_delta state.meta_awareness, 0.52, 0.001
    end
  end

  describe "changeset/2" do
    test "validates ranges" do
      params = %{presence_level: 1.5}
      changeset = ConsciousnessState.changeset(%ConsciousnessState{}, params)
      refute changeset.valid?
      assert %{presence_level: ["must be less than or equal to 1.0"]} = errors_on(changeset)
    end

    test "accepts valid params" do
      params = %{
        presence_level: 0.5,
        stream_tempo: :fast,
        meta_awareness: 0.9,
        flow_state: 0.8
      }

      changeset = ConsciousnessState.changeset(%ConsciousnessState{}, params)
      assert changeset.valid?
    end
  end

  describe "query helpers" do
    test "dissociated?/1" do
      assert ConsciousnessState.dissociated?(%ConsciousnessState{presence_level: 0.3})
      refute ConsciousnessState.dissociated?(%ConsciousnessState{presence_level: 0.5})
    end

    test "in_flow?/1" do
      assert ConsciousnessState.in_flow?(%ConsciousnessState{flow_state: 0.8})
      refute ConsciousnessState.in_flow?(%ConsciousnessState{flow_state: 0.6})
    end

    test "self_aware?/1" do
      assert ConsciousnessState.self_aware?(%ConsciousnessState{meta_awareness: 0.7})
      refute ConsciousnessState.self_aware?(%ConsciousnessState{meta_awareness: 0.5})
    end

    test "racing_thoughts?/1" do
      assert ConsciousnessState.racing_thoughts?(%ConsciousnessState{stream_tempo: :racing})
      refute ConsciousnessState.racing_thoughts?(%ConsciousnessState{stream_tempo: :fast})
    end

    test "frozen?/1" do
      assert ConsciousnessState.frozen?(%ConsciousnessState{stream_tempo: :frozen})
      refute ConsciousnessState.frozen?(%ConsciousnessState{stream_tempo: :slow})
    end

    test "ruminating?/1" do
      assert ConsciousnessState.ruminating?(%ConsciousnessState{temporal_focus: :past})
      refute ConsciousnessState.ruminating?(%ConsciousnessState{temporal_focus: :present})
    end

    test "anticipating?/1" do
      assert ConsciousnessState.anticipating?(%ConsciousnessState{temporal_focus: :future})
      refute ConsciousnessState.anticipating?(%ConsciousnessState{temporal_focus: :present})
    end
  end

  describe "content accessors" do
    test "current_focus/1" do
      content = %{type: :thought, content: "foo"}
      state = %ConsciousnessState{focal_content: content}
      assert ConsciousnessState.current_focus(state) == content
    end

    test "latest_experience/1" do
      latest = %{content: "now"}
      older = %{content: "before"}
      state = %ConsciousnessState{experience_stream: [latest, older]}

      assert ConsciousnessState.latest_experience(state) == latest
      assert ConsciousnessState.latest_experience(%ConsciousnessState{experience_stream: []}) == nil
    end
  end

  describe "describe_state/1" do
    test "returns narrative description" do
      state_pres = %ConsciousnessState{presence_level: 0.9, stream_tempo: :normal}
      assert ConsciousnessState.describe_state(state_pres) =~ "fully present"
      assert ConsciousnessState.describe_state(state_pres) =~ "flowing naturally"

      state_diss = %{state_pres | presence_level: 0.1, stream_tempo: :racing}
      assert ConsciousnessState.describe_state(state_diss) =~ "dissociated"
      assert ConsciousnessState.describe_state(state_diss) =~ "racing uncontrollably"
    end
  end
end
