defmodule Viva.Avatars.Systems.ConsciousnessTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SensoryState
  alias Viva.Avatars.Systems.Consciousness

  setup do
    consciousness_data = ConsciousnessState.new()
    sensory_data = SensoryState.new()
    bio_data = %BioState{}
    emotional_data = %EmotionalState{}
    personality_data = %Personality{}

    {:ok,
     consciousness: consciousness_data,
     sensory: sensory_data,
     bio: bio_data,
     emotional: emotional_data,
     personality: personality_data}
  end

  describe "integrate/6" do
    test "updates consciousness state with new moment", context do
      %{consciousness: c, sensory: s, bio: b, emotional: e, personality: p} = context

      thought = "I wonder what that is."
      updated = Consciousness.integrate(c, s, b, e, thought, p)

      assert length(updated.experience_stream) == 1
      assert updated.focal_content.type == :thought
      assert updated.stream_tempo == :normal
    end
  end

  describe "calculate_tempo/2" do
    test "returns frozen for high adenosine" do
      assert Consciousness.calculate_tempo(0.5, 0.95) == :frozen
    end

    test "returns slow for low arousal" do
      assert Consciousness.calculate_tempo(0.0, 0.5) == :slow
    end

    test "returns racing for high arousal" do
      assert Consciousness.calculate_tempo(0.9, 0.0) == :racing
    end

    test "returns fast for moderate high arousal" do
      assert Consciousness.calculate_tempo(0.6, 0.0) == :fast
    end

    test "returns normal for balanced state" do
      assert Consciousness.calculate_tempo(0.5, 0.4) == :normal
    end
  end

  describe "update_workspace/4" do
    test "prioritizes high salience perception" do
      current = %{}
      qualia = %{narrative: "Loud bang!", intensity: 0.9}
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.5, mood_label: "neutral"}
      thought = "What?"

      {focal, _} = Consciousness.update_workspace(current, qualia, emotional, thought)

      assert focal.type == :perception
      assert focal.content == qualia
    end

    test "prioritizes intense emotion" do
      current = %{}
      qualia = %{narrative: "Quiet room", intensity: 0.2}
      emotional = %EmotionalState{pleasure: -0.9, arousal: 0.8, mood_label: "devastated"}
      thought = "Sad."

      {focal, _} = Consciousness.update_workspace(current, qualia, emotional, thought)

      assert focal.type == :emotion
      assert focal.content == "devastated"
    end

    test "prioritizes thought when other inputs low" do
      current = %{}
      qualia = %{narrative: "Wall", intensity: 0.1}
      emotional = %EmotionalState{pleasure: 0.1, arousal: 0.1, mood_label: "calm"}
      thought = "Deep philosophy."

      {focal, _} = Consciousness.update_workspace(current, qualia, emotional, thought)

      assert focal.type == :thought
      assert focal.content == "Deep philosophy."
    end
  end

  describe "calculate_presence/3" do
    test "calculates presence based on factors" do
      # High stress reduces presence
      emotional_state = %EmotionalState{arousal: 0.5}
      bio_state = %BioState{cortisol: 0.9, adenosine: 0.0}
      surprise_val = 0.0

      pres_val = Consciousness.calculate_presence(emotional_state, bio_state, surprise_val)
      assert pres_val < 0.7

      # High surprise reduces presence
      bio_low = %BioState{cortisol: 0.2, adenosine: 0.0}
      surprise_high = 0.95
      pres_surp = Consciousness.calculate_presence(emotional_state, bio_low, surprise_high)
      assert pres_surp < 0.7

      # High arousal increases presence
      emotional_high = %EmotionalState{arousal: 0.9}
      surprise_none = 0.0
      pres_high = Consciousness.calculate_presence(emotional_high, bio_low, surprise_none)
      assert pres_high > 0.7
    end
  end

  describe "calculate_flow/3" do
    test "high flow state conditions" do
      sensory = %SensoryState{attention_intensity: 0.9}
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.4}
      consciousness = %ConsciousnessState{temporal_focus: :present}

      flow = Consciousness.calculate_flow(sensory, emotional, consciousness)
      assert flow > 0.8
    end

    test "low flow state conditions" do
      sensory = %SensoryState{attention_intensity: 0.2}
      emotional = %EmotionalState{pleasure: -0.5, arousal: 0.9}
      consciousness = %ConsciousnessState{temporal_focus: :future}

      flow = Consciousness.calculate_flow(sensory, emotional, consciousness)
      assert flow < 0.4
    end
  end

  describe "maybe_metacognate/3" do
    test "triggers metacognition on strong emotion" do
      c_state = %ConsciousnessState{
        meta_awareness: 0.4,
        focal_content: %{type: :emotion, content: "joy"}
      }

      e_state = %EmotionalState{pleasure: 0.9, arousal: 0.8}
      p_traits = %Personality{openness: 0.5}

      {new_meta, obs} = Consciousness.maybe_metacognate(c_state, e_state, p_traits)

      assert new_meta > 0.3
      assert obs =~ "I notice I'm feeling"
    end

    test "decays metacognition when calm" do
      c = %ConsciousnessState{meta_awareness: 0.8}
      e = %EmotionalState{pleasure: 0.1, arousal: 0.1}
      p = %Personality{openness: 0.5, neuroticism: 0.5}

      {new_meta, _} = Consciousness.maybe_metacognate(c, e, p)

      assert new_meta < 0.8
    end
  end

  describe "determine_temporal_focus/2" do
    test "detects past keywords" do
      emotional = %EmotionalState{}
      assert Consciousness.determine_temporal_focus("I remember yesterday", emotional) == :past
      assert Consciousness.determine_temporal_focus("Back when I was young", emotional) == :past
    end

    test "detects future keywords" do
      emotional = %EmotionalState{}
      assert Consciousness.determine_temporal_focus("I will go tomorrow", emotional) == :future
      assert Consciousness.determine_temporal_focus("I worry about the plan", emotional) == :future
    end

    test "detects past focus from negative mood" do
      emotional = %EmotionalState{pleasure: -0.8}
      assert Consciousness.determine_temporal_focus("thinking", emotional) == :past
    end

    test "detects future focus from anxiety" do
      emotional = %EmotionalState{pleasure: -0.2, arousal: 0.8}
      assert Consciousness.determine_temporal_focus("thinking", emotional) == :future
    end

    test "defaults to present" do
      emotional = %EmotionalState{pleasure: 0.5, arousal: 0.5}
      assert Consciousness.determine_temporal_focus("Looking at this tree", emotional) == :present
    end
  end

  describe "synthesize_experience_narrative/3" do
    test "generates rich description" do
      c = %ConsciousnessState{
        presence_level: 0.9,
        stream_tempo: :normal,
        focal_content: %{content: "a beautiful sunset"},
        meta_observation: "I feel peaceful",
        temporal_focus: :present
      }

      s = %SensoryState{current_qualia: %{narrative: "Golden light"}}
      e = %EmotionalState{mood_label: "serene", pleasure: 0.8, arousal: 0.2}

      narrative = Consciousness.synthesize_experience_narrative(c, s, e)

      assert narrative =~ "fully present"
      assert narrative =~ "flowing naturally"
      assert narrative =~ "beautiful sunset"
      assert narrative =~ "Golden light"
      assert narrative =~ "serene"
      assert narrative =~ "I feel peaceful"
    end
  end

  describe "tick/2" do
    test "decays meta awareness and drifts to present" do
      c = %ConsciousnessState{meta_awareness: 0.9, temporal_focus: :past}
      p = %Personality{openness: 0.5}

      # Run multiple ticks to ensure drift happens (it's probabilistic)
      updated = Enum.reduce(1..20, c, fn _, acc -> Consciousness.tick(acc, p) end)

      assert updated.meta_awareness < 0.9
    end

    test "ages experience stream" do
      old_moment = %{timestamp: DateTime.add(DateTime.utc_now(), -700, :second)}
      new_moment = %{timestamp: DateTime.utc_now()}

      c = %ConsciousnessState{experience_stream: [new_moment, old_moment]}
      p = %Personality{}

      updated = Consciousness.tick(c, p)

      assert length(updated.experience_stream) == 1
      assert hd(updated.experience_stream) == new_moment
    end
  end
end
