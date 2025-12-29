defmodule Viva.Avatars.InternalStateTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.AllostasisState
  alias Viva.Avatars.BioState
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.EmotionRegulationState
  alias Viva.Avatars.InternalState
  alias Viva.Avatars.MotivationState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SensoryState
  alias Viva.Avatars.SomaticMarkersState

  setup do
    personality_data = %Personality{enneagram_type: :type_5}
    state_data = InternalState.new()

    {:ok, state: state_data, personality: personality_data}
  end

  describe "new/0" do
    test "initializes all subsystems" do
      state = InternalState.new()

      assert %BioState{} = state.bio
      assert %EmotionalState{} = state.emotional
      assert %SensoryState{} = state.sensory
      assert %ConsciousnessState{} = state.consciousness
      assert %AllostasisState{} = state.allostasis
      assert %EmotionRegulationState{} = state.regulation
      assert %SomaticMarkersState{} = state.somatic
      assert %MotivationState{} = state.motivation
    end
  end

  describe "from_personality/1" do
    test "initializes conscioussness and motivation from personality" do
      pers = %Personality{
        # Challenger
        enneagram_type: :type_8,
        openness: 0.9
      }

      state = InternalState.from_personality(pers)

      # Check Motivation (Type 8 -> Autonomy seeking)
      assert state.motivation.primary_drive == :autonomy_seeking

      # Check Consciousness (initialized via from_personality logic)
      assert state.consciousness != nil
    end
  end

  describe "changeset/2" do
    test "validates attributes" do
      params = %{
        current_thought: "Thinking...",
        current_desire: :wants_rest,
        current_activity: :thinking
      }

      changeset = InternalState.changeset(%InternalState{}, params)
      assert changeset.valid?
    end
  end

  describe "dominant_emotion/1" do
    test "delegates to emotional state" do
      state = %InternalState{emotional: %EmotionalState{mood_label: "joyful"}}
      assert InternalState.dominant_emotion(state) == "joyful"
    end
  end

  describe "wellbeing/1" do
    test "calculates score based on pleasure, cortisol and adenosine" do
      # Best case: High pleasure (1.0), Low cortisol (0.0), Low adenosine (0.0)
      # Score: (1.0 * 0.6) + (1.0 * 0.2) + (1.0 * 0.2) = 1.0
      state_best = %InternalState{
        emotional: %EmotionalState{pleasure: 1.0},
        bio: %BioState{cortisol: 0.0, adenosine: 0.0}
      }

      assert InternalState.wellbeing(state_best) == 1.0

      # Worst case: Low pleasure (-1.0), High cortisol (1.0), High adenosine (1.0)
      # Score: (0.0 * 0.6) + (0.0 * 0.2) + (0.0 * 0.2) = 0.0
      state_worst = %InternalState{
        emotional: %EmotionalState{pleasure: -1.0},
        bio: %BioState{cortisol: 1.0, adenosine: 1.0}
      }

      assert InternalState.wellbeing(state_worst) == 0.0

      # Mixed case
      # Pleasure 0.0 -> score 0.5 -> contrib 0.3
      # Cortisol 0.5 -> contrib 0.1
      # Adenosine 0.5 -> contrib 0.1
      # Total: 0.5
      state_mixed = %InternalState{
        emotional: %EmotionalState{pleasure: 0.0},
        bio: %BioState{cortisol: 0.5, adenosine: 0.5}
      }

      assert_in_delta InternalState.wellbeing(state_mixed), 0.5, 0.001
    end
  end

  describe "delegated helpers" do
    test "qualia_narrative/1 delegates to sensory" do
      # We need a SensoryState that returns a narrative
      sens_val = %SensoryState{
        current_qualia: %{narrative: "Seeing cat. Hearing meow."}
      }

      state_val = %InternalState{sensory: sens_val}

      # SensoryState.experience_narrative logic:
      # returns map key :narrative
      narrative_val = InternalState.qualia_narrative(state_val)
      assert narrative_val == "Seeing cat. Hearing meow."
    end

    test "surprised?/1 delegates to sensory" do
      sens_surp = %SensoryState{surprise_level: 0.8}
      state_surp = %InternalState{sensory: sens_surp}
      assert InternalState.surprised?(state_surp)

      sens_norm = %SensoryState{surprise_level: 0.2}
      state_norm = %InternalState{sensory: sens_norm}
      refute InternalState.surprised?(state_norm)
    end

    test "dissociated?/1 delegates to consciousness" do
      # dissociation occurs when presence_level < 0.4
      cons_low = %ConsciousnessState{presence_level: 0.2}
      state_low = %InternalState{consciousness: cons_low}
      assert InternalState.dissociated?(state_low)

      cons_high = %ConsciousnessState{presence_level: 0.8}
      state_high = %InternalState{consciousness: cons_high}
      refute InternalState.dissociated?(state_high)
    end

    test "meta_observation/1 returns consciousness field" do
      cons_obs = %ConsciousnessState{meta_observation: "I am testing."}
      state_obs = %InternalState{consciousness: cons_obs}
      assert InternalState.meta_observation(state_obs) == "I am testing."
    end
  end

  describe "ensure_integrity/2" do
    test "creates new state if nil" do
      p_traits = %Personality{enneagram_type: :type_5}
      state_res = InternalState.ensure_integrity(nil, p_traits)
      assert state_res != nil
      # Type 5
      assert state_res.motivation.primary_drive == :safety_seeking
    end

    test "fills missing subsystems" do
      p_traits = %Personality{enneagram_type: :type_5}

      # State with EVERYTHING missing except bio
      state_raw = %InternalState{
        bio: %BioState{cortisol: 0.5},
        emotional: nil,
        sensory: nil,
        consciousness: nil,
        allostasis: nil,
        regulation: nil,
        somatic: nil,
        motivation: nil
      }

      restored = InternalState.ensure_integrity(state_raw, p_traits)

      # Preserved
      assert restored.bio.cortisol == 0.5
      assert restored.emotional != nil
      assert restored.sensory != nil
      assert restored.consciousness != nil
      assert restored.allostasis != nil
      assert restored.regulation != nil
      assert restored.somatic != nil
      assert restored.motivation != nil
      assert restored.motivation.primary_drive == :safety_seeking
    end

    test "keeps existing subsystems" do
      p_traits = %Personality{enneagram_type: :type_5}
      existing_mot = Map.put(MotivationState.new(), :survival_urgency, 0.9)

      state_part = %InternalState{
        motivation: existing_mot,
        # Missing one
        bio: nil
      }

      restored = InternalState.ensure_integrity(state_part, p_traits)

      # Preserved
      assert restored.motivation.survival_urgency == 0.9
      # Created
      assert restored.bio != nil
    end
  end
end
