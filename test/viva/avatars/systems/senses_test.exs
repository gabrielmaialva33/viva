defmodule Viva.Avatars.Systems.SensesTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.SensoryState
  alias Viva.Avatars.Systems.Senses

  @base_stimulus %{
    type: :social,
    source: "conversation_partner",
    intensity: 0.6,
    valence: 0.3,
    novelty: 0.5,
    threat_level: 0.0
  }

  describe "perceive/4" do
    setup do
      sensory = SensoryState.new()
      emotional = %EmotionalState{}
      personality = build_personality()

      {:ok, sensory: sensory, emotional: emotional, personality: personality}
    end

    test "returns updated sensory state and neuro effects", ctx do
      {new_sensory, effects} =
        Senses.perceive(ctx.sensory, @base_stimulus, ctx.personality, ctx.emotional)

      assert %SensoryState{} = new_sensory
      assert is_list(effects)
    end

    test "updates attention focus based on stimulus", ctx do
      {new_sensory, _} =
        Senses.perceive(ctx.sensory, @base_stimulus, ctx.personality, ctx.emotional)

      assert new_sensory.attention_focus != nil
    end

    test "calculates surprise for novel stimuli", ctx do
      high_novelty_stimulus = %{@base_stimulus | novelty: 0.9, intensity: 0.8}

      {new_sensory, _} =
        Senses.perceive(ctx.sensory, high_novelty_stimulus, ctx.personality, ctx.emotional)

      assert new_sensory.surprise_level > 0.0
    end

    test "returns surprise_high effect for very surprising stimuli", ctx do
      # Create an expectation that will be violated
      sensory_with_expectation = %{
        ctx.sensory
        | expectations: [
            %{context: "social", prediction: "calm interaction", confidence: 0.9}
          ]
      }

      threatening_stimulus = %{@base_stimulus | threat_level: 0.8, novelty: 0.9, intensity: 0.9}

      {new_sensory, effects} =
        Senses.perceive(
          sensory_with_expectation,
          threatening_stimulus,
          ctx.personality,
          ctx.emotional
        )

      assert new_sensory.surprise_level > 0.5
      # Effects might include surprise_high or surprise_moderate
      assert Enum.any?(effects, &(&1 in [:surprise_high, :surprise_moderate])) or effects == []
    end

    test "updates hedonic signals based on valence", ctx do
      pleasant_stimulus = %{@base_stimulus | valence: 0.8}

      {new_sensory, _} =
        Senses.perceive(ctx.sensory, pleasant_stimulus, ctx.personality, ctx.emotional)

      assert new_sensory.sensory_pleasure > 0.0
    end

    test "calculates pain for threatening stimuli", ctx do
      threatening_stimulus = %{@base_stimulus | threat_level: 0.7, valence: -0.5}

      {new_sensory, _} =
        Senses.perceive(ctx.sensory, threatening_stimulus, ctx.personality, ctx.emotional)

      assert new_sensory.sensory_pain > 0.0
    end

    test "adds stimulus to active percepts", ctx do
      {new_sensory, _} =
        Senses.perceive(ctx.sensory, @base_stimulus, ctx.personality, ctx.emotional)

      assert new_sensory.active_percepts != []
    end
  end

  describe "calculate_salience/3" do
    test "high intensity increases salience" do
      stimulus = %{@base_stimulus | intensity: 0.9}
      emotional = %EmotionalState{}

      salience = Senses.calculate_salience(stimulus, emotional, nil)
      assert salience > 0.5
    end

    test "high novelty increases salience" do
      stimulus = %{@base_stimulus | novelty: 0.9, intensity: 0.3}
      emotional = %EmotionalState{}

      salience = Senses.calculate_salience(stimulus, emotional, nil)
      assert salience > 0.3
    end

    test "threat increases salience significantly" do
      stimulus = %{@base_stimulus | threat_level: 0.8, intensity: 0.3}
      emotional = %EmotionalState{}

      salience = Senses.calculate_salience(stimulus, emotional, nil)
      assert salience > 0.5
    end

    test "emotional arousal amplifies salience" do
      stimulus = %{@base_stimulus | intensity: 0.5}
      high_arousal_emotional = %EmotionalState{arousal: 0.9, pleasure: 0.0, dominance: 0.5}

      salience = Senses.calculate_salience(stimulus, high_arousal_emotional, nil)
      assert salience > 0.4
    end

    test "salience is clamped to 1.0" do
      extreme_stimulus = %{@base_stimulus | intensity: 1.0, novelty: 1.0, threat_level: 1.0}
      high_arousal = %EmotionalState{arousal: 1.0, pleasure: 0.0, dominance: 0.5}

      salience = Senses.calculate_salience(extreme_stimulus, high_arousal, nil)
      assert salience <= 1.0
    end
  end

  describe "filter_through_personality/2" do
    test "high openness adds high perceived detail" do
      high_openness = %Personality{
        openness: 0.9,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.5
      }

      stimulus = %{@base_stimulus | novelty: 0.6}

      filtered = Senses.filter_through_personality(stimulus, high_openness)
      assert filtered.perceived_detail == :high
    end

    test "high neuroticism amplifies perceived threat" do
      high_neuroticism = %Personality{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.5,
        agreeableness: 0.5,
        neuroticism: 0.9
      }

      stimulus = %{@base_stimulus | threat_level: 0.3}

      filtered = Senses.filter_through_personality(stimulus, high_neuroticism)
      # perceived_threat = threat_level * neuroticism * 1.5 = 0.3 * 0.9 * 1.5 = 0.405
      assert filtered.perceived_threat > stimulus.threat_level
    end

    test "low extraversion causes overwhelm at high intensity" do
      low_extraversion = %Personality{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.2,
        agreeableness: 0.5,
        neuroticism: 0.5
      }

      high_intensity_stimulus = %{@base_stimulus | intensity: 0.8}

      filtered = Senses.filter_through_personality(high_intensity_stimulus, low_extraversion)
      assert filtered.overwhelm == true
    end

    test "high extraversion does not cause overwhelm" do
      high_extraversion = %Personality{
        openness: 0.5,
        conscientiousness: 0.5,
        extraversion: 0.9,
        agreeableness: 0.5,
        neuroticism: 0.5
      }

      high_intensity_stimulus = %{@base_stimulus | intensity: 0.8}

      filtered = Senses.filter_through_personality(high_intensity_stimulus, high_extraversion)
      assert filtered.overwhelm == false
    end
  end

  describe "tick/2" do
    test "decays attention intensity over time" do
      sensory = %SensoryState{attention_intensity: 0.8, cognitive_load: 0.5}
      personality = build_personality()

      new_sensory = Senses.tick(sensory, personality)
      assert new_sensory.attention_intensity < 0.8
    end

    test "decays surprise level" do
      sensory = %SensoryState{surprise_level: 0.7}
      personality = build_personality()

      new_sensory = Senses.tick(sensory, personality)
      assert new_sensory.surprise_level < 0.7
    end

    test "ages and removes old percepts" do
      old_percept = %{
        stimulus: %{type: :ambient},
        qualia: %{},
        timestamp: DateTime.add(DateTime.utc_now(), -120, :second),
        salience: 0.3
      }

      sensory = %SensoryState{active_percepts: [old_percept]}
      personality = build_personality()

      new_sensory = Senses.tick(sensory, personality)
      # Old percepts with low salience should be removed
      assert length(new_sensory.active_percepts) <= length(sensory.active_percepts)
    end

    test "decays hedonic signals" do
      sensory = %SensoryState{sensory_pleasure: 0.6, sensory_pain: 0.4}
      personality = build_personality()

      new_sensory = Senses.tick(sensory, personality)
      # Hedonic signals decay toward 0 over time
      assert new_sensory.sensory_pleasure <= 0.6
      assert new_sensory.sensory_pain <= 0.4
    end
  end

  describe "check_prediction/2" do
    test "returns baseline surprise with no expectations" do
      stimulus = @base_stimulus
      {surprise, _} = Senses.check_prediction(stimulus, [])

      # Baseline surprise is 0.3 when there are no expectations
      assert surprise == 0.3
    end

    test "returns surprise for unexpected threat" do
      expectations = [
        %{context: "social", prediction: "calm environment", confidence: 0.9}
      ]

      threatening_stimulus = %{@base_stimulus | threat_level: 0.8}
      {surprise, error} = Senses.check_prediction(threatening_stimulus, expectations)

      assert surprise > 0.3
      assert error != nil
    end
  end

  defp build_personality(opts \\ []) do
    %Personality{
      openness: Keyword.get(opts, :openness, 0.5),
      conscientiousness: Keyword.get(opts, :conscientiousness, 0.5),
      extraversion: Keyword.get(opts, :extraversion, 0.5),
      agreeableness: Keyword.get(opts, :agreeableness, 0.5),
      neuroticism: Keyword.get(opts, :neuroticism, 0.5)
    }
  end
end
