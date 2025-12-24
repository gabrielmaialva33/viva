defmodule Viva.Avatars.Systems.DreamsTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.Memory
  alias Viva.Avatars.SelfModel
  alias Viva.Avatars.Systems.Dreams

  describe "should_dream?/1" do
    test "returns false for empty experiences" do
      refute Dreams.should_dream?([])
    end

    test "returns true for high emotional intensity day" do
      experiences = [
        %{emotion: %{pleasure: 0.9, arousal: 0.8}, surprise: 0.7},
        %{emotion: %{pleasure: 0.8, arousal: 0.7}, surprise: 0.6},
        %{emotion: %{pleasure: -0.8, arousal: 0.9}, surprise: 0.8}
      ]

      # High intensity should always dream
      assert Dreams.should_dream?(experiences)
    end

    test "probabilistically returns based on moderate intensity" do
      experiences = [
        %{emotion: %{pleasure: 0.4, arousal: 0.5}, surprise: 0.3},
        %{emotion: %{pleasure: 0.3, arousal: 0.4}, surprise: 0.2}
      ]

      # Run multiple times to test probability
      results = for _ <- 1..20, do: Dreams.should_dream?(experiences)

      # Should have some true and some false
      true_count = Enum.count(results, & &1)
      assert true_count > 0 and true_count < 20
    end

    test "rarely dreams for low intensity day" do
      experiences = [
        %{emotion: %{pleasure: 0.1, arousal: 0.1}, surprise: 0.1},
        %{emotion: %{pleasure: 0.0, arousal: 0.1}, surprise: 0.0}
      ]

      # Run multiple times - most should be false
      results = for _ <- 1..20, do: Dreams.should_dream?(experiences)
      false_count = Enum.count(results, &(!&1))

      assert false_count >= 15
    end
  end

  describe "calculate_emotional_intensity/1" do
    test "returns 0.0 for empty list" do
      assert Dreams.calculate_emotional_intensity([]) == 0.0
    end

    test "calculates average intensity from experiences" do
      experiences = [
        %{emotion: %{pleasure: 0.8, arousal: 0.6}, surprise: 0.4},
        %{emotion: %{pleasure: 0.6, arousal: 0.4}, surprise: 0.2}
      ]

      intensity = Dreams.calculate_emotional_intensity(experiences)
      assert intensity > 0.0
      assert intensity <= 1.0
    end

    test "weights pleasure, arousal, and surprise" do
      # High pleasure
      high_pleasure = [%{emotion: %{pleasure: 1.0, arousal: 0.0}, surprise: 0.0}]
      pleasure_intensity = Dreams.calculate_emotional_intensity(high_pleasure)

      # High arousal
      high_arousal = [%{emotion: %{pleasure: 0.0, arousal: 1.0}, surprise: 0.0}]
      arousal_intensity = Dreams.calculate_emotional_intensity(high_arousal)

      # Both should contribute to intensity
      assert pleasure_intensity > 0.0
      assert arousal_intensity > 0.0
    end

    test "absolute pleasure value is used" do
      negative_experience = [%{emotion: %{pleasure: -0.8, arousal: 0.5}, surprise: 0.3}]
      intensity = Dreams.calculate_emotional_intensity(negative_experience)

      # Negative pleasure should still contribute to intensity
      assert intensity > 0.3
    end
  end

  describe "identify_significant_experiences/1" do
    test "returns empty list for no significant experiences" do
      experiences = [
        %{emotion: %{pleasure: 0.1, arousal: 0.1}, surprise: 0.1},
        %{emotion: %{pleasure: 0.1, arousal: 0.1}, surprise: 0.0}
      ]

      result = Dreams.identify_significant_experiences(experiences)
      assert result == []
    end

    test "identifies high intensity experiences" do
      experiences = [
        %{emotion: %{pleasure: 0.9, arousal: 0.8}, surprise: 0.7, id: 1},
        %{emotion: %{pleasure: 0.1, arousal: 0.1}, surprise: 0.0, id: 2},
        %{emotion: %{pleasure: 0.8, arousal: 0.7}, surprise: 0.6, id: 3}
      ]

      result = Dreams.identify_significant_experiences(experiences)

      # Should include high intensity ones
      ids = Enum.map(result, & &1.id)
      assert 1 in ids
      assert 3 in ids
      refute 2 in ids
    end

    test "limits to top 5 significant experiences" do
      experiences =
        for i <- 1..10 do
          %{emotion: %{pleasure: 0.9, arousal: 0.8}, surprise: 0.7, id: i}
        end

      result = Dreams.identify_significant_experiences(experiences)
      assert Enum.count(result) == 5
    end

    test "sorts by significance descending" do
      experiences = [
        %{emotion: %{pleasure: 0.6, arousal: 0.5}, surprise: 0.4, id: 1},
        %{emotion: %{pleasure: 0.9, arousal: 0.9}, surprise: 0.9, id: 2},
        %{emotion: %{pleasure: 0.7, arousal: 0.6}, surprise: 0.5, id: 3}
      ]

      result = Dreams.identify_significant_experiences(experiences)
      ids = Enum.map(result, & &1.id)

      # Most significant should be first
      assert hd(ids) == 2
    end
  end

  describe "consolidate_memories/1" do
    test "returns message for empty experiences" do
      result = Dreams.consolidate_memories([])
      assert result == "No significant experiences to consolidate."
    end

    test "generates consolidation notes for experiences" do
      experiences = [
        %{
          emotion: %{mood: "happy"},
          qualia: %{narrative: "Had a wonderful conversation"},
          significance: 0.8
        },
        %{
          emotion: %{mood: "excited"},
          qualia: %{narrative: "Discovered something new"},
          significance: 0.7
        }
      ]

      result = Dreams.consolidate_memories(experiences)

      assert String.contains?(result, "Consolidated memories:")
      assert String.contains?(result, "happy") or String.contains?(result, "excited")
    end

    test "truncates long narratives" do
      experiences = [
        %{
          emotion: %{mood: "thoughtful"},
          qualia: %{
            narrative:
              "This is a very long narrative that goes on and on describing many details of what happened"
          },
          significance: 0.8
        }
      ]

      result = Dreams.consolidate_memories(experiences)

      # Should be truncated
      assert String.contains?(result, "...")
    end
  end

  describe "extract_emotional_residue/1" do
    test "extracts negative experiences" do
      experiences = [
        %{emotion: %{pleasure: -0.5, mood: "sad"}, id: 1},
        %{emotion: %{pleasure: 0.5, mood: "happy"}, id: 2},
        %{emotion: %{pleasure: -0.7, mood: "anxious"}, id: 3}
      ]

      result = Dreams.extract_emotional_residue(experiences)

      assert Enum.count(result.negative_experiences) == 2
      assert result.intensity > 0.0
    end

    test "returns zero intensity for no negative experiences" do
      experiences = [
        %{emotion: %{pleasure: 0.5, mood: "happy"}},
        %{emotion: %{pleasure: 0.3, mood: "content"}}
      ]

      result = Dreams.extract_emotional_residue(experiences)

      assert result.negative_experiences == []
      assert result.intensity == 0.0
    end

    test "identifies dominant negative theme" do
      experiences = [
        %{emotion: %{pleasure: -0.5, mood: "anxious"}},
        %{emotion: %{pleasure: -0.6, mood: "anxious"}},
        %{emotion: %{pleasure: -0.4, mood: "sad"}}
      ]

      result = Dreams.extract_emotional_residue(experiences)

      assert result.dominant_negative_theme == "anxious"
    end
  end

  describe "update_self_model/2" do
    test "returns new model for nil input" do
      result = Dreams.update_self_model(nil, [])
      assert %SelfModel{} = result
    end

    test "adjusts self-esteem based on positive experiences" do
      model = SelfModel.new()

      positive_experiences = [
        %{emotion: %{pleasure: 0.8}},
        %{emotion: %{pleasure: 0.7}},
        %{emotion: %{pleasure: 0.6}}
      ]

      result = Dreams.update_self_model(model, positive_experiences)
      assert result.self_esteem > model.self_esteem
    end

    test "adjusts self-esteem based on negative experiences" do
      model = SelfModel.new()

      negative_experiences = [
        %{emotion: %{pleasure: -0.8}},
        %{emotion: %{pleasure: -0.7}},
        %{emotion: %{pleasure: -0.6}}
      ]

      result = Dreams.update_self_model(model, negative_experiences)
      assert result.self_esteem < model.self_esteem
    end

    test "clamps self-esteem adjustments" do
      model = %{SelfModel.new() | self_esteem: 0.15}

      very_negative_experiences =
        for _ <- 1..20, do: %{emotion: %{pleasure: -1.0}}

      result = Dreams.update_self_model(model, very_negative_experiences)

      # Should not go below 0.1
      assert result.self_esteem >= 0.1
    end

    test "detects behavioral patterns from repeated emotions" do
      model = SelfModel.new()

      experiences = [
        %{emotion: %{mood: "anxious"}},
        %{emotion: %{mood: "anxious"}},
        %{emotion: %{mood: "anxious"}},
        %{emotion: %{mood: "happy"}}
      ]

      result = Dreams.update_self_model(model, experiences)

      # Should detect pattern for anxious (appeared 3+ times)
      assert result.behavioral_patterns != []
    end
  end

  describe "create_dream_memory/3" do
    test "returns nil for nil dream content" do
      result = Dreams.create_dream_memory("avatar-123", nil, "notes")
      assert result == nil
    end

    test "creates memory struct for dream content" do
      avatar_id = Ecto.UUID.generate()
      dream_content = "I wandered through a maze of mirrors."
      notes = "Consolidated: important conversation"

      result = Dreams.create_dream_memory(avatar_id, dream_content, notes)

      assert %Memory{} = result
      assert result.avatar_id == avatar_id
      assert result.content == dream_content
      assert result.type == :dream
      assert result.importance == 0.4
      assert result.context.consolidation == notes
    end
  end

  describe "light_sleep_processing/1" do
    test "clears experience stream" do
      consciousness = %{
        ConsciousnessState.new()
        | experience_stream: [%{}, %{}, %{}]
      }

      result = Dreams.light_sleep_processing(consciousness)
      assert result.experience_stream == []
    end

    test "resets temporal focus to present" do
      consciousness = %{ConsciousnessState.new() | temporal_focus: :past}

      result = Dreams.light_sleep_processing(consciousness)
      assert result.temporal_focus == :present
    end

    test "clears meta observation" do
      consciousness = %{
        ConsciousnessState.new()
        | meta_observation: "I notice I'm tired"
      }

      result = Dreams.light_sleep_processing(consciousness)
      assert result.meta_observation == nil
    end

    test "resets focal content" do
      consciousness = %{
        ConsciousnessState.new()
        | focal_content: %{type: :thought, content: "something", source: :internal}
      }

      result = Dreams.light_sleep_processing(consciousness)
      assert result.focal_content.type == nil
    end
  end
end
