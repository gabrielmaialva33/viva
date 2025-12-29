defmodule Viva.Avatars.Systems.DreamsTest do
  use Viva.DataCase, async: true
  import Mox

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.Memory
  alias Viva.Avatars.SelfModel
  alias Viva.Avatars.Systems.Dreams

  # Mock setup
  setup :verify_on_exit!

  describe "should_dream?/1" do
    test "always dreams on high intensity days" do
      # Intensity based on abs(p)*0.4 + a*0.3 + s*0.3. Max 1.0.
      high_exp = [%{emotion: %{pleasure: -0.9, arousal: 0.9}, surprise: 0.9}]
      assert Dreams.should_dream?(high_exp) == true
    end

    test "rarely dreams on low intensity days" do
      low_exp = [%{emotion: %{pleasure: 0.0, arousal: 0.0}, surprise: 0.0}]
      # 10% chance. We sample.
      results = for _ <- 1..100, do: Dreams.should_dream?(low_exp)
      # Probabilistic
      assert Enum.count(results, & &1) < 25
    end

    test "returns false for no experiences" do
      refute Dreams.should_dream?([])
    end
  end

  describe "process_dream_cycle/3" do
    setup do
      consciousness = %ConsciousnessState{self_model: %SelfModel{self_esteem: 0.5}}

      experiences = [
        %{emotion: %{pleasure: 0.9, arousal: 0.5}, qualia: %{narrative: "Good day"}},
        %{emotion: %{pleasure: -0.1, arousal: 0.8}, qualia: %{narrative: "Minor event"}}
      ]

      {:ok, c: consciousness, e: experiences}
    end

    test "processes cycle and creates dream memory", %{c: c, e: e} do
      # Mock LLM Client
      expect(Viva.AI.LLM.MockClient, :generate, fn _, _ ->
        {:ok, "I was flying over a vast ocean."}
      end)

      {:ok, updated_c, memory} = Dreams.process_dream_cycle("avatar-1", c, e)

      assert %ConsciousnessState{} = updated_c
      assert %Memory{} = memory
      assert memory.type == :dream
      assert memory.content =~ "flying"
      assert updated_c.self_model.self_esteem != c.self_model.self_esteem
    end

    test "handles LLM error with fallback dream", %{c: c, e: e} do
      expect(Viva.AI.LLM.MockClient, :generate, fn _, _ ->
        {:error, "API Down"}
      end)

      {:ok, _, memory} = Dreams.process_dream_cycle("avatar-1", c, e)
      # Should use fallback
      assert memory.content != nil
    end
  end

  describe "light_sleep_processing/1" do
    test "clears stream and resets focus" do
      c = %ConsciousnessState{
        experience_stream: [%{foo: :bar}],
        temporal_focus: :past,
        meta_observation: "something"
      }

      updated = Dreams.light_sleep_processing(c)

      assert updated.experience_stream == []
      assert updated.temporal_focus == :present
      assert updated.meta_observation == nil
    end
  end

  describe "helpers" do
    test "identify_significant_experiences/1 filters correctly" do
      exps = [
        # High
        %{emotion: %{pleasure: 0.9, arousal: 0.9}},
        # Low
        %{emotion: %{pleasure: 0.1, arousal: 0.1}}
      ]

      sig = Dreams.identify_significant_experiences(exps)
      assert length(sig) == 1
    end

    test "extract_emotional_residue/1 finds negative themes" do
      exps = [
        %{emotion: %{pleasure: -0.8, mood: "anxious"}},
        %{emotion: %{pleasure: -0.7, mood: "anxious"}}
      ]

      res = Dreams.extract_emotional_residue(exps)
      assert res.dominant_negative_theme == "anxious"
      assert res.intensity > 0.5
    end
  end
end
