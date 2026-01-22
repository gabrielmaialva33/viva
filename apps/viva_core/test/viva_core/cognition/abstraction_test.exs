defmodule VivaCore.Cognition.AbstractionTest do
  use ExUnit.Case

  alias VivaCore.Cognition.Abstraction

  describe "pad_to_concept/1" do
    test "maps anxious state correctly" do
      pad = %{pleasure: -0.5, arousal: 0.5, dominance: -0.5}
      assert Abstraction.pad_to_concept(pad) == :anxious
    end

    test "maps exuberant state correctly" do
      pad = %{pleasure: 0.5, arousal: 0.5, dominance: 0.5}
      assert Abstraction.pad_to_concept(pad) == :exuberant
    end

    test "maps dependent state correctly (P+ A+ D-)" do
      pad = %{pleasure: 0.5, arousal: 0.5, dominance: -0.5}
      assert Abstraction.pad_to_concept(pad) == :dependent
    end

    test "maps relaxed state correctly (P+ A- D+)" do
      pad = %{pleasure: 0.5, arousal: -0.5, dominance: 0.5}
      assert Abstraction.pad_to_concept(pad) == :relaxed
    end

    test "maps hostile state correctly (P- A+ D+)" do
      pad = %{pleasure: -0.5, arousal: 0.5, dominance: 0.5}
      assert Abstraction.pad_to_concept(pad) == :hostile
    end

    test "maps bored state correctly (P- A- D-)" do
      pad = %{pleasure: -0.5, arousal: -0.5, dominance: -0.5}
      assert Abstraction.pad_to_concept(pad) == :bored
    end

    test "maps neutral state to nearest concept" do
      pad = %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
      # Based on >= 0 logic, this should be exuberant
      assert Abstraction.pad_to_concept(pad) == :exuberant
    end
  end

  describe "abstract_state/1" do
    test "returns list of concepts" do
      state = %{
        pad: %{pleasure: 0.3, arousal: -0.2, dominance: 0.4},
        feeling: :homeostatic,
        free_energy: 0.1
      }

      concepts = Abstraction.abstract_state(state)
      assert is_list(concepts)
      assert length(concepts) > 0
      # P+ A- D+
      assert :relaxed in concepts
      # homeostatic
      assert :balanced in concepts
    end

    test "handles hardware metrics" do
      state = %{
        pad: %{pleasure: 0, arousal: 0, dominance: 0},
        hardware: %{
          cpu_temp: 85,
          cpu_usage: 90,
          memory_used_pct: 10
        }
      }

      concepts = Abstraction.abstract_state(state)
      assert :overheating in concepts
      assert :working_hard in concepts
    end
  end
end
