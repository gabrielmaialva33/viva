defmodule Viva.Avatars.EmotionRegulationStateTest do
  use Viva.DataCase, async: true
  alias Viva.Avatars.EmotionRegulationState

  describe "new/0" do
    test "returns default state" do
      s = EmotionRegulationState.new()
      assert s.active_strategy == nil
      assert s.reappraise_effectiveness == 0.5
    end
  end

  describe "changeset/2" do
    test "validates exhaustion range" do
      params = %{regulation_exhaustion: 1.1}
      changeset = EmotionRegulationState.changeset(%EmotionRegulationState{}, params)
      refute changeset.valid?
      assert %{regulation_exhaustion: ["must be less than or equal to 1.0"]} = errors_on(changeset)
    end
  end

  describe "query functions" do
    test "dominant_strategy/1 returns strategy with max count" do
      s = %EmotionRegulationState{
        ruminate_count: 5,
        reappraise_count: 10,
        distract_count: 2
      }

      assert EmotionRegulationState.dominant_strategy(s) == :reappraise
    end

    test "dominant_strategy/1 returns nil if no attempts" do
      assert EmotionRegulationState.dominant_strategy(EmotionRegulationState.new()) == nil
    end

    test "total_attempts/1 sums all counts" do
      s = %EmotionRegulationState{
        ruminate_count: 1,
        reappraise_count: 1,
        seek_support_count: 1,
        suppress_count: 1,
        distract_count: 1
      }

      assert EmotionRegulationState.total_attempts(s) == 5
    end
  end
end
