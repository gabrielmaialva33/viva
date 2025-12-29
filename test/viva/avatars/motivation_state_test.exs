defmodule Viva.Avatars.MotivationStateTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.MotivationState

  describe "new/0" do
    test "returns default motivation state" do
      state = MotivationState.new()
      assert %MotivationState{} = state
      assert state.primary_drive == :safety_seeking
      assert state.survival_urgency == 0.1
    end
  end

  describe "changeset/2" do
    @valid_attrs %{
      primary_drive: :achievement,
      survival_urgency: 0.5,
      safety_urgency: 0.5,
      belonging_urgency: 0.5,
      status_urgency: 0.5,
      autonomy_urgency: 0.5,
      transcendence_urgency: 0.5,
      blocked_drive: :survival,
      block_duration: 5,
      current_urgent_drive: :survival,
      last_updated: DateTime.utc_now()
    }

    test "validates attributes correctly" do
      changeset = MotivationState.changeset(%MotivationState{}, @valid_attrs)
      assert changeset.valid?
    end

    test "invalidates out of range urgencies" do
      invalid_params_high = %{@valid_attrs | survival_urgency: 1.1}
      changeset_high = MotivationState.changeset(%MotivationState{}, invalid_params_high)
      refute changeset_high.valid?
      assert %{survival_urgency: ["must be less than or equal to 1.0"]} = errors_on(changeset_high)

      invalid_params_low = %{@valid_attrs | survival_urgency: -0.1}
      changeset_low = MotivationState.changeset(%MotivationState{}, invalid_params_low)
      refute changeset_low.valid?

      assert %{survival_urgency: ["must be greater than or equal to 0.0"]} =
               errors_on(changeset_low)
    end

    test "invalidates negative block_duration" do
      invalid_attrs = %{@valid_attrs | block_duration: -1}
      changeset = MotivationState.changeset(%MotivationState{}, invalid_attrs)
      refute changeset.valid?
      assert %{block_duration: ["must be greater than or equal to 0"]} = errors_on(changeset)
    end
  end

  describe "from_enneagram/1" do
    test "returns default state for nil" do
      assert MotivationState.from_enneagram(nil) == MotivationState.new()
    end

    test "maps Heart triad (1, 2, 3) to achievement" do
      for type <- [:type_1, :type_2, :type_3] do
        state = MotivationState.from_enneagram(type)
        assert state.primary_drive == :achievement
        assert state.status_urgency == 0.7
      end
    end

    test "maps Head triad (5, 6, 7) to safety_seeking" do
      for type <- [:type_5, :type_6, :type_7] do
        state = MotivationState.from_enneagram(type)
        assert state.primary_drive == :safety_seeking
        assert state.safety_urgency == 0.7
      end
    end

    test "maps Gut triad (4, 8, 9) to autonomy_seeking" do
      for type <- [:type_4, :type_8, :type_9] do
        state = MotivationState.from_enneagram(type)
        assert state.primary_drive == :autonomy_seeking
        assert state.autonomy_urgency == 0.7
      end
    end

    test "defaults to safety_seeking for unknown types" do
      state = MotivationState.from_enneagram(:unknown)
      assert state.primary_drive == :safety_seeking
    end
  end

  describe "urgencies/1" do
    test "returns map of urgencies" do
      state = %MotivationState{
        survival_urgency: 0.1,
        safety_urgency: 0.2,
        belonging_urgency: 0.3,
        status_urgency: 0.4,
        autonomy_urgency: 0.5,
        transcendence_urgency: 0.6
      }

      assert MotivationState.urgencies(state) == %{
               survival: 0.1,
               safety: 0.2,
               belonging: 0.3,
               status: 0.4,
               autonomy: 0.5,
               transcendence: 0.6
             }
    end
  end

  describe "frustrated?/1" do
    test "returns false if no drive is blocked" do
      refute MotivationState.frustrated?(%MotivationState{blocked_drive: nil})
    end

    test "returns true if block duration > 3" do
      assert MotivationState.frustrated?(%MotivationState{
               blocked_drive: :survival,
               block_duration: 4
             })
    end

    test "returns false if block duration <= 3" do
      refute MotivationState.frustrated?(%MotivationState{
               blocked_drive: :survival,
               block_duration: 3
             })
    end
  end

  describe "survival_mode?/1" do
    test "returns true if survival urgency > 0.7" do
      assert MotivationState.survival_mode?(%MotivationState{survival_urgency: 0.71})
    end

    test "returns false if survival urgency <= 0.7" do
      refute MotivationState.survival_mode?(%MotivationState{survival_urgency: 0.7})
    end
  end
end
