defmodule Viva.Sessions.StimulusGatheringTest do
  use Viva.DataCase, async: true
  alias Viva.Sessions.StimulusGathering

  setup do
    state = %{
      current_conversation: nil,
      owner_online?: false,
      tick_count: 5,
      state: %{
        current_activity: :idle,
        emotional: %{arousal: 0.0}
      }
    }

    {:ok, state: state}
  end

  describe "gather/1" do
    test "social stimulus during conversation", %{state: state} do
      other_id = Ecto.UUID.generate()
      state = %{state | current_conversation: other_id}

      res = StimulusGathering.gather(state)

      assert res.type == :social
      assert res.source == "conversation_partner"
      assert res.social_context == :conversation
      assert res.partner_id == other_id
      assert res.intensity >= 0.7
    end

    test "social_ambient stimulus when owner is online", %{state: state} do
      state = %{state | owner_online?: true}

      res = StimulusGathering.gather(state)

      assert res.type == :social_ambient
      assert res.source == "owner_presence"
      assert res.valence == 0.4
    end

    test "rest stimulus during sleep", %{state: state} do
      state = put_in(state.state.current_activity, :sleeping)

      res = StimulusGathering.gather(state)

      assert res.type == :rest
      assert res.intensity <= 0.3
    end

    test "ambient stimulus otherwise", %{state: state} do
      res = StimulusGathering.gather(state)

      assert res.type == :ambient
      assert res.source == "environment"
    end

    test "novelty is higher at start of connection", %{state: state} do
      state = %{state | owner_online?: true, tick_count: 1}
      res_new = StimulusGathering.gather(state)

      state_old = %{state | owner_online?: true, tick_count: 10}
      res_old = StimulusGathering.gather(state_old)

      assert res_new.novelty > res_old.novelty
    end
  end
end
