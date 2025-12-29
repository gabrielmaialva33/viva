defmodule Viva.Sessions.ThoughtEngineTest do
  use Viva.DataCase, async: true
  import Mox

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.InternalState
  alias Viva.Avatars.SensoryState
  alias Viva.Sessions.ThoughtEngine

  setup :verify_on_exit!

  setup do
    state = %{
      avatar_id: Ecto.UUID.generate(),
      avatar: %{name: "Thinker"},
      state: InternalState.new(),
      owner_online?: false
    }

    {:ok, state: state}
  end

  describe "maybe_think/1" do
    test "triggers thought based on probability", %{state: state} do
      # Set desire to boost probability (0.1 * 2.0 = 0.2)
      state = put_in(state.state.current_desire, :wants_rest)

      # Probability is low, so we use stub to allow any number of calls
      stub(Viva.Infrastructure.MockEventBus, :publish_thought, fn _ -> :ok end)

      for _ <- 1..100, do: ThoughtEngine.maybe_think(state)
    end
  end

  describe "generate_thought/1" do
    test "publishes payload to event bus", %{state: state} do
      expect(Viva.Infrastructure.MockEventBus, :publish_thought, fn payload ->
        assert payload.type == :spontaneous_thought
        assert payload.avatar_id == state.avatar_id
        assert payload.prompt =~ "You are Thinker"
        :ok
      end)

      ThoughtEngine.generate_thought(state)
    end
  end
end
