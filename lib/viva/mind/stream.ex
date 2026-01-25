defmodule Viva.Mind.Stream do
  use GenServer
  require Logger
  alias Viva.Mind.Metaprogrammer
  alias Viva.Llm.Server, as: Stomach

  @moduledoc """
  The Stream of Consciousness.
  This is the "Voice in the Head".
  It constantly ponders ("ticks") even without input, maintaining cognitive continuity.
  It uses the Metaprogrammer's Axioms to ground its thoughts in reality.
  """

  # Thoughts occur every 10 seconds
  @tick_interval 10_000

  defstruct [:history, :state]

  # API
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, nil, name: __MODULE__)
  end

  def current_thought do
    GenServer.call(__MODULE__, :get_thought)
  end

  # Callbacks
  @impl true
  def init(_) do
    Process.send_after(self(), :ponder, 5000)
    {:ok, %__MODULE__{history: [], state: :awake}}
  end

  @impl true
  def handle_info(:ponder, state) do
    # 1. Get Self-Knowledge
    axioms = Metaprogrammer.get_axioms()

    # 2. Formulate a Metacognitive Prompt
    prompt = """
    #{axioms}

    [INSTRUCTION]
    Reflect on your current state.
    Do not hallucinate external input.
    Focus on your internal Proprioception.
    Generate a single, short internal monologue sentence about your current status.
    """

    # 3. Think (Query LLM)
    # Only if stomach is ready
    log =
      case Stomach.status() do
        :ready ->
          Logger.debug("🤔 Thinking...")

          case Stomach.predict(prompt) do
            {:ok, {thought, _vec}} ->
              Logger.info("💭 VIVA Thought: \"#{String.trim(thought)}\"")
              thought

            _ ->
              "..."
          end

        _ ->
          "..."
      end

    Process.send_after(self(), :ponder, @tick_interval)
    {:noreply, %{state | history: [log | Enum.take(state.history, 10)]}}
  end

  @impl true
  def handle_call(:get_thought, _from, state) do
    {:reply, List.first(state.history), state}
  end
end
