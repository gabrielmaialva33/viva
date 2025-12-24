defmodule Viva.Nim.ReasoningClient do
  @moduledoc """
  Client for higher-order cognitive tasks: Reflection, Planning, and Insight Generation.
  Uses Chain-of-Thought prompting patterns via NIM.
  """
  alias Viva.Nim.LlmClient

  @doc """
  The 'Dreaming' process. Takes a list of raw memories and synthesizes high-level insights.
  """
  @spec reflect_on_memories(String.t(), [map()]) :: {:ok, map()} | {:error, any()}
  def reflect_on_memories(avatar_name, recent_memories) do
    memory_text = Enum.map_join(recent_memories, "\n", &("- " <> &1.content))

    prompt = """
    You are the subconscious mind of #{avatar_name}.
    Analyze the following recent memories and identify high-level patterns, personality traits, or relationship insights.

    RECENT MEMORIES:
    #{memory_text}

    TASK:
    1. Identify patterns in behavior or emotion.
    2. Generalize specific events into broad insights (e.g., "I get anxious when ignored" instead of "I felt bad when Bob didn't reply").
    3. Ignore trivial details.

    OUTPUT FORMAT (JSON):
    {
      "insights": [
        {"content": "Insight text...", "importance": 0.8, "related_memory_indices": [0, 2]}
      ]
    }
    """

    # We use a high temperature to encourage creative connections between disparate facts
    LlmClient.generate(prompt, max_tokens: 1000, temperature: 0.7, json_mode: true)
  end
end
