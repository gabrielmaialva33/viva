defmodule Viva.AI.LLM.ReasoningClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.ReasoningClient

  describe "module structure" do
    test "exports expected functions" do
      functions = ReasoningClient.__info__(:functions)

      assert {:reflect_on_memories, 2} in functions
    end

    test "module is loaded correctly" do
      assert {:module, ReasoningClient} = Code.ensure_loaded(ReasoningClient)
    end
  end
end
