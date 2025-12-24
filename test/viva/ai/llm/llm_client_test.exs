defmodule Viva.AI.LLM.LlmClientTest do
  use ExUnit.Case, async: true

  alias Viva.AI.LLM.LlmClient

  setup_all do
    # Ensure the module is fully loaded before testing
    {:module, _} = Code.ensure_loaded(LlmClient)
    :ok
  end

  # Note: Functions with default arguments export multiple arities
  # generate(prompt, opts \\ []) exports /1 and /2
  # chat(messages, opts \\ []) exports /1 and /2
  # chat_stream(messages, callback, opts \\ []) exports /2 and /3
  # chat_with_tools(messages, tools, opts \\ []) exports /2 and /3
  # generate_avatar_response(avatar, other, history, context \\ %{}) exports /3 and /4

  describe "generate/1 and generate/2" do
    test "function with arity 1 is exported" do
      assert function_exported?(LlmClient, :generate, 1)
    end

    test "function with arity 2 is exported" do
      assert function_exported?(LlmClient, :generate, 2)
    end
  end

  describe "chat/1 and chat/2" do
    test "function with arity 1 is exported" do
      assert function_exported?(LlmClient, :chat, 1)
    end

    test "function with arity 2 is exported" do
      assert function_exported?(LlmClient, :chat, 2)
    end
  end

  describe "chat_stream/2 and chat_stream/3" do
    test "function with arity 2 is exported" do
      assert function_exported?(LlmClient, :chat_stream, 2)
    end

    test "function with arity 3 is exported" do
      assert function_exported?(LlmClient, :chat_stream, 3)
    end
  end

  describe "chat_with_tools/2 and chat_with_tools/3" do
    test "function with arity 2 is exported" do
      assert function_exported?(LlmClient, :chat_with_tools, 2)
    end

    test "function with arity 3 is exported" do
      assert function_exported?(LlmClient, :chat_with_tools, 3)
    end
  end

  describe "analyze_conversation/3" do
    test "function is exported" do
      assert function_exported?(LlmClient, :analyze_conversation, 3)
    end
  end

  describe "generate_avatar_response/3 and generate_avatar_response/4" do
    test "function with arity 3 is exported" do
      assert function_exported?(LlmClient, :generate_avatar_response, 3)
    end

    test "function with arity 4 is exported" do
      assert function_exported?(LlmClient, :generate_avatar_response, 4)
    end
  end

  describe "generate_thought/2" do
    test "function is exported" do
      assert function_exported?(LlmClient, :generate_thought, 2)
    end
  end

  describe "generate_greeting/2" do
    test "function is exported" do
      assert function_exported?(LlmClient, :generate_greeting, 2)
    end
  end
end
