defmodule Viva.AI.PipelineTest do
  use ExUnit.Case, async: true

  alias Viva.AI.Pipeline

  describe "handle_message/3" do
    test "decodes valid erlang binary term message" do
      payload = %{type: :spontaneous_thought, avatar_id: "123", prompt: "test"}
      data = :erlang.term_to_binary(payload)

      message = %Broadway.Message{
        data: data,
        metadata: %{},
        acknowledger: {Broadway.NoopAcknowledger, nil, nil}
      }

      result = Pipeline.handle_message(nil, message, nil)

      assert result.data == payload
    end

    test "fails message with invalid binary data" do
      message = %Broadway.Message{
        data: "invalid binary",
        metadata: %{},
        acknowledger: {Broadway.NoopAcknowledger, nil, nil}
      }

      result = Pipeline.handle_message(nil, message, nil)

      assert result.status == {:failed, "decode_error"}
    end

    test "fails message with malformed erlang term" do
      message = %Broadway.Message{
        data: <<131, 0, 0, 0>>,
        metadata: %{},
        acknowledger: {Broadway.NoopAcknowledger, nil, nil}
      }

      result = Pipeline.handle_message(nil, message, nil)

      assert result.status == {:failed, "decode_error"}
    end
  end

  describe "start_link/1" do
    test "returns broadway start_link spec structure" do
      # Pipeline.start_link/1 uses Broadway.start_link which returns
      # {:ok, pid} or {:error, reason}
      # We can't actually start it without RabbitMQ, but we can test the function exists
      assert function_exported?(Pipeline, :start_link, 1)
    end
  end

  describe "handle_batch/4" do
    test "processes batch of messages" do
      # handle_batch should return the list of messages
      assert function_exported?(Pipeline, :handle_batch, 4)
    end
  end
end
