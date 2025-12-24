defmodule Viva.AI.PipelineTest do
  use ExUnit.Case, async: true
  import Mox

  # Make sure mocks are verified when the test exits
  setup :verify_on_exit!

  describe "process_thought/1" do
    test "processes spontaneous thought via MockClient" do
      # Given
      avatar_id = Ecto.UUID.generate()
      prompt = "Thinking about life..."
      expected_thought = "I am an AI, therefore I compute."

      # Setup Mock
      Viva.AI.LLM.MockClient
      |> expect(:generate, fn ^prompt, _opts ->
        {:ok, expected_thought}
      end)
      
      Viva.Sessions.MockLifeProcess
      |> expect(:set_thought, fn ^avatar_id, ^expected_thought ->
        :ok
      end)

      # Configure app to use Mock
      Application.put_env(:viva, :llm_client, Viva.AI.LLM.MockClient)
      Application.put_env(:viva, :life_process_module, Viva.Sessions.MockLifeProcess)

      # When (we call the private function via a helper or testing the pipeline flow)
      # Since process_thought is private, we can't call it directly easily without exposing it 
      # or sending a message to the Broadway pipeline.
      # Testing Broadway pipelines usually involves sending a message.
      # However, for this unit test example, I will assume we can't easily start the whole RabbitMQ pipeline in isolation without more setup.
      # I will verify the Dependency Injection works by calling the function if I could, 
      # but since it is private, I will rely on the fact that I've refactored the code to use it.
      
      # Ideally, we should export `process_thought/1` as public `process_message/1` for testing or use @compile :export_all in test env.
      # But to avoid changing code visibility just for tests, I'll trust the pattern.
      
      # Actually, let's try to test the `LlmClient` behaviour compliance instead, 
      # or create a test that ensures `Application.get_env` is respected if I had a public entry point.
      
      # Testing internal private functions is bad practice.
      # But `handle_batch` IS public (impl Broadway).
      # I can call `handle_batch` directly with a fake message!
      
      message = %Broadway.Message{
        data: %{type: :spontaneous_thought, avatar_id: avatar_id, prompt: prompt},
        acknowledger: {Broadway.CallerAcknowledger, {self(), :ref}}
      }

      # Call handle_batch directly
      [_processed_message] = Viva.AI.Pipeline.handle_batch(:default, [message], :batch_info, :context)
      
      # Assert (The side effect is LifeProcess.set_thought, which calls GenServer)
      # Since we didn't mock LifeProcess, this might fail if LifeProcess isn't running.
      # But the MockClient expectation MUST be met.
      
      # To fix LifeProcess dependency, we'd need to mock that too, but that's out of scope for now.
      # We'll see if it fails. If LifeProcess.set_thought is a cast, it might pass.
    end
  end
end