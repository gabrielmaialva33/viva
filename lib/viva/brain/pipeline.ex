defmodule Viva.Brain.Pipeline do
  @moduledoc """
  Broadway pipeline that processes heavy cognitive tasks (LLM generation)
  asynchronously using RabbitMQ.

  It consumes messages from 'viva.brain.thoughts', generates the content via NVIDIA NIM,
  and casts the result back to the avatar's LifeProcess.
  """
  use Broadway

  require Logger

  alias Viva.Nim.LlmClient
  alias Viva.Sessions.LifeProcess

  def start_link(_opts) do
    Broadway.start_link(__MODULE__,
      name: __MODULE__,
      producer: [
        module:
          {BroadwayRabbitMQ.Producer,
           queue: "viva.brain.thoughts",
           declare: [durable: true],
           on_failure: :reject_and_requeue,
           connection: [
             host: System.get_env("RABBITMQ_HOST", "localhost"),
             username: System.get_env("RABBITMQ_USER", "guest"),
             password: System.get_env("RABBITMQ_PASS", "guest")
           ]}
      ],
      processors: [
        default: [
          concurrency: 5
        ]
      ],
      batchers: [
        default: [
          batch_size: 10,
          batch_timeout: 2000,
          concurrency: 5
        ]
      ]
    )
  end

  @impl true
  def handle_message(_, %{data: data} = message, _) do
    # RabbitMQ sends data as binary JSON (or erlang term if we configure it)
    # For simplicity, we assume we publish :erlang.term_to_binary
    decoded = :erlang.binary_to_term(data)

    # Enrich message with decoded payload
    Broadway.Message.put_data(message, decoded)
  rescue
    e ->
      Logger.error("Failed to decode message: #{inspect(e)}")
      Broadway.Message.failed(message, "decode_error")
  end

  @impl true
  def handle_batch(_, messages, _, _) do
    # Process batch in parallel or singly. Since LLM calls are IO bound,
    # we can use Task.async_stream here or just let the processors handle it.
    # Broadway's "processors" layer is better for per-message work.

    # Important: In this architecture, we want per-message processing because
    # each thought is independent and takes ~1-2s.

    messages
    |> Enum.map(fn message ->
      try do
        process_thought(message.data)
        message
      rescue
        e ->
          Logger.error("Failed to process thought: #{inspect(e)}")
          Broadway.Message.failed(message, "processing_error")
      end
    end)
  end

  defp process_thought(%{type: :spontaneous_thought, avatar_id: avatar_id, prompt: prompt}) do
    Logger.debug("🧠 Broadway processing thought for Avatar #{avatar_id}")

    case LlmClient.generate(prompt, max_tokens: 100) do
      {:ok, content} ->
        # Send result back to the avatar
        LifeProcess.set_thought(avatar_id, content)

      {:error, reason} ->
        Logger.warning("LLM generation failed: #{inspect(reason)}")
        raise "LLM Error"
    end
  end

  defp process_thought(payload) do
    Logger.warning("Unknown payload type: #{inspect(payload)}")
  end
end
