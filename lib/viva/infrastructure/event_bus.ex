defmodule Viva.Infrastructure.EventBus do
  @moduledoc """
  Wrapper for publishing events to the message broker (RabbitMQ).
  Decouples domain logic from AMQP details.
  """
  @behaviour Viva.Infrastructure.EventBusBehaviour

  require Logger

  @exchange ""
  @queue "viva.brain.thoughts"

  @spec publish_thought(map()) :: :ok | {:error, term()}
  def publish_thought(payload) do
    # In a real app, use a pool or a persistent connection process (e.g. GenServer)
    # For now, we open/close per message (inefficient but simple for MVP)
    case AMQP.Connection.open(host: System.get_env("RABBITMQ_HOST", "localhost")) do
      {:ok, conn} ->
        try do
          {:ok, chan} = AMQP.Channel.open(conn)
          AMQP.Queue.declare(chan, @queue, durable: true)
          AMQP.Basic.publish(chan, @exchange, @queue, :erlang.term_to_binary(payload))
          :ok
        after
          AMQP.Connection.close(conn)
        end

      {:error, reason} ->
        Logger.error("Failed to connect to RabbitMQ: #{inspect(reason)}")
        {:error, reason}
    end
  end
end
