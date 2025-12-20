defmodule VivaWeb.WorldChannel do
  @moduledoc """
  Channel for world-wide events and broadcasts.
  """
  use VivaWeb, :channel

  alias Viva.World.Clock

  @impl true
  def join("world:clock", _payload, socket) do
    # Subscribe to clock updates
    Phoenix.PubSub.subscribe(Viva.PubSub, "world:clock")

    {:ok, %{world_time: Clock.now()}, socket}
  end

  def join("world:events", _payload, socket) do
    Phoenix.PubSub.subscribe(Viva.PubSub, "world:events")
    {:ok, socket}
  end

  def join("world:" <> _subtopic, _payload, _socket) do
    {:error, %{reason: "unknown topic"}}
  end

  @impl true
  def handle_info({:time_update, world_time}, socket) do
    push(socket, "time_update", %{world_time: world_time})
    {:noreply, socket}
  end

  def handle_info({:world_event, event}, socket) do
    push(socket, "world_event", event)
    {:noreply, socket}
  end

  @impl true
  def handle_in("get_time", _payload, socket) do
    {:reply, {:ok, %{world_time: Clock.now()}}, socket}
  end
end
