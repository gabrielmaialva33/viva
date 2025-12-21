defmodule VivaWeb.WorldChannel do
  @moduledoc """
  Channel for world-wide events and broadcasts.
  """
  use VivaWeb, :channel

  alias Viva.World.Clock

  @impl Phoenix.Channel
  def join("world:clock", _, socket) do
    # Subscribe to clock updates
    Phoenix.PubSub.subscribe(Viva.PubSub, "world:clock")

    {:ok, %{world_time: Clock.now()}, socket}
  end

  @impl Phoenix.Channel
  def join("world:events", _, socket) do
    Phoenix.PubSub.subscribe(Viva.PubSub, "world:events")
    {:ok, socket}
  end

  @impl Phoenix.Channel
  def join("world:" <> _, _, _) do
    {:error, %{reason: "unknown topic"}}
  end

  @impl Phoenix.Channel
  def handle_info({:time_update, world_time}, socket) do
    push(socket, "time_update", %{world_time: world_time})
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_info({:world_event, event}, socket) do
    push(socket, "world_event", event)
    {:noreply, socket}
  end

  @impl Phoenix.Channel
  def handle_in("get_time", _, socket) do
    {:reply, {:ok, %{world_time: Clock.now()}}, socket}
  end
end
