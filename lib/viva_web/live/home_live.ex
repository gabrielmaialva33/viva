defmodule VivaWeb.HomeLive do
  @moduledoc """
  LiveView for the VIVA home page - a real-time observatory of autonomous avatar life.
  Shows avatars living, thinking, and interacting in real-time.
  """
  use VivaWeb, :live_view

  import VivaWeb.AvatarComponents

  require Logger

  alias Viva.Avatars
  alias Viva.Relationships
  alias Viva.Sessions.LifeProcess
  alias Viva.World.Clock

  @max_feed_items 50

  @impl Phoenix.LiveView
  def mount(_, _, socket) do
    # Always load avatars (for both server and client render)
    avatars = load_avatars_with_state()
    relationships = load_relationships()

    socket =
      socket
      |> assign(avatars: avatars)
      |> assign(world_time: get_world_time())
      |> assign(relationships: relationships)
      |> then(fn s ->
        # Subscribe to PubSub only when connected (client-side)
        if connected?(s) do
          Phoenix.PubSub.subscribe(Viva.PubSub, "world:clock")

          Enum.each(avatars, fn {avatar_id, _} ->
            Phoenix.PubSub.subscribe(Viva.PubSub, "avatar:#{avatar_id}")
          end)
        end

        s
      end)

    {:ok,
     assign(socket,
       page_title: "VIVA - Observe Vidas Digitais",
       activity_feed: [],
       show_modal: false,
       selected_avatar: nil,
       selected_avatar_relationships: []
     )}
  end

  @impl Phoenix.LiveView
  def handle_info({:time_update, world_time}, socket) do
    {:noreply, assign(socket, world_time: world_time)}
  end

  @impl Phoenix.LiveView
  def handle_info({:thought, thought}, socket) do
    # Find which avatar this thought belongs to by checking subscriptions
    # The thought comes from the avatar channel, we need to identify which one
    # For now, we'll update the feed with the thought
    event = %{
      id: System.unique_integer([:positive]),
      type: :thought,
      content: thought,
      avatar_id: nil,
      avatar_name: nil,
      timestamp: DateTime.utc_now()
    }

    feed = prepend_to_feed(socket.assigns.activity_feed, event)
    {:noreply, assign(socket, activity_feed: feed)}
  end

  @impl Phoenix.LiveView
  def handle_info({:status, _}, socket) do
    # Avatar status update - could refresh that specific avatar's state
    {:noreply, socket}
  end

  @impl Phoenix.LiveView
  def handle_info({:state_update, _}, socket) do
    # Avatar state update
    {:noreply, socket}
  end

  @impl Phoenix.LiveView
  def handle_info(_, socket) do
    {:noreply, socket}
  end

  @impl Phoenix.LiveView
  def handle_event("select_avatar", %{"id" => id}, socket) do
    case Map.get(socket.assigns.avatars, id) do
      nil ->
        {:noreply, socket}

      avatar_data ->
        relationships = Relationships.list_for_avatar(id)

        {:noreply,
         socket
         |> assign(selected_avatar: avatar_data)
         |> assign(selected_avatar_relationships: relationships)
         |> assign(show_modal: true)}
    end
  end

  @impl Phoenix.LiveView
  def handle_event("select_avatar_from_graph", %{"id" => id}, socket) do
    handle_event("select_avatar", %{"id" => id}, socket)
  end

  @impl Phoenix.LiveView
  def handle_event("close_modal", _, socket) do
    {:noreply, assign(socket, show_modal: false)}
  end

  # === Private Functions ===

  defp load_avatars_with_state do
    avatars = Avatars.list_avatars(active: true)

    avatars
    |> Enum.map(fn avatar ->
      process_state = get_process_state(avatar.id)

      data = %{
        avatar: avatar,
        internal_state: (process_state && process_state.state) || avatar.internal_state,
        last_thought: process_state && process_state.last_thought,
        is_online: process_state != nil,
        current_activity: get_current_activity(process_state, avatar)
      }

      {avatar.id, data}
    end)
    |> Map.new()
  end

  defp get_process_state(avatar_id) do
    LifeProcess.get_state(avatar_id)
  catch
    :exit, _ -> nil
  end

  defp get_current_activity(nil, avatar) do
    (avatar.internal_state && avatar.internal_state.current_activity) || :idle
  end

  defp get_current_activity(process_state, _) do
    (process_state.state && process_state.state.current_activity) || :idle
  end

  defp get_world_time do
    Clock.now()
  catch
    :exit, _ -> DateTime.utc_now()
  end

  defp load_relationships do
    Relationships.list_all(limit: 100)
  rescue
    _ -> []
  end

  defp prepend_to_feed(feed, event) do
    [event | Enum.take(feed, @max_feed_items - 1)]
  end
end
