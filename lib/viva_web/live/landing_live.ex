defmodule VivaWeb.LandingLive do
  @moduledoc """
  Landing page LiveView for VIVA.
  Shows hero, features, live demo with real avatars, and technology stack.
  """
  use VivaWeb, :live_view

  import VivaWeb.LandingComponents
  import VivaWeb.AvatarComponents, only: [avatar_life_card: 1, relationship_graph: 1]
  import VivaWeb.NimComponents, only: [nim_grid: 1]

  alias Viva.Avatars
  alias Viva.Relationships
  alias Viva.World.Clock

  @max_demo_avatars 3

  @impl Phoenix.LiveView
  def mount(_, _, socket) do
    if connected?(socket) do
      # Subscribe to avatar updates
      Phoenix.PubSub.subscribe(Viva.PubSub, "avatars:thoughts")
      Phoenix.PubSub.subscribe(Viva.PubSub, "avatars:state")
      Phoenix.PubSub.subscribe(Viva.PubSub, "world:tick")
    end

    # Load demo avatars (first 3 active ones)
    avatars = load_demo_avatars()
    relationships = load_demo_relationships(avatars)

    # Get stats for hero section
    all_active_avatars = Avatars.list_avatars(active: true)
    active_avatar_count = length(all_active_avatars)

    # Extract avatar structs for the avatar stack (up to 5)
    demo_avatar_list =
      avatars
      |> Map.values()
      |> Enum.map(& &1.avatar)
      |> Enum.take(5)

    socket =
      socket
      |> assign(:page_title, "VIVA - Avatares Digitais Autonomos")
      |> assign(:avatars, avatars)
      |> assign(:relationships, relationships)
      |> assign(:world_time, Clock.now())
      |> assign(:active_avatar_count, active_avatar_count)
      |> assign(:demo_avatar_list, demo_avatar_list)

    {:ok, socket}
  end

  @impl Phoenix.LiveView
  def handle_info({:avatar_thought, avatar_id, thought}, socket) do
    avatars = socket.assigns.avatars

    if Map.has_key?(avatars, avatar_id) do
      avatar_data = Map.get(avatars, avatar_id)
      updated_data = Map.put(avatar_data, :last_thought, thought)
      updated_avatars = Map.put(avatars, avatar_id, updated_data)

      {:noreply, assign(socket, :avatars, updated_avatars)}
    else
      {:noreply, socket}
    end
  end

  @impl Phoenix.LiveView
  def handle_info({:avatar_state_update, avatar_id, internal_state}, socket) do
    avatars = socket.assigns.avatars

    if Map.has_key?(avatars, avatar_id) do
      avatar_data = Map.get(avatars, avatar_id)
      updated_data = Map.put(avatar_data, :internal_state, internal_state)
      updated_avatars = Map.put(avatars, avatar_id, updated_data)

      {:noreply, assign(socket, :avatars, updated_avatars)}
    else
      {:noreply, socket}
    end
  end

  @impl Phoenix.LiveView
  def handle_info({:world_tick, world_time}, socket) do
    {:noreply, assign(socket, :world_time, world_time)}
  end

  @impl Phoenix.LiveView
  def handle_info(_, socket) do
    {:noreply, socket}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp load_demo_avatars do
    [active: true]
    |> Avatars.list_avatars()
    |> Enum.take(@max_demo_avatars)
    |> Enum.map(fn avatar ->
      {avatar.id,
       %{
         avatar: avatar,
         internal_state: avatar.internal_state || default_internal_state(),
         last_thought: nil,
         is_online: true,
         current_activity: :idle
       }}
    end)
    |> Map.new()
  end

  defp load_demo_relationships(avatars) when map_size(avatars) < 2, do: []

  defp load_demo_relationships(avatars) do
    avatar_ids =
      avatars
      |> Map.keys()
      |> MapSet.new()

    [limit: 50]
    |> Relationships.list_all()
    |> Enum.filter(fn rel ->
      MapSet.member?(avatar_ids, rel.avatar_a_id) and MapSet.member?(avatar_ids, rel.avatar_b_id)
    end)
  rescue
    _ -> []
  end

  defp default_internal_state do
    %{
      energy: 70,
      social: 60,
      stimulation: 50,
      comfort: 65,
      mood: 0.2,
      emotions: %{
        joy: 0.3,
        sadness: 0.0,
        anger: 0.0,
        fear: 0.0,
        surprise: 0.1,
        disgust: 0.0,
        love: 0.0,
        loneliness: 0.1,
        curiosity: 0.4,
        excitement: 0.2
      }
    }
  end
end
