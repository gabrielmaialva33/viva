defmodule VivaWeb.HomeLive do
  @moduledoc """
  Home dashboard showing avatars living in real-time.
  Displays avatar state, thoughts, emotions, and provides quick actions.
  """
  use VivaWeb, :live_view

  alias Phoenix.PubSub
  alias Viva.Avatars
  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.InternalState
  alias Viva.Sessions.LifeProcess
  alias Viva.Sessions.Supervisor, as: SessionsSupervisor

  @impl Phoenix.LiveView
  def mount(%{"avatar_id" => avatar_id}, _, socket) do
    socket =
      socket
      |> assign(page_title: "Home")
      |> assign(loading: true)
      |> assign(avatar: nil)
      |> assign(avatars: [])
      |> assign(life_state: nil)
      |> assign(avatar_alive?: false)
      |> assign(error: nil)

    if connected?(socket) do
      send(self(), {:load_avatar, avatar_id})
    end

    {:ok, socket}
  end

  @impl Phoenix.LiveView
  def mount(_, _, socket) do
    socket =
      socket
      |> assign(page_title: "Home")
      |> assign(loading: true)
      |> assign(avatar: nil)
      |> assign(avatars: [])
      |> assign(life_state: nil)
      |> assign(avatar_alive?: false)
      |> assign(error: nil)

    if connected?(socket) do
      send(self(), :load_avatars)
    end

    {:ok, socket}
  end

  # === Handle Info ===

  @impl Phoenix.LiveView
  def handle_info(:load_avatars, socket) do
    avatars = Avatars.list_avatars(active: true)

    case avatars do
      [] ->
        {:noreply, assign(socket, loading: false, error: :no_avatars)}

      [first | _] = all_avatars ->
        socket = assign(socket, avatars: all_avatars)
        send(self(), {:load_avatar, first.id})
        {:noreply, socket}
    end
  end

  @impl Phoenix.LiveView
  def handle_info({:load_avatar, avatar_id}, socket) do
    # Unsubscribe from previous avatar if any
    if socket.assigns.avatar do
      PubSub.unsubscribe(Viva.PubSub, "avatar:#{socket.assigns.avatar.id}")
    end

    case Avatars.get_avatar(avatar_id) do
      nil ->
        {:noreply, assign(socket, loading: false, error: :not_found)}

      avatar ->
        # Subscribe to avatar events
        PubSub.subscribe(Viva.PubSub, "avatar:#{avatar.id}")

        # Ensure avatar process is running
        case ensure_avatar_running(avatar.id) do
          {:ok, life_state} ->
            {:noreply,
             socket
             |> assign(avatar: avatar)
             |> assign(life_state: life_state)
             |> assign(avatar_alive?: true)
             |> assign(loading: false)
             |> assign(error: nil)}

          {:error, _} ->
            {:noreply,
             socket
             |> assign(avatar: avatar)
             |> assign(avatar_alive?: false)
             |> assign(loading: false)
             |> assign(error: nil)}
        end
    end
  end

  @impl Phoenix.LiveView
  def handle_info({:thought, thought}, socket) do
    if socket.assigns.life_state do
      updated_state = %{socket.assigns.life_state | last_thought: thought}
      {:noreply, assign(socket, life_state: updated_state)}
    else
      {:noreply, socket}
    end
  end

  @impl Phoenix.LiveView
  def handle_info({:status, :alive}, socket) do
    if socket.assigns.avatar do
      case LifeProcess.get_state(socket.assigns.avatar.id) do
        nil ->
          {:noreply, socket}

        state ->
          {:noreply, assign(socket, life_state: state, avatar_alive?: true)}
      end
    else
      {:noreply, socket}
    end
  end

  @impl Phoenix.LiveView
  def handle_info({:greeting, greeting}, socket) do
    {:noreply, put_flash(socket, :info, greeting)}
  end

  @impl Phoenix.LiveView
  def handle_info(_, socket), do: {:noreply, socket}

  # === Events ===

  @impl Phoenix.LiveView
  def handle_event("select_avatar", %{"avatar_id" => avatar_id}, socket) do
    send(self(), {:load_avatar, avatar_id})
    {:noreply, assign(socket, loading: true)}
  end

  @impl Phoenix.LiveView
  def handle_event("trigger_thought", _, socket) do
    if socket.assigns.avatar_alive? do
      LifeProcess.trigger_thought(socket.assigns.avatar.id)
    end

    {:noreply, socket}
  end

  @impl Phoenix.LiveView
  def handle_event("refresh_state", _, socket) do
    if socket.assigns.avatar_alive? do
      state = LifeProcess.get_state(socket.assigns.avatar.id)
      {:noreply, assign(socket, life_state: state)}
    else
      {:noreply, socket}
    end
  end

  # === Render ===

  @impl Phoenix.LiveView
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash}>
      <div class="min-h-screen pb-8">
        <%= cond do %>
          <% @loading -> %>
            <.loading_state />
          <% @error == :no_avatars -> %>
            <.no_avatars_state />
          <% @error -> %>
            <.error_state error={@error} />
          <% @avatar && @life_state -> %>
            <.avatar_dashboard
              avatar={@avatar}
              avatars={@avatars}
              life_state={@life_state}
              avatar_alive?={@avatar_alive?}
            />
          <% @avatar && !@avatar_alive? -> %>
            <.avatar_offline avatar={@avatar} avatars={@avatars} />
          <% true -> %>
            <.loading_state />
        <% end %>
      </div>
    </Layouts.app>
    """
  end

  # === Private Functions ===

  defp ensure_avatar_running(avatar_id) do
    case SessionsSupervisor.start_avatar(avatar_id) do
      {:ok, _} ->
        state = LifeProcess.get_state(avatar_id)
        {:ok, state}

      {:error, {:already_started, _}} ->
        state = LifeProcess.get_state(avatar_id)
        {:ok, state}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # === Function Components ===

  defp loading_state(assigns) do
    ~H"""
    <div class="flex flex-col items-center justify-center min-h-[60vh]">
      <h1 class="text-3xl font-bold mb-6">VIVA</h1>
      <div class="loading loading-ring loading-lg text-primary"></div>
      <p class="mt-4 text-base-content/70">Connecting to avatar...</p>
    </div>
    """
  end

  defp no_avatars_state(assigns) do
    ~H"""
    <div class="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
      <div class="text-6xl mb-4">
        <.icon name="hero-user-group" class="size-16 text-primary/50" />
      </div>
      <h2 class="text-2xl font-semibold mb-2">No Avatars Found</h2>
      <p class="text-base-content/70 mb-6 max-w-md">
        Run the seeds to create avatars first.
      </p>
      <code class="bg-base-300 px-4 py-2 rounded-lg text-sm">
        mix ecto.reset
      </code>
    </div>
    """
  end

  defp error_state(assigns) do
    ~H"""
    <div class="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
      <div class="text-error mb-4">
        <.icon name="hero-exclamation-triangle" class="size-12" />
      </div>
      <h2 class="text-xl font-semibold mb-2">Something went wrong</h2>
      <p class="text-base-content/70">{inspect(@error)}</p>
    </div>
    """
  end

  defp avatar_offline(assigns) do
    ~H"""
    <div class="max-w-4xl mx-auto px-4 py-6">
      <.avatar_selector avatars={@avatars} current_id={@avatar.id} />

      <div class="flex flex-col items-center justify-center min-h-[40vh] text-center">
        <.avatar_image avatar={@avatar} size="lg" />
        <h2 class="text-xl font-semibold mt-4">{@avatar.name}</h2>
        <div class="badge badge-warning mt-2">Offline</div>
        <p class="text-base-content/70 mt-2">Avatar process is not running.</p>
        <button phx-click="refresh_state" class="btn btn-primary mt-4">
          <.icon name="hero-arrow-path" class="size-4 mr-2" /> Reconnect
        </button>
      </div>
    </div>
    """
  end

  defp avatar_dashboard(assigns) do
    internal = assigns.life_state.state

    assigns =
      assigns
      |> assign(:internal, internal)
      |> assign(:wellbeing, InternalState.wellbeing(internal))
      |> assign(:dominant_emotion, InternalState.dominant_emotion(internal))
      |> assign(:qualia, InternalState.qualia_narrative(internal))
      |> assign(:consciousness_desc, ConsciousnessState.describe_state(internal.consciousness))

    ~H"""
    <div class="max-w-4xl mx-auto px-4 py-6 space-y-6 animate-fade-in">
      <!-- Avatar Selector -->
      <.avatar_selector avatars={@avatars} current_id={@avatar.id} />
      
    <!-- Header: Avatar Identity -->
      <.avatar_header
        avatar={@avatar}
        internal={@internal}
        avatar_alive?={@avatar_alive?}
      />
      
    <!-- Mood Ring & Wellbeing -->
      <.mood_section
        wellbeing={@wellbeing}
        dominant_emotion={@dominant_emotion}
        emotional={@internal.emotional}
      />
      
    <!-- Current Activity & Thought Bubble -->
      <.activity_section
        activity={@internal.current_activity}
        desire={@internal.current_desire}
        thought={@life_state.last_thought}
      />
      
    <!-- Bio Bars -->
      <.bio_bars bio={@internal.bio} />
      
    <!-- Consciousness Narrative -->
      <.consciousness_section
        qualia={@qualia}
        consciousness_desc={@consciousness_desc}
        consciousness={@internal.consciousness}
      />
      
    <!-- Quick Actions -->
      <.quick_actions />
    </div>
    """
  end

  defp avatar_selector(assigns) do
    ~H"""
    <div :if={length(@avatars) > 1} class="flex justify-end">
      <select
        phx-change="select_avatar"
        name="avatar_id"
        class="select select-bordered select-sm"
      >
        <option :for={avatar <- @avatars} value={avatar.id} selected={avatar.id == @current_id}>
          {avatar.name}
        </option>
      </select>
    </div>
    """
  end

  defp avatar_header(assigns) do
    ~H"""
    <div class="card bg-base-200 border border-base-300">
      <div class="card-body flex-row items-center gap-4">
        <.avatar_image avatar={@avatar} size="lg" />
        <div class="flex-1 min-w-0">
          <h1 class="text-2xl font-bold truncate">{@avatar.name}</h1>
          <p class="text-base-content/70 text-sm line-clamp-2">{@avatar.bio}</p>
        </div>
        <div class="flex flex-col items-end gap-2">
          <div class={[
            "badge gap-1",
            @avatar_alive? && "badge-success",
            !@avatar_alive? && "badge-warning"
          ]}>
            <span
              :if={@avatar_alive?}
              class="size-2 rounded-full bg-success animate-pulse"
            >
            </span>
            {activity_label(@internal.current_activity)}
          </div>
          <button phx-click="refresh_state" class="btn btn-ghost btn-xs">
            <.icon name="hero-arrow-path" class="size-3" />
          </button>
        </div>
      </div>
    </div>
    """
  end

  defp avatar_image(initial_assigns) do
    size = Map.get(initial_assigns, :size, "md")

    size_class =
      case size do
        "lg" -> "w-20 h-20"
        "md" -> "w-16 h-16"
        "sm" -> "w-12 h-12"
        _ -> "w-16 h-16"
      end

    assigns = assign(initial_assigns, :size_class, size_class)

    ~H"""
    <div class="avatar">
      <div class={[
        "rounded-full ring ring-primary ring-offset-base-100 ring-offset-2",
        @size_class
      ]}>
        <img :if={@avatar.avatar_url} src={@avatar.avatar_url} alt={@avatar.name} />
        <div
          :if={!@avatar.avatar_url}
          class={[
            "bg-gradient-to-br from-primary to-secondary flex items-center justify-center",
            @size_class
          ]}
        >
          <span class="text-2xl font-bold text-primary-content">
            {String.first(@avatar.name)}
          </span>
        </div>
      </div>
    </div>
    """
  end

  defp mood_section(assigns) do
    hue = mood_to_hue(assigns.emotional.pleasure, assigns.emotional.arousal)
    assigns = assign(assigns, :mood_hue, hue)

    ~H"""
    <div class="card bg-base-200 border border-base-300">
      <div class="card-body">
        <h2 class="card-title text-sm uppercase tracking-wider text-base-content/60">
          <.icon name="hero-heart" class="size-4" /> Emotional State
        </h2>

        <div class="flex flex-col sm:flex-row items-center gap-6">
          <!-- Mood Ring -->
          <div
            class="w-24 h-24 rounded-full animate-synapse flex items-center justify-center shrink-0"
            style={"background: radial-gradient(circle, oklch(70% 0.2 #{@mood_hue}), oklch(30% 0.1 #{@mood_hue}));"}
          >
            <span class="text-3xl">{mood_emoji(@dominant_emotion)}</span>
          </div>

          <div class="flex-1 space-y-3 w-full">
            <div class="text-center sm:text-left">
              <span class="text-lg font-semibold capitalize">{@dominant_emotion}</span>
              <p class="text-sm text-base-content/60">
                {mood_description(@emotional)}
              </p>
            </div>
            
    <!-- PAD Values -->
            <div class="flex flex-wrap justify-center sm:justify-start gap-4 text-xs">
              <.pad_indicator label="Pleasure" value={@emotional.pleasure} />
              <.pad_indicator label="Arousal" value={@emotional.arousal} />
              <.pad_indicator label="Dominance" value={@emotional.dominance} />
            </div>
            
    <!-- Wellbeing Bar -->
            <div>
              <div class="flex justify-between text-xs mb-1">
                <span>Wellbeing</span>
                <span>{round(@wellbeing * 100)}%</span>
              </div>
              <progress
                class="progress progress-success w-full"
                value={@wellbeing * 100}
                max="100"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
    """
  end

  defp pad_indicator(assigns) do
    ~H"""
    <div class="flex items-center gap-1">
      <span class="text-base-content/50">{@label}:</span>
      <span class={[
        "font-mono",
        @value > 0.3 && "text-success",
        @value < -0.3 && "text-error",
        @value >= -0.3 && @value <= 0.3 && "text-base-content/70"
      ]}>
        {format_pad_value(@value)}
      </span>
    </div>
    """
  end

  defp activity_section(assigns) do
    ~H"""
    <div class="card bg-base-200 border border-base-300">
      <div class="card-body">
        <h2 class="card-title text-sm uppercase tracking-wider text-base-content/60">
          <.icon name="hero-bolt" class="size-4" /> Current State
        </h2>

        <div class="flex flex-col sm:flex-row items-start gap-4">
          <!-- Activity Icon -->
          <div class="p-4 rounded-xl bg-base-300 shrink-0">
            <.icon name={activity_icon(@activity)} class="size-8 text-primary" />
          </div>

          <div class="flex-1 w-full">
            <div class="flex flex-wrap items-center gap-2 mb-3">
              <span class="text-lg font-semibold capitalize">
                {activity_label(@activity)}
              </span>
              <span
                :if={@desire != :none}
                class="badge badge-secondary badge-outline badge-sm"
              >
                {desire_label(@desire)}
              </span>
            </div>
            
    <!-- Thought Bubble -->
            <div
              :if={@thought}
              class="relative bg-base-100 rounded-2xl rounded-tl-sm p-4 border border-base-300"
            >
              <div class="absolute -left-2 top-4 w-0 h-0 border-t-8 border-t-transparent border-r-8 border-r-base-100 border-b-8 border-b-transparent">
              </div>
              <p class="text-sm italic text-base-content/80">
                "{@thought}"
              </p>
            </div>

            <p :if={!@thought} class="text-sm text-base-content/50 italic">
              No recent thoughts...
            </p>

            <button phx-click="trigger_thought" class="btn btn-ghost btn-xs mt-3">
              <.icon name="hero-sparkles" class="size-3" /> Prompt thought
            </button>
          </div>
        </div>
      </div>
    </div>
    """
  end

  defp bio_bars(assigns) do
    ~H"""
    <div class="card bg-base-200 border border-base-300">
      <div class="card-body">
        <h2 class="card-title text-sm uppercase tracking-wider text-base-content/60">
          <.icon name="hero-beaker" class="size-4" /> Bio State
        </h2>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <.bio_bar
            label="Energy"
            value={1.0 - @bio.adenosine}
            color="primary"
            icon="hero-bolt"
          />
          <.bio_bar
            label="Stress"
            value={@bio.cortisol}
            color="error"
            icon="hero-exclamation-triangle"
          />
          <.bio_bar label="Bonding" value={@bio.oxytocin} color="secondary" icon="hero-heart" />
          <.bio_bar label="Reward" value={@bio.dopamine} color="accent" icon="hero-star" />
        </div>
      </div>
    </div>
    """
  end

  defp bio_bar(assigns) do
    ~H"""
    <div class="space-y-2">
      <div class="flex items-center justify-between text-sm">
        <span class="flex items-center gap-1">
          <.icon name={@icon} class="size-4" />
          {@label}
        </span>
        <span class="font-mono text-xs">{round(@value * 100)}%</span>
      </div>
      <progress class={["progress w-full", "progress-#{@color}"]} value={@value * 100} max="100" />
    </div>
    """
  end

  defp consciousness_section(assigns) do
    ~H"""
    <div class="card bg-base-200 border border-base-300">
      <div class="card-body">
        <h2 class="card-title text-sm uppercase tracking-wider text-base-content/60">
          <.icon name="hero-eye" class="size-4" /> Consciousness
        </h2>

        <div class="space-y-4">
          <!-- Badges -->
          <div class="flex flex-wrap gap-2 text-sm">
            <div class="badge badge-outline">
              Tempo: {tempo_label(@consciousness.stream_tempo)}
            </div>
            <div class="badge badge-outline">
              Presence: {round(@consciousness.presence_level * 100)}%
            </div>
            <div class="badge badge-outline">
              Focus: {temporal_focus_label(@consciousness.temporal_focus)}
            </div>
          </div>
          
    <!-- Consciousness Description -->
          <p class="text-base-content/80">{@consciousness_desc}</p>
          
    <!-- Qualia Narrative -->
          <div :if={@qualia} class="bg-base-100 rounded-lg p-4 border-l-4 border-primary">
            <p class="text-sm italic">{@qualia}</p>
          </div>
          
    <!-- Meta Observation -->
          <div :if={@consciousness.meta_observation} class="text-xs text-base-content/60">
            <span class="font-medium">Meta:</span> {@consciousness.meta_observation}
          </div>
        </div>
      </div>
    </div>
    """
  end

  defp quick_actions(assigns) do
    ~H"""
    <div class="flex flex-wrap gap-3 justify-center pt-4">
      <.button disabled class="opacity-50 cursor-not-allowed">
        <.icon name="hero-chat-bubble-left-right" class="size-5 mr-2" /> Chat
      </.button>
      <.button disabled class="opacity-50 cursor-not-allowed">
        <.icon name="hero-sparkles" class="size-5 mr-2" /> Memories
      </.button>
      <.button disabled class="opacity-50 cursor-not-allowed">
        <.icon name="hero-users" class="size-5 mr-2" /> Relationships
      </.button>
      <.button navigate={~p"/discover"}>
        <.icon name="hero-globe-alt" class="size-5 mr-2" /> Discover
      </.button>
    </div>
    """
  end

  # === Helper Functions ===

  defp activity_label(:idle), do: "Idle"
  defp activity_label(:resting), do: "Resting"
  defp activity_label(:thinking), do: "Thinking"
  defp activity_label(:talking), do: "Talking"
  defp activity_label(:waiting), do: "Waiting"
  defp activity_label(:excited), do: "Excited"
  defp activity_label(:sleeping), do: "Sleeping"
  defp activity_label(_), do: "Active"

  defp activity_icon(:idle), do: "hero-minus-circle"
  defp activity_icon(:resting), do: "hero-moon"
  defp activity_icon(:thinking), do: "hero-light-bulb"
  defp activity_icon(:talking), do: "hero-chat-bubble-left-right"
  defp activity_icon(:waiting), do: "hero-clock"
  defp activity_icon(:excited), do: "hero-bolt"
  defp activity_icon(:sleeping), do: "hero-moon"
  defp activity_icon(_), do: "hero-user"

  defp desire_label(:none), do: nil
  defp desire_label(:wants_to_talk), do: "Wants to talk"
  defp desire_label(:wants_to_see_crush), do: "Thinking of someone"
  defp desire_label(:wants_something_new), do: "Craving novelty"
  defp desire_label(:wants_rest), do: "Needs rest"
  defp desire_label(:wants_attention), do: "Wants attention"
  defp desire_label(:wants_to_express), do: "Wants to express"
  defp desire_label(other), do: to_string(other)

  defp tempo_label(:frozen), do: "Frozen"
  defp tempo_label(:slow), do: "Slow"
  defp tempo_label(:normal), do: "Normal"
  defp tempo_label(:fast), do: "Fast"
  defp tempo_label(:racing), do: "Racing"
  defp tempo_label(_), do: "Normal"

  defp temporal_focus_label(:past), do: "Past"
  defp temporal_focus_label(:present), do: "Present"
  defp temporal_focus_label(:future), do: "Future"
  defp temporal_focus_label(_), do: "Present"

  defp mood_emoji("happy"), do: "😊"
  defp mood_emoji("content"), do: "😌"
  defp mood_emoji("excited"), do: "🤩"
  defp mood_emoji("anxious"), do: "😰"
  defp mood_emoji("sad"), do: "😢"
  defp mood_emoji("angry"), do: "😠"
  defp mood_emoji("loving"), do: "🥰"
  defp mood_emoji("peaceful"), do: "😇"
  defp mood_emoji("neutral"), do: "😐"
  defp mood_emoji("serene"), do: "😌"
  defp mood_emoji("hostile"), do: "😤"
  defp mood_emoji("depressed"), do: "😔"
  defp mood_emoji("critical"), do: "🤨"
  defp mood_emoji("ambitious"), do: "😎"
  defp mood_emoji("relaxed"), do: "😊"
  defp mood_emoji(_), do: "🙂"

  defp mood_to_hue(pleasure, arousal) do
    cond do
      pleasure > 0 && arousal > 0 -> 60
      pleasure > 0 && arousal <= 0 -> 160
      pleasure <= 0 && arousal > 0 -> 0
      true -> 280
    end
  end

  defp mood_description(emotional) do
    cond do
      emotional.pleasure > 0.5 && emotional.arousal > 0.3 ->
        "Feeling energized and positive"

      emotional.pleasure > 0.3 && emotional.arousal < 0 ->
        "Calm and content"

      emotional.pleasure < -0.3 && emotional.arousal > 0.3 ->
        "Tense and uneasy"

      emotional.pleasure < -0.3 ->
        "Feeling down"

      true ->
        "In a neutral state"
    end
  end

  defp format_pad_value(value) when value >= 0, do: "+#{Float.round(value, 2)}"
  defp format_pad_value(value), do: "#{Float.round(value, 2)}"
end
