defmodule VivaWeb.AvatarComponents do
  @moduledoc """
  Components for displaying avatars with personality traits, emotions, and status.
  Includes real-time life observatory components.
  """
  use Phoenix.Component

  import VivaWeb.CoreComponents, only: [icon: 1]

  alias Viva.Avatars.Personality

  # ============================================================================
  # WORLD CLOCK COMPONENT
  # ============================================================================

  @doc """
  Renders the world clock showing simulation time.
  """
  attr :world_time, DateTime, required: true
  attr :online_count, :integer, default: 0

  @spec world_clock(map()) :: Phoenix.LiveView.Rendered.t()
  def world_clock(assigns) do
    ~H"""
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-800/60 border border-white/5">
        <span class="w-2 h-2 rounded-full bg-cyan-400 animate-synapse"></span>
        <span class="text-sm font-mono text-zinc-300">
          {Calendar.strftime(@world_time, "%H:%M")}
        </span>
        <span class="text-xs text-zinc-500">
          {Calendar.strftime(@world_time, "%d %b")}
        </span>
      </div>
      <div class="text-xs text-zinc-500">
        {@online_count} avatars online
      </div>
    </div>
    """
  end

  # ============================================================================
  # AVATAR LIFE CARD - Enhanced card showing live state
  # ============================================================================

  @doc """
  Renders an enhanced avatar card with live state information.
  Shows needs, emotions, activity, and recent thoughts.
  """
  attr :avatar, :map, required: true
  attr :internal_state, :map, required: true
  attr :last_thought, :string, default: nil
  attr :is_online, :boolean, default: false
  attr :current_activity, :atom, default: :idle
  attr :on_click, :string, default: nil

  @spec avatar_life_card(map()) :: Phoenix.LiveView.Rendered.t()
  def avatar_life_card(assigns) do
    ~H"""
    <div
      class={[
        "relative group p-5 rounded-2xl bg-zinc-900/50 border border-white/5 backdrop-blur-md transition-all duration-300",
        "hover:bg-zinc-800/50 hover:border-white/10 hover:-translate-y-1 hover:shadow-xl cursor-pointer",
        if(@is_online, do: "ring-1 ring-cyan-500/20", else: "")
      ]}
      phx-click={@on_click}
      phx-value-id={@avatar.id}
    >
      <!-- Top section: Avatar image + basic info -->
      <div class="flex items-start gap-4">
        <!-- Image with status -->
        <div class="relative flex-shrink-0">
          <div class="w-16 h-16 rounded-full p-0.5 bg-gradient-to-b from-white/10 to-transparent border border-white/5">
            <img
              src={get_avatar_image(@avatar)}
              alt={@avatar.name}
              class="w-full h-full rounded-full object-cover"
            />
          </div>
          <.life_pulse is_online={@is_online} mood={@internal_state && @internal_state.mood} />
        </div>
        
    <!-- Name + activity + emotion -->
        <div class="flex-1 min-w-0">
          <h3 class="text-base font-semibold text-white truncate">{@avatar.name}</h3>
          <div class="flex items-center gap-2 mt-1">
            <.activity_badge activity={@current_activity} />
            <span class="text-zinc-600">·</span>
            <.mood_indicator internal_state={@internal_state} />
          </div>
        </div>
      </div>
      
    <!-- Needs bars -->
      <div class="mt-4">
        <.needs_bars internal_state={@internal_state} />
      </div>
      
    <!-- Last thought (if recent) -->
      <div :if={@last_thought} class="mt-3 p-2 rounded-lg bg-zinc-800/40 border border-white/5">
        <p class="text-xs text-zinc-400 italic line-clamp-2">"{@last_thought}"</p>
      </div>
    </div>
    """
  end

  @doc """
  Renders the life pulse indicator showing avatar online/mood status.
  """
  attr :is_online, :boolean, required: true
  attr :mood, :float, default: 0.0

  @spec life_pulse(map()) :: Phoenix.LiveView.Rendered.t()
  def life_pulse(assigns) do
    mood = assigns.mood || 0.0
    assigns = assign(assigns, mood: mood)

    ~H"""
    <span class={[
      "absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full border-2 border-zinc-900",
      life_pulse_color(@is_online, @mood)
    ]}>
    </span>
    """
  end

  defp life_pulse_color(true, mood) when mood > 0.3, do: "bg-cyan-400 animate-pulse"
  defp life_pulse_color(true, mood) when mood < -0.3, do: "bg-rose-400"
  defp life_pulse_color(true, _), do: "bg-fuchsia-400"
  defp life_pulse_color(false, _), do: "bg-zinc-600"

  # ============================================================================
  # NEEDS BARS - Compact horizontal needs visualization
  # ============================================================================

  @doc """
  Renders compact needs bars for energy, social, stimulation, and comfort.
  """
  attr :internal_state, :map, required: true
  attr :class, :string, default: nil

  @spec needs_bars(map()) :: Phoenix.LiveView.Rendered.t()
  def needs_bars(assigns) do
    internal = assigns.internal_state || %{energy: 50, social: 50, stimulation: 50, comfort: 50}
    assigns = assign(assigns, internal: internal)

    ~H"""
    <div class={["grid grid-cols-4 gap-2", @class]}>
      <.need_bar_compact label="E" value={@internal.energy || 50} color="yellow" title="Energia" />
      <.need_bar_compact label="S" value={@internal.social || 50} color="blue" title="Social" />
      <.need_bar_compact
        label="St"
        value={@internal.stimulation || 50}
        color="purple"
        title="Estimulação"
      />
      <.need_bar_compact label="C" value={@internal.comfort || 50} color="green" title="Conforto" />
    </div>
    """
  end

  attr :label, :string, required: true
  attr :value, :float, required: true
  attr :color, :string, required: true
  attr :title, :string, required: true

  defp need_bar_compact(assigns) do
    percentage = min(100, max(0, round(assigns.value)))
    assigns = assign(assigns, percentage: percentage)

    ~H"""
    <div class="flex flex-col items-center gap-1" title={@title}>
      <div class="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          class={["h-full rounded-full transition-all duration-500", need_bar_bg(@color, @percentage)]}
          style={"width: #{@percentage}%"}
        >
        </div>
      </div>
      <span class={["text-[10px] font-medium", need_label_color(@color)]}>{@label}</span>
    </div>
    """
  end

  defp need_bar_bg("yellow", p) when p < 30, do: "bg-red-500"
  defp need_bar_bg("yellow", _), do: "bg-yellow-500"
  defp need_bar_bg("blue", p) when p < 30, do: "bg-red-500"
  defp need_bar_bg("blue", _), do: "bg-blue-500"
  defp need_bar_bg("purple", p) when p < 30, do: "bg-red-500"
  defp need_bar_bg("purple", _), do: "bg-purple-500"
  defp need_bar_bg("green", p) when p < 30, do: "bg-red-500"
  defp need_bar_bg("green", _), do: "bg-green-500"
  defp need_bar_bg(_, _), do: "bg-zinc-500"

  defp need_label_color("yellow"), do: "text-yellow-500/70"
  defp need_label_color("blue"), do: "text-blue-500/70"
  defp need_label_color("purple"), do: "text-purple-500/70"
  defp need_label_color("green"), do: "text-green-500/70"
  defp need_label_color(_), do: "text-zinc-500"

  # ============================================================================
  # ACTIVITY BADGE - Shows current avatar activity
  # ============================================================================

  @doc """
  Renders a badge showing the avatar's current activity.
  """
  attr :activity, :atom, required: true

  @spec activity_badge(map()) :: Phoenix.LiveView.Rendered.t()
  def activity_badge(assigns) do
    ~H"""
    <div class="flex items-center gap-1 text-xs text-zinc-400">
      <.icon name={activity_icon(@activity)} class="w-3 h-3" />
      <span>{activity_label(@activity)}</span>
    </div>
    """
  end

  defp activity_icon(:idle), do: "hero-moon-mini"
  defp activity_icon(:resting), do: "hero-moon-mini"
  defp activity_icon(:thinking), do: "hero-light-bulb-mini"
  defp activity_icon(:talking), do: "hero-chat-bubble-left-right-mini"
  defp activity_icon(:waiting), do: "hero-clock-mini"
  defp activity_icon(:excited), do: "hero-sparkles-mini"
  defp activity_icon(_), do: "hero-ellipsis-horizontal-mini"

  defp activity_label(:idle), do: "Tranquilo"
  defp activity_label(:resting), do: "Descansando"
  defp activity_label(:thinking), do: "Pensando"
  defp activity_label(:talking), do: "Conversando"
  defp activity_label(:waiting), do: "Aguardando"
  defp activity_label(:excited), do: "Empolgado"
  defp activity_label(_), do: "—"

  # ============================================================================
  # ACTIVITY FEED COMPONENTS
  # ============================================================================

  @doc """
  Renders the activity feed container.
  """
  attr :events, :list, required: true
  attr :class, :string, default: nil

  @spec activity_feed(map()) :: Phoenix.LiveView.Rendered.t()
  def activity_feed(assigns) do
    ~H"""
    <div class={[
      "space-y-2 overflow-y-auto max-h-[600px] scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent",
      @class
    ]}>
      <div :if={@events == []} class="flex flex-col items-center justify-center py-12 text-zinc-500">
        <.icon name="hero-clock" class="w-8 h-8 mb-2 opacity-50" />
        <p class="text-sm">Aguardando atividade...</p>
      </div>
      <.activity_item :for={event <- @events} event={event} />
    </div>
    """
  end

  @doc """
  Renders a single activity item in the feed.
  """
  attr :event, :map, required: true

  @spec activity_item(map()) :: Phoenix.LiveView.Rendered.t()
  def activity_item(assigns) do
    ~H"""
    <div class="flex gap-3 p-3 rounded-xl bg-zinc-900/40 border border-white/5 hover:bg-zinc-800/40 transition-colors">
      <div class={[
        "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
        event_icon_bg(@event.type)
      ]}>
        <.icon name={event_icon(@event.type)} class="w-4 h-4 text-white" />
      </div>
      <div class="flex-1 min-w-0">
        <p class="text-sm text-zinc-300 line-clamp-2">{@event.content}</p>
        <div class="flex items-center gap-2 mt-1">
          <span :if={@event.avatar_name} class="text-xs text-zinc-500">{@event.avatar_name}</span>
          <span class="text-xs text-zinc-600">{format_time(@event.timestamp)}</span>
        </div>
      </div>
    </div>
    """
  end

  defp event_icon(:thought), do: "hero-light-bulb-mini"
  defp event_icon(:conversation_started), do: "hero-chat-bubble-left-right-mini"
  defp event_icon(:conversation_ended), do: "hero-check-circle-mini"
  defp event_icon(:relationship_evolved), do: "hero-heart-mini"
  defp event_icon(:memory_formed), do: "hero-sparkles-mini"
  defp event_icon(:need_critical), do: "hero-exclamation-triangle-mini"
  defp event_icon(_), do: "hero-bell-mini"

  defp event_icon_bg(:thought), do: "bg-amber-500/20"
  defp event_icon_bg(:conversation_started), do: "bg-blue-500/20"
  defp event_icon_bg(:conversation_ended), do: "bg-green-500/20"
  defp event_icon_bg(:relationship_evolved), do: "bg-pink-500/20"
  defp event_icon_bg(:memory_formed), do: "bg-purple-500/20"
  defp event_icon_bg(:need_critical), do: "bg-red-500/20"
  defp event_icon_bg(_), do: "bg-zinc-500/20"

  defp format_time(nil), do: ""

  defp format_time(datetime) do
    now = DateTime.utc_now()
    diff = DateTime.diff(now, datetime, :second)

    cond do
      diff < 60 -> "agora"
      diff < 3600 -> "há #{div(diff, 60)} min"
      diff < 86_400 -> "há #{div(diff, 3600)} h"
      true -> Calendar.strftime(datetime, "%d/%m %H:%M")
    end
  end

  # ============================================================================
  # RELATIONSHIP GRAPH COMPONENT
  # ============================================================================

  @doc """
  Renders the relationship graph container.
  Uses a JavaScript hook with vis-network for visualization.
  """
  attr :relationships, :list, required: true
  attr :avatars, :map, required: true
  attr :id, :string, default: "relationship-graph"

  @spec relationship_graph(map()) :: Phoenix.LiveView.Rendered.t()
  def relationship_graph(assigns) do
    relationships_data = format_relationships_for_graph(assigns.relationships)
    avatars_data = format_avatars_for_graph(assigns.avatars)

    assigns =
      assigns
      |> assign(relationships_json: Jason.encode!(relationships_data))
      |> assign(avatars_json: Jason.encode!(avatars_data))

    ~H"""
    <div
      id={@id}
      phx-hook="RelationshipGraph"
      data-relationships={@relationships_json}
      data-avatars={@avatars_json}
      class="w-full h-[350px] bg-zinc-900/40 rounded-2xl border border-white/5 overflow-hidden"
    >
      <div class="flex items-center justify-center h-full text-zinc-500">
        <.icon name="hero-arrow-path" class="w-6 h-6 animate-spin" />
      </div>
    </div>
    """
  end

  defp format_relationships_for_graph(relationships) do
    Enum.map(relationships, fn rel ->
      %{
        from: rel.avatar_a_id,
        to: rel.avatar_b_id,
        status: rel.status,
        strength: calculate_relationship_strength(rel)
      }
    end)
  end

  defp calculate_relationship_strength(rel) do
    ((rel.trust || 0) + (rel.affection || 0) + (rel.familiarity || 0)) / 3
  end

  defp format_avatars_for_graph(avatars) do
    Enum.map(avatars, fn {id, data} ->
      %{
        id: id,
        name: data.avatar.name,
        image: get_avatar_image(data.avatar)
      }
    end)
  end

  # ============================================================================
  # AVATAR DETAIL MODAL
  # ============================================================================

  @doc """
  Renders a modal with full avatar details.
  """
  attr :avatar, :map, required: true
  attr :internal_state, :map, required: true
  attr :relationships, :list, default: []
  attr :on_close, :string, required: true

  @spec avatar_detail_modal(map()) :: Phoenix.LiveView.Rendered.t()
  def avatar_detail_modal(assigns) do
    ~H"""
    <div
      class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm"
      phx-click={@on_close}
    >
      <div
        class="w-full max-w-xl max-h-[85vh] overflow-y-auto bg-zinc-900 rounded-3xl border border-white/10 shadow-2xl"
        phx-click-away={@on_close}
      >
        <!-- Header -->
        <div class="relative h-32 bg-gradient-to-b from-zinc-800 to-zinc-900 rounded-t-3xl">
          <button
            phx-click={@on_close}
            class="absolute top-4 right-4 p-2 rounded-full bg-zinc-800/50 text-zinc-400 hover:text-white hover:bg-zinc-700/50 transition-colors"
          >
            <.icon name="hero-x-mark" class="w-5 h-5" />
          </button>
        </div>
        
    <!-- Avatar image overlapping header -->
        <div class="flex justify-center -mt-16">
          <div class="w-32 h-32 rounded-full p-1 bg-gradient-to-b from-white/10 to-transparent border-4 border-zinc-900">
            <img
              src={get_avatar_image(@avatar)}
              alt={@avatar.name}
              class="w-full h-full rounded-full object-cover"
            />
          </div>
        </div>
        
    <!-- Content -->
        <div class="px-6 pb-6">
          <!-- Name and basic info -->
          <div class="text-center mt-4">
            <h2 class="text-2xl font-bold text-white">{@avatar.name}</h2>
            <p class="text-zinc-400 mt-1">{gender_label(@avatar.gender)} · {@avatar.age} anos</p>
          </div>
          
    <!-- Bio -->
          <p class="text-sm text-zinc-400 text-center mt-4 leading-relaxed">{@avatar.bio}</p>
          
    <!-- Current State -->
          <div class="mt-6 p-4 rounded-xl bg-zinc-800/40 border border-white/5">
            <h3 class="text-xs font-semibold text-zinc-500 uppercase mb-3">Estado Atual</h3>
            <div class="flex items-center justify-between mb-4">
              <.mood_indicator internal_state={@internal_state} />
              <.wellbeing_bar internal_state={@internal_state} class="w-32" />
            </div>
            <.needs_bars internal_state={@internal_state} />
          </div>
          
    <!-- Personality -->
          <div class="mt-4 p-4 rounded-xl bg-zinc-800/40 border border-white/5">
            <h3 class="text-xs font-semibold text-zinc-500 uppercase mb-3">Personalidade</h3>
            <.big_five_bars personality={@avatar.personality} />
            <div class="flex items-center gap-2 mt-3">
              <.temperament_badge personality={@avatar.personality} />
              <.enneagram_badge personality={@avatar.personality} />
            </div>
          </div>
          
    <!-- Relationships (if any) -->
          <div
            :if={@relationships != []}
            class="mt-4 p-4 rounded-xl bg-zinc-800/40 border border-white/5"
          >
            <h3 class="text-xs font-semibold text-zinc-500 uppercase mb-3">
              Relacionamentos ({length(@relationships)})
            </h3>
            <div class="space-y-2 max-h-32 overflow-y-auto">
              <div
                :for={rel <- Enum.take(@relationships, 5)}
                class="flex items-center justify-between text-sm"
              >
                <span class="text-zinc-300">
                  {get_other_avatar_name(rel, @avatar.id)}
                </span>
                <span class={["text-xs px-2 py-0.5 rounded-full", relationship_status_color(rel.status)]}>
                  {relationship_status_label(rel.status)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
  end

  defp get_other_avatar_name(rel, current_avatar_id) do
    cond do
      rel.avatar_a_id == current_avatar_id && rel.avatar_b -> rel.avatar_b.name
      rel.avatar_b_id == current_avatar_id && rel.avatar_a -> rel.avatar_a.name
      true -> "Avatar"
    end
  end

  defp relationship_status_label(:strangers), do: "Desconhecidos"
  defp relationship_status_label(:acquaintances), do: "Conhecidos"
  defp relationship_status_label(:friends), do: "Amigos"
  defp relationship_status_label(:close_friends), do: "Amigos próximos"
  defp relationship_status_label(:best_friends), do: "Melhores amigos"
  defp relationship_status_label(:crush), do: "Interesse"
  defp relationship_status_label(:mutual_crush), do: "Interesse mútuo"
  defp relationship_status_label(:dating), do: "Namorando"
  defp relationship_status_label(:partners), do: "Parceiros"
  defp relationship_status_label(:complicated), do: "Complicado"
  defp relationship_status_label(:ex), do: "Ex"
  defp relationship_status_label(_), do: "—"

  defp relationship_status_color(:strangers), do: "bg-zinc-700 text-zinc-400"
  defp relationship_status_color(:acquaintances), do: "bg-zinc-600 text-zinc-300"
  defp relationship_status_color(:friends), do: "bg-blue-500/20 text-blue-400"
  defp relationship_status_color(:close_friends), do: "bg-green-500/20 text-green-400"
  defp relationship_status_color(:best_friends), do: "bg-cyan-500/20 text-cyan-400"
  defp relationship_status_color(:crush), do: "bg-pink-500/20 text-pink-400"
  defp relationship_status_color(:mutual_crush), do: "bg-rose-500/20 text-rose-400"
  defp relationship_status_color(:dating), do: "bg-red-500/20 text-red-400"
  defp relationship_status_color(:partners), do: "bg-red-600/20 text-red-300"
  defp relationship_status_color(:complicated), do: "bg-amber-500/20 text-amber-400"
  defp relationship_status_color(:ex), do: "bg-zinc-500/20 text-zinc-400"
  defp relationship_status_color(_), do: "bg-zinc-700 text-zinc-400"

  @doc """
  Renders a complete avatar card with all traits and status.
  """
  attr :avatar, :map, required: true
  attr :selected, :boolean, default: false
  attr :on_click, :string, default: nil

  @spec avatar_card(map()) :: Phoenix.LiveView.Rendered.t()
  def avatar_card(assigns) do
    ~H"""
    <div
      class={[
        "relative group p-6 rounded-[2rem] bg-zinc-900/40 border border-white/5 backdrop-blur-md transition-all duration-500",
        "hover:bg-zinc-800/40 hover:border-white/10 hover:-translate-y-2 hover:shadow-2xl hover:shadow-black/50 cursor-pointer",
        if(@selected, do: "ring-1 ring-white/20 bg-zinc-800/60 shadow-xl", else: "")
      ]}
      phx-click={@on_click}
      phx-value-id={@avatar.id}
    >
      <div class="flex flex-col items-center text-center">
        <!-- Image Container with soft glow -->
        <div class="relative mb-5 group-hover:scale-105 transition-transform duration-500">
          <div class="absolute inset-0 bg-white/5 rounded-full blur-xl scale-110 opacity-0 group-hover:opacity-100 transition-opacity duration-700">
          </div>
          <div class="w-28 h-28 rounded-full p-1 bg-gradient-to-b from-white/10 to-transparent border border-white/5 relative z-10">
            <img
              src={get_avatar_image(@avatar)}
              alt={@avatar.name}
              class="w-full h-full rounded-full object-cover filter brightness-90 group-hover:brightness-105 transition-all duration-500"
            />
          </div>
          <.status_dot
            avatar={@avatar}
            class="absolute bottom-2 right-2 ring-4 ring-[#0c0c0e] !w-4 !h-4 z-20"
          />
        </div>

        <h3 class="text-xl font-semibold text-white tracking-tight mb-1 group-hover:text-white transition-colors">
          {@avatar.name}
        </h3>
        <p class="text-sm text-zinc-400 font-medium mb-4">
          {gender_label(@avatar.gender)} · {@avatar.age} anos
        </p>

        <p class="text-[13px] text-zinc-500 leading-relaxed line-clamp-2 px-2 group-hover:text-zinc-400 transition-colors">
          {@avatar.bio}
        </p>
        
    <!-- Minimalist Mood Pill -->
        <div class="mt-6 flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 border border-white/5 group-hover:bg-white/10 transition-colors">
          <.mood_indicator internal_state={@avatar.internal_state} />
        </div>
      </div>
    </div>
    """
  end

  @doc """
  Renders Big Five personality trait bars.
  """
  attr :personality, :map, required: true
  attr :class, :string, default: nil

  @spec big_five_bars(map()) :: Phoenix.LiveView.Rendered.t()
  def big_five_bars(assigns) do
    ~H"""
    <div class={["space-y-1.5", @class]}>
      <.trait_bar label="O" name="Abertura" value={@personality.openness} color="purple" />
      <.trait_bar label="C" name="Consciência" value={@personality.conscientiousness} color="blue" />
      <.trait_bar label="E" name="Extroversão" value={@personality.extraversion} color="orange" />
      <.trait_bar label="A" name="Amabilidade" value={@personality.agreeableness} color="green" />
      <.trait_bar label="N" name="Neuroticismo" value={@personality.neuroticism} color="red" />
    </div>
    """
  end

  attr :label, :string, required: true
  attr :name, :string, required: true
  attr :value, :float, required: true
  attr :color, :string, required: true

  defp trait_bar(assigns) do
    assigns = assign(assigns, :percentage, round(assigns.value * 100))

    ~H"""
    <div class="flex items-center gap-2">
      <span class={["w-5 text-xs font-bold", trait_color(@color)]} title={@name}>{@label}</span>
      <div class="flex-1 h-1.5 bg-base-300 rounded-full overflow-hidden">
        <div class={["h-full rounded-full", trait_bg(@color)]} style={"width: #{@percentage}%"}></div>
      </div>
      <span class="text-xs text-base-content/50 w-8">{@percentage}%</span>
    </div>
    """
  end

  @doc """
  Renders Enneagram type badge.
  """
  attr :personality, :map, required: true

  @spec enneagram_badge(map()) :: Phoenix.LiveView.Rendered.t()
  def enneagram_badge(assigns) do
    type_num = enneagram_number(assigns.personality.enneagram_type)
    type_name = enneagram_name(assigns.personality.enneagram_type)
    assigns = assign(assigns, type_num: type_num, type_name: type_name)

    ~H"""
    <div class={["badge gap-1", enneagram_color(@type_num)]}>
      <span class="font-bold">{@type_num}</span>
      <span class="text-xs">{@type_name}</span>
    </div>
    """
  end

  @doc """
  Renders mood indicator with emoji.
  """
  attr :internal_state, :map, required: true

  @spec mood_indicator(map()) :: Phoenix.LiveView.Rendered.t()
  def mood_indicator(assigns) do
    emotion = dominant_emotion(assigns.internal_state)
    assigns = assign(assigns, emotion: emotion)

    ~H"""
    <div class="flex items-center gap-2">
      <span class="text-sm filter saturate-50">{emotion_emoji(@emotion)}</span>
      <span class="text-xs font-medium text-zinc-300">{emotion_label(@emotion)}</span>
    </div>
    """
  end

  @doc """
  Renders wellbeing progress bar.
  """
  attr :internal_state, :map, required: true
  attr :class, :string, default: nil

  @spec wellbeing_bar(map()) :: Phoenix.LiveView.Rendered.t()
  def wellbeing_bar(assigns) do
    wellbeing = calculate_wellbeing(assigns.internal_state)
    assigns = assign(assigns, wellbeing: wellbeing, percentage: round(wellbeing * 100))

    ~H"""
    <div class={["space-y-1", @class]}>
      <div class="flex justify-between text-xs">
        <span class="text-base-content/60">Bem-estar</span>
        <span class={wellbeing_color(@wellbeing)}>{@percentage}%</span>
      </div>
      <div class="h-2 bg-base-300 rounded-full overflow-hidden">
        <div class={["h-full rounded-full", wellbeing_bg(@wellbeing)]} style={"width: #{@percentage}%"}>
        </div>
      </div>
    </div>
    """
  end

  @doc """
  Renders temperament badge.
  """
  attr :personality, :map, required: true

  @spec temperament_badge(map()) :: Phoenix.LiveView.Rendered.t()
  def temperament_badge(assigns) do
    temperament = Personality.temperament(assigns.personality)
    assigns = assign(assigns, temperament: temperament)

    ~H"""
    <span class={["badge badge-sm", temperament_color(@temperament)]}>
      {temperament_label(@temperament)}
    </span>
    """
  end

  @doc """
  Renders status indicator dot.
  """
  attr :avatar, :map, required: true
  attr :class, :string, default: nil

  @spec status_dot(map()) :: Phoenix.LiveView.Rendered.t()
  def status_dot(assigns) do
    mood = (assigns.avatar.internal_state && assigns.avatar.internal_state.mood) || 0
    is_active = assigns.avatar.is_active

    assigns = assign(assigns, mood: mood, is_active: is_active)

    ~H"""
    <span class={[
      "w-4 h-4 rounded-full border-2 border-zinc-900 shadow-sm",
      status_color(@is_active, @mood),
      @class
    ]}>
    </span>
    """
  end

  # Helper functions

  defp get_avatar_image(avatar) do
    if avatar.profile_image_url do
      avatar.profile_image_url
    else
      # Deterministic fallback based on ID
      fallbacks = [
        "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=256&q=80",
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=256&q=80",
        "https://images.unsplash.com/photo-1527980965255-d3b416303d12?auto=format&fit=crop&w=256&q=80",
        "https://images.unsplash.com/photo-1580489944761-15a19d654956?auto=format&fit=crop&w=256&q=80",
        "https://images.unsplash.com/photo-1633332755192-727a05c4013d?auto=format&fit=crop&w=256&q=80",
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=256&q=80"
      ]

      # Use the integer ID or hash the string ID to pick an index
      id_int =
        case avatar.id do
          id when is_integer(id) -> id
          id when is_binary(id) -> :erlang.phash2(id)
          _ -> 0
        end

      Enum.at(fallbacks, rem(id_int, length(fallbacks)))
    end
  end

  defp gender_label(:male), do: "Masculino"
  defp gender_label(:female), do: "Feminino"
  defp gender_label(:non_binary), do: "Não-binário"
  defp gender_label(:other), do: "Outro"
  defp gender_label(_), do: ""

  defp trait_color("purple"), do: "text-purple-500"
  defp trait_color("blue"), do: "text-blue-500"
  defp trait_color("orange"), do: "text-orange-500"
  defp trait_color("green"), do: "text-green-500"
  defp trait_color("red"), do: "text-red-500"
  defp trait_color(_), do: "text-base-content"

  defp trait_bg("purple"), do: "bg-purple-500"
  defp trait_bg("blue"), do: "bg-blue-500"
  defp trait_bg("orange"), do: "bg-orange-500"
  defp trait_bg("green"), do: "bg-green-500"
  defp trait_bg("red"), do: "bg-red-500"
  defp trait_bg(_), do: "bg-primary"

  defp enneagram_number(:type_1), do: 1
  defp enneagram_number(:type_2), do: 2
  defp enneagram_number(:type_3), do: 3
  defp enneagram_number(:type_4), do: 4
  defp enneagram_number(:type_5), do: 5
  defp enneagram_number(:type_6), do: 6
  defp enneagram_number(:type_7), do: 7
  defp enneagram_number(:type_8), do: 8
  defp enneagram_number(:type_9), do: 9
  defp enneagram_number(_), do: 9

  defp enneagram_name(:type_1), do: "Perfeccionista"
  defp enneagram_name(:type_2), do: "Ajudante"
  defp enneagram_name(:type_3), do: "Realizador"
  defp enneagram_name(:type_4), do: "Individualista"
  defp enneagram_name(:type_5), do: "Investigador"
  defp enneagram_name(:type_6), do: "Leal"
  defp enneagram_name(:type_7), do: "Entusiasta"
  defp enneagram_name(:type_8), do: "Desafiador"
  defp enneagram_name(:type_9), do: "Pacificador"
  defp enneagram_name(_), do: "Pacificador"

  defp enneagram_color(1), do: "badge-warning"
  defp enneagram_color(2), do: "badge-success"
  defp enneagram_color(3), do: "badge-info"
  defp enneagram_color(4), do: "badge-secondary"
  defp enneagram_color(5), do: "badge-accent"
  defp enneagram_color(6), do: "badge-primary"
  defp enneagram_color(7), do: "badge-warning"
  defp enneagram_color(8), do: "badge-error"
  defp enneagram_color(9), do: "badge-neutral"
  defp enneagram_color(_), do: "badge-ghost"

  defp dominant_emotion(%{emotions: emotions}) when is_map(emotions) do
    emotions
    |> Map.from_struct()
    |> Enum.max_by(fn {_, v} -> v end, fn -> {:neutral, 0} end)
    |> elem(0)
  end

  defp dominant_emotion(_), do: :neutral

  defp emotion_emoji(:joy), do: "😊"
  defp emotion_emoji(:sadness), do: "😢"
  defp emotion_emoji(:anger), do: "😠"
  defp emotion_emoji(:fear), do: "😰"
  defp emotion_emoji(:surprise), do: "😮"
  defp emotion_emoji(:disgust), do: "😒"
  defp emotion_emoji(:love), do: "🥰"
  defp emotion_emoji(:loneliness), do: "😔"
  defp emotion_emoji(:curiosity), do: "🤔"
  defp emotion_emoji(:excitement), do: "🤩"
  defp emotion_emoji(_), do: "😐"

  defp emotion_label(:joy), do: "Alegre"
  defp emotion_label(:sadness), do: "Triste"
  defp emotion_label(:anger), do: "Irritado"
  defp emotion_label(:fear), do: "Ansioso"
  defp emotion_label(:surprise), do: "Surpreso"
  defp emotion_label(:disgust), do: "Desconfortável"
  defp emotion_label(:love), do: "Apaixonado"
  defp emotion_label(:loneliness), do: "Solitário"
  defp emotion_label(:curiosity), do: "Curioso"
  defp emotion_label(:excitement), do: "Empolgado"
  defp emotion_label(_), do: "Neutro"

  defp calculate_wellbeing(%{energy: e, social: s, stimulation: st, comfort: c, mood: m})
       when is_number(e) and is_number(s) and is_number(st) and is_number(c) do
    needs_avg = (e + s + st + c) / 4 / 100
    mood_factor = (m + 1) / 2
    needs_avg * 0.7 + mood_factor * 0.3
  end

  defp calculate_wellbeing(_), do: 0.5

  defp wellbeing_color(w) when w >= 0.7, do: "text-success"
  defp wellbeing_color(w) when w >= 0.4, do: "text-warning"
  defp wellbeing_color(_), do: "text-error"

  defp wellbeing_bg(w) when w >= 0.7, do: "bg-success"
  defp wellbeing_bg(w) when w >= 0.4, do: "bg-warning"
  defp wellbeing_bg(_), do: "bg-error"

  defp temperament_label(:sanguine), do: "Sanguíneo"
  defp temperament_label(:choleric), do: "Colérico"
  defp temperament_label(:phlegmatic), do: "Fleumático"
  defp temperament_label(:melancholic), do: "Melancólico"
  defp temperament_label(_), do: ""

  defp temperament_color(:sanguine), do: "badge-warning"
  defp temperament_color(:choleric), do: "badge-error"
  defp temperament_color(:phlegmatic), do: "badge-info"
  defp temperament_color(:melancholic), do: "badge-primary"
  defp temperament_color(_), do: "badge-ghost"

  defp status_color(true, mood) when mood > 0.3, do: "bg-cyan-400 animate-pulse"
  defp status_color(true, mood) when mood < -0.3, do: "bg-rose-500"
  defp status_color(true, _), do: "bg-fuchsia-400"
  defp status_color(false, _), do: "bg-zinc-600"
end
