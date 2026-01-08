defmodule VivaWeb.MetricsLive do
  @moduledoc """
  Dashboard cyberpunk das métricas de simulação do VIVA.
  """

  use VivaWeb, :live_view

  alias Phoenix.PubSub
  alias Viva.Metrics.Collector

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      PubSub.subscribe(Viva.PubSub, "metrics:live")
      send(self(), :load_metrics)
      :timer.send_interval(2_000, :refresh)
    end

    socket =
      socket
      |> assign(:page_title, "VIVA // Simulação")
      |> assign(:avatars, %{})
      |> assign(:global, %{
        total_ticks: 0,
        total_thoughts: 0,
        total_dreams: 0,
        total_crystallizations: 0,
        avg_happiness: 0.0,
        avg_energy: 0.0
      })
      |> assign(:selected_avatar, nil)

    {:ok, socket}
  end

  @impl true
  def handle_info(:load_metrics, socket) do
    case Collector.get_all() do
      %{avatars: avatars, global: global} ->
        {:noreply, assign(socket, avatars: avatars, global: global)}
      _ ->
        {:noreply, socket}
    end
  end

  @impl true
  def handle_info(:refresh, socket) do
    send(self(), :load_metrics)
    {:noreply, socket}
  end

  @impl true
  def handle_info({:metrics_update, avatars, global}, socket) do
    {:noreply, assign(socket, avatars: avatars, global: global)}
  end

  @impl true
  def handle_event("select_avatar", %{"id" => avatar_id}, socket) do
    {:noreply, assign(socket, :selected_avatar, avatar_id)}
  end

  @impl true
  def handle_event("close_detail", _, socket) do
    {:noreply, assign(socket, :selected_avatar, nil)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="min-h-screen bg-[#0a0a0f] text-white font-mono">
      <!-- Scanlines overlay -->
      <div class="fixed inset-0 pointer-events-none bg-[repeating-linear-gradient(0deg,transparent,transparent_2px,rgba(0,0,0,0.1)_2px,rgba(0,0,0,0.1)_4px)] z-50"></div>

      <!-- Header -->
      <header class="border-b border-cyan-500/30 bg-gradient-to-r from-cyan-950/50 via-transparent to-purple-950/50">
        <div class="max-w-7xl mx-auto px-4 py-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-4">
              <div class="text-2xl font-bold tracking-wider">
                <span class="text-cyan-400">VIVA</span>
                <span class="text-white/50">//</span>
                <span class="text-purple-400">SIMULAÇÃO</span>
              </div>
              <div class="flex items-center gap-2 px-3 py-1 bg-green-500/20 border border-green-500/50 rounded">
                <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span class="text-green-400 text-xs uppercase tracking-widest">Conectado</span>
              </div>
            </div>
            <div class="text-xs text-white/40">
              <%= DateTime.utc_now() |> Calendar.strftime("%H:%M:%S UTC") %>
            </div>
          </div>
        </div>
      </header>

      <!-- Stats Grid -->
      <div class="max-w-7xl mx-auto px-4 py-6">
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <.stat_card value={map_size(@avatars)} label="ENTIDADES" color="cyan" icon="◉" />
          <.stat_card value={@global.total_ticks} label="CICLOS" color="blue" icon="⟳" />
          <.stat_card value={@global.total_thoughts} label="PENSAMENTOS" color="purple" icon="◈" />
          <.stat_card value={@global.total_crystallizations} label="CRISTAIS" color="pink" icon="◇" />
          <.stat_card value={format_percent(@global.avg_happiness)} label="PRAZER" color="green" icon="▲" />
          <.stat_card value={format_percent(@global.avg_energy)} label="ENERGIA" color="yellow" icon="⚡" />
        </div>
      </div>

      <!-- Avatar Grid -->
      <div class="max-w-7xl mx-auto px-4 pb-8">
        <%= if map_size(@avatars) == 0 do %>
          <div class="flex flex-col items-center justify-center py-20 text-center">
            <div class="text-6xl mb-4 animate-pulse">◎</div>
            <div class="text-xl text-white/60 mb-2">AGUARDANDO ENTIDADES</div>
            <div class="text-sm text-white/30">Inicie processos de vida para monitorar</div>
          </div>
        <% else %>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <%= for {_id, avatar} <- @avatars do %>
              <.avatar_card avatar={avatar} />
            <% end %>
          </div>
        <% end %>
      </div>

      <!-- Modal de Detalhes -->
      <%= if @selected_avatar && @avatars[@selected_avatar] do %>
        <.avatar_modal avatar={@avatars[@selected_avatar]} />
      <% end %>
    </div>
    """
  end

  # === Componentes ===

  attr :value, :any, required: true
  attr :label, :string, required: true
  attr :color, :string, default: "cyan"
  attr :icon, :string, default: "◉"

  defp stat_card(assigns) do
    color_classes = %{
      "cyan" => "border-cyan-500/50 text-cyan-400",
      "blue" => "border-blue-500/50 text-blue-400",
      "purple" => "border-purple-500/50 text-purple-400",
      "pink" => "border-pink-500/50 text-pink-400",
      "green" => "border-green-500/50 text-green-400",
      "yellow" => "border-yellow-500/50 text-yellow-400"
    }
    assigns = assign(assigns, :color_class, color_classes[assigns.color] || color_classes["cyan"])

    ~H"""
    <div class={"bg-black/50 border rounded-lg p-4 #{@color_class}"}>
      <div class="flex items-center gap-2 mb-1">
        <span class="text-lg opacity-60"><%= @icon %></span>
        <span class="text-[10px] uppercase tracking-widest opacity-60"><%= @label %></span>
      </div>
      <div class="text-2xl font-bold"><%= @value %></div>
    </div>
    """
  end

  attr :avatar, :map, required: true

  defp avatar_card(assigns) do
    ~H"""
    <div
      class="group relative bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-700/50 rounded-xl overflow-hidden cursor-pointer hover:border-cyan-500/50 transition-all duration-300 hover:shadow-[0_0_30px_rgba(0,255,255,0.1)]"
      phx-click="select_avatar"
      phx-value-id={@avatar.avatar_id}
    >
      <!-- Glow effect on hover -->
      <div class="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>

      <!-- Header -->
      <div class="relative p-4 border-b border-slate-700/50">
        <div class="flex items-center gap-3">
          <div class={"w-12 h-12 rounded-lg flex items-center justify-center text-2xl #{emotion_gradient(@avatar.current_emotion)}"}>
            <%= emotion_emoji(@avatar.current_emotion) %>
          </div>
          <div class="flex-1 min-w-0">
            <div class="font-bold text-white truncate"><%= @avatar.name %></div>
            <div class="text-xs text-slate-500 flex items-center gap-2">
              <span class={"w-1.5 h-1.5 rounded-full #{if @avatar.owner_online?, do: "bg-green-400", else: "bg-slate-600"}"}></span>
              <span><%= emotion_label(@avatar.current_emotion) %></span>
            </div>
          </div>
          <div class="text-right">
            <div class="text-[10px] text-slate-500 uppercase">Ciclo</div>
            <div class="text-sm font-mono text-cyan-400">#<%= @avatar.tick_count %></div>
          </div>
        </div>
      </div>

      <!-- Body -->
      <div class="p-4 space-y-4">
        <!-- Neuro bars -->
        <div class="space-y-2">
          <.cyber_bar label="DPA" sublabel="Dopamina" value={@avatar.dopamine} color="yellow" />
          <.cyber_bar label="OXI" sublabel="Ocitocina" value={@avatar.oxytocin} color="pink" />
          <.cyber_bar label="CRT" sublabel="Cortisol" value={@avatar.cortisol} color="red" />
          <.cyber_bar label="NRG" sublabel="Energia" value={@avatar.energy} color="cyan" />
        </div>

        <!-- PAD Vector -->
        <div class="grid grid-cols-3 gap-2 pt-2 border-t border-slate-700/50">
          <.pad_display label="P" sublabel="Prazer" value={@avatar.pleasure} />
          <.pad_display label="A" sublabel="Ativação" value={@avatar.arousal} />
          <.pad_display label="D" sublabel="Domínio" value={@avatar.dominance} />
        </div>

        <!-- Last thought -->
        <%= if @avatar.last_thought do %>
          <div class="pt-2 border-t border-slate-700/50">
            <div class="text-[10px] text-slate-500 uppercase mb-1">Último pensamento</div>
            <div class="text-xs text-slate-300 italic line-clamp-2">
              "<%= String.slice(@avatar.last_thought || "", 0..80) %>"
            </div>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  attr :label, :string, required: true
  attr :sublabel, :string, required: true
  attr :value, :float, required: true
  attr :color, :string, default: "cyan"

  defp cyber_bar(assigns) do
    bar_colors = %{
      "cyan" => "bg-cyan-500",
      "yellow" => "bg-yellow-500",
      "pink" => "bg-pink-500",
      "red" => "bg-red-500",
      "green" => "bg-green-500",
      "purple" => "bg-purple-500"
    }
    glow_colors = %{
      "cyan" => "shadow-cyan-500/50",
      "yellow" => "shadow-yellow-500/50",
      "pink" => "shadow-pink-500/50",
      "red" => "shadow-red-500/50",
      "green" => "shadow-green-500/50",
      "purple" => "shadow-purple-500/50"
    }
    assigns = assign(assigns, :bar_color, bar_colors[assigns.color] || bar_colors["cyan"])
    assigns = assign(assigns, :glow_color, glow_colors[assigns.color] || glow_colors["cyan"])

    ~H"""
    <div class="flex items-center gap-2">
      <div class="w-8 text-[10px] font-bold text-slate-500" title={@sublabel}><%= @label %></div>
      <div class="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          class={"h-full rounded-full transition-all duration-500 shadow-lg #{@bar_color} #{@glow_color}"}
          style={"width: #{round(@value * 100)}%"}
        ></div>
      </div>
      <div class="w-8 text-right text-[10px] font-mono text-slate-400"><%= round(@value * 100) %></div>
    </div>
    """
  end

  attr :label, :string, required: true
  attr :sublabel, :string, required: true
  attr :value, :float, required: true

  defp pad_display(assigns) do
    {text_color, bg_color} = cond do
      assigns.value > 0.3 -> {"text-green-400", "bg-green-500/20"}
      assigns.value < -0.3 -> {"text-red-400", "bg-red-500/20"}
      true -> {"text-slate-400", "bg-slate-700/50"}
    end
    assigns = assign(assigns, :text_color, text_color)
    assigns = assign(assigns, :bg_color, bg_color)

    ~H"""
    <div class={"rounded-lg p-2 text-center #{@bg_color}"}>
      <div class="text-[10px] text-slate-500 uppercase"><%= @sublabel %></div>
      <div class={"text-lg font-bold font-mono #{@text_color}"}>
        <%= :erlang.float_to_binary(@value, decimals: 2) %>
      </div>
    </div>
    """
  end

  attr :avatar, :map, required: true

  defp avatar_modal(assigns) do
    ~H"""
    <div class="fixed inset-0 z-50 flex items-center justify-center p-4" phx-click="close_detail">
      <!-- Backdrop -->
      <div class="absolute inset-0 bg-black/80 backdrop-blur-sm"></div>

      <!-- Modal -->
      <div class="relative w-full max-w-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-cyan-500/30 rounded-2xl shadow-2xl shadow-cyan-500/10" phx-click="close_detail" phx-value-stop="true">
        <!-- Header -->
        <div class="p-6 border-b border-slate-700/50">
          <div class="flex items-center gap-4">
            <div class={"w-16 h-16 rounded-xl flex items-center justify-center text-4xl #{emotion_gradient(@avatar.current_emotion)}"}>
              <%= emotion_emoji(@avatar.current_emotion) %>
            </div>
            <div class="flex-1">
              <h2 class="text-2xl font-bold text-white"><%= @avatar.name %></h2>
              <div class="text-sm text-slate-400"><%= emotion_label(@avatar.current_emotion) %> · Ciclo #<%= @avatar.tick_count %></div>
            </div>
            <button class="w-10 h-10 rounded-lg bg-slate-800 hover:bg-slate-700 flex items-center justify-center text-slate-400 hover:text-white transition-colors" phx-click="close_detail">
              ✕
            </button>
          </div>
        </div>

        <!-- Content -->
        <div class="p-6 grid grid-cols-2 gap-6">
          <!-- Neuroquímica -->
          <div>
            <h3 class="text-xs uppercase tracking-widest text-cyan-400 mb-4">◈ Neuroquímica</h3>
            <div class="space-y-3">
              <.detail_bar label="Dopamina" desc="Recompensa" value={@avatar.dopamine} color="yellow" />
              <.detail_bar label="Cortisol" desc="Estresse" value={@avatar.cortisol} color="red" />
              <.detail_bar label="Ocitocina" desc="Conexão" value={@avatar.oxytocin} color="pink" />
              <.detail_bar label="Adenosina" desc="Fadiga" value={@avatar.adenosine} color="purple" />
              <.detail_bar label="Libido" desc="Desejo" value={@avatar.libido} color="pink" />
            </div>
          </div>

          <!-- Consciência -->
          <div>
            <h3 class="text-xs uppercase tracking-widest text-purple-400 mb-4">◉ Consciência</h3>
            <div class="space-y-3">
              <.detail_bar label="Presença" desc="Aterramento" value={@avatar.presence_level} color="cyan" />
              <.detail_bar label="Meta-Consciencia" desc="Reflexao" value={@avatar.meta_awareness} color="purple" />
              <.detail_bar label="Intensidade" desc="Vivacidade" value={@avatar.experience_intensity} color="green" />
              <.detail_bar label="Energia" desc="Vitalidade" value={@avatar.energy} color="cyan" />
            </div>
          </div>
        </div>

        <!-- PAD Section -->
        <div class="px-6 pb-6">
          <h3 class="text-xs uppercase tracking-widest text-green-400 mb-4">▲ Vetor Emocional (PAD)</h3>
          <div class="grid grid-cols-3 gap-4">
            <div class="bg-slate-800/50 rounded-xl p-4 text-center">
              <div class="text-[10px] uppercase text-slate-500 mb-1">Prazer</div>
              <div class={"text-3xl font-bold font-mono #{pad_text_color(@avatar.pleasure)}"}>
                <%= :erlang.float_to_binary(@avatar.pleasure, decimals: 2) %>
              </div>
            </div>
            <div class="bg-slate-800/50 rounded-xl p-4 text-center">
              <div class="text-[10px] uppercase text-slate-500 mb-1">Ativação</div>
              <div class={"text-3xl font-bold font-mono #{pad_text_color(@avatar.arousal - 0.5)}"}>
                <%= :erlang.float_to_binary(@avatar.arousal, decimals: 2) %>
              </div>
            </div>
            <div class="bg-slate-800/50 rounded-xl p-4 text-center">
              <div class="text-[10px] uppercase text-slate-500 mb-1">Domínio</div>
              <div class={"text-3xl font-bold font-mono #{pad_text_color(@avatar.dominance - 0.5)}"}>
                <%= :erlang.float_to_binary(@avatar.dominance, decimals: 2) %>
              </div>
            </div>
          </div>
        </div>

        <!-- Estado Atual -->
        <div class="px-6 pb-6">
          <div class="grid grid-cols-4 gap-3 text-center">
            <div class="bg-slate-800/30 rounded-lg p-3">
              <div class="text-[10px] text-slate-500 uppercase">Atividade</div>
              <div class="text-sm text-white"><%= activity_label(@avatar.current_activity) %></div>
            </div>
            <div class="bg-slate-800/30 rounded-lg p-3">
              <div class="text-[10px] text-slate-500 uppercase">Desejo</div>
              <div class="text-sm text-white"><%= desire_label(@avatar.current_desire) %></div>
            </div>
            <div class="bg-slate-800/30 rounded-lg p-3">
              <div class="text-[10px] text-slate-500 uppercase">Ritmo</div>
              <div class="text-sm text-white"><%= tempo_label(@avatar.stream_tempo) %></div>
            </div>
            <div class="bg-slate-800/30 rounded-lg p-3">
              <div class="text-[10px] text-slate-500 uppercase">Dono</div>
              <div class={"text-sm #{if @avatar.owner_online?, do: "text-green-400", else: "text-slate-500"}"}>
                <%= if @avatar.owner_online?, do: "Conectado", else: "Desconectado" %>
              </div>
            </div>
          </div>
        </div>

        <!-- Último Pensamento -->
        <%= if @avatar.last_thought do %>
          <div class="px-6 pb-6">
            <div class="bg-slate-800/30 rounded-xl p-4 border border-slate-700/50">
              <div class="text-[10px] text-slate-500 uppercase mb-2">◈ Último Pensamento</div>
              <div class="text-slate-300 italic">"<%= @avatar.last_thought %>"</div>
            </div>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  attr :label, :string, required: true
  attr :desc, :string, required: true
  attr :value, :float, required: true
  attr :color, :string, default: "cyan"

  defp detail_bar(assigns) do
    bar_colors = %{
      "cyan" => "bg-gradient-to-r from-cyan-500 to-cyan-400",
      "yellow" => "bg-gradient-to-r from-yellow-500 to-yellow-400",
      "pink" => "bg-gradient-to-r from-pink-500 to-pink-400",
      "red" => "bg-gradient-to-r from-red-500 to-red-400",
      "green" => "bg-gradient-to-r from-green-500 to-green-400",
      "purple" => "bg-gradient-to-r from-purple-500 to-purple-400"
    }
    assigns = assign(assigns, :bar_color, bar_colors[assigns.color] || bar_colors["cyan"])

    ~H"""
    <div>
      <div class="flex justify-between text-xs mb-1">
        <span class="text-slate-300"><%= @label %></span>
        <span class="text-slate-500"><%= @desc %></span>
      </div>
      <div class="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div class={"h-full rounded-full #{@bar_color}"} style={"width: #{round(@value * 100)}%"}></div>
      </div>
      <div class="text-right text-[10px] text-slate-500 mt-0.5"><%= Float.round(@value * 100, 1) %>%</div>
    </div>
    """
  end

  # === Funções Helper ===

  defp emotion_emoji(emotion) do
    case emotion do
      :passionate -> "🔥"
      :vigilant -> "👁"
      :content -> "☺"
      :melancholic -> "🌧"
      :serene -> "🌊"
      :aggressive -> "⚡"
      :focused -> "◎"
      :wise -> "🦉"
      :playful -> "✦"
      :idealistic -> "✧"
      :nurturing -> "♡"
      :driven -> "▲"
      :critical -> "◇"
      :happy -> "☀"
      :excited -> "★"
      :calm -> "○"
      :sad -> "●"
      :anxious -> "◈"
      :angry -> "✕"
      :bored -> "—"
      :curious -> "?"
      :loving -> "♥"
      :neutral -> "◦"
      # Estados negativos severos
      :suffering -> "☠"
      :depressed -> "⊘"
      :defeated -> "⊗"
      :distressed -> "◉"
      :numb -> "◯"
      :exhausted -> "⊛"
      :tense -> "◆"
      :relaxed -> "◇"
      _ -> "◌"
    end
  end

  defp emotion_label(emotion) do
    case emotion do
      :passionate -> "Apaixonado"
      :vigilant -> "Vigilante"
      :content -> "Contente"
      :melancholic -> "Melancólico"
      :serene -> "Sereno"
      :aggressive -> "Agressivo"
      :focused -> "Focado"
      :wise -> "Sábio"
      :playful -> "Brincalhão"
      :idealistic -> "Idealista"
      :nurturing -> "Acolhedor"
      :driven -> "Determinado"
      :critical -> "Crítico"
      :happy -> "Feliz"
      :excited -> "Empolgado"
      :calm -> "Calmo"
      :sad -> "Triste"
      :anxious -> "Ansioso"
      :angry -> "Irritado"
      :bored -> "Entediado"
      :curious -> "Curioso"
      :loving -> "Amoroso"
      :neutral -> "Neutro"
      # Estados negativos severos
      :suffering -> "Sofrendo"
      :depressed -> "Deprimido"
      :defeated -> "Derrotado"
      :distressed -> "Angustiado"
      :numb -> "Entorpecido"
      :exhausted -> "Exausto"
      :tense -> "Tenso"
      :relaxed -> "Relaxado"
      _ -> "Indefinido"
    end
  end

  defp emotion_gradient(emotion) do
    case emotion do
      :passionate -> "bg-gradient-to-br from-red-500/30 to-orange-500/30 border border-red-500/50"
      :vigilant -> "bg-gradient-to-br from-amber-500/30 to-yellow-500/30 border border-amber-500/50"
      :content -> "bg-gradient-to-br from-green-500/30 to-emerald-500/30 border border-green-500/50"
      :melancholic -> "bg-gradient-to-br from-blue-500/30 to-indigo-500/30 border border-blue-500/50"
      :serene -> "bg-gradient-to-br from-cyan-500/30 to-teal-500/30 border border-cyan-500/50"
      :aggressive -> "bg-gradient-to-br from-red-600/30 to-red-500/30 border border-red-600/50"
      :focused -> "bg-gradient-to-br from-violet-500/30 to-purple-500/30 border border-violet-500/50"
      :wise -> "bg-gradient-to-br from-indigo-500/30 to-blue-500/30 border border-indigo-500/50"
      :playful -> "bg-gradient-to-br from-pink-500/30 to-rose-500/30 border border-pink-500/50"
      :idealistic -> "bg-gradient-to-br from-purple-500/30 to-pink-500/30 border border-purple-500/50"
      :nurturing -> "bg-gradient-to-br from-rose-500/30 to-pink-500/30 border border-rose-500/50"
      :driven -> "bg-gradient-to-br from-orange-500/30 to-amber-500/30 border border-orange-500/50"
      :critical -> "bg-gradient-to-br from-slate-500/30 to-gray-500/30 border border-slate-500/50"
      # Estados negativos severos - tons escuros/vermelhos
      :suffering -> "bg-gradient-to-br from-red-900/40 to-rose-900/40 border border-red-800/60"
      :depressed -> "bg-gradient-to-br from-gray-800/40 to-slate-900/40 border border-gray-700/60"
      :defeated -> "bg-gradient-to-br from-stone-800/40 to-gray-800/40 border border-stone-700/60"
      :distressed -> "bg-gradient-to-br from-orange-800/40 to-red-800/40 border border-orange-700/60"
      :numb -> "bg-gradient-to-br from-slate-700/40 to-slate-800/40 border border-slate-600/60"
      :exhausted -> "bg-gradient-to-br from-zinc-800/40 to-neutral-800/40 border border-zinc-700/60"
      :tense -> "bg-gradient-to-br from-amber-800/40 to-orange-800/40 border border-amber-700/60"
      :relaxed -> "bg-gradient-to-br from-teal-500/30 to-cyan-500/30 border border-teal-500/50"
      _ -> "bg-gradient-to-br from-slate-600/30 to-slate-700/30 border border-slate-600/50"
    end
  end

  defp activity_label(activity) do
    case activity do
      :idle -> "Ocioso"
      :sleeping -> "Dormindo"
      :talking -> "Conversando"
      :thinking -> "Pensando"
      :exploring -> "Explorando"
      :reflecting -> "Refletindo"
      :dreaming -> "Sonhando"
      :resting -> "Descansando"
      :waiting -> "Esperando"
      :excited -> "Animado"
      _ -> to_string(activity)
    end
  end

  defp desire_label(desire) do
    case desire do
      :wants_to_talk -> "Conversar"
      :wants_to_see_crush -> "Ver crush"
      :wants_something_new -> "Novidade"
      :wants_rest -> "Descansar"
      :wants_attention -> "Atenção"
      :wants_to_express -> "Expressar"
      :none -> "Nenhum"
      nil -> "Nenhum"
      _ -> to_string(desire)
    end
  end

  defp tempo_label(tempo) do
    case tempo do
      :frozen -> "Congelado"
      :slow -> "Lento"
      :normal -> "Normal"
      :fast -> "Rápido"
      :racing -> "Acelerado"
      _ -> "Normal"
    end
  end

  defp pad_text_color(value) when value > 0.3, do: "text-green-400"
  defp pad_text_color(value) when value < -0.3, do: "text-red-400"
  defp pad_text_color(_), do: "text-slate-300"

  defp format_percent(value) when is_float(value), do: "#{round(value * 100)}%"
  defp format_percent(_), do: "0%"
end
