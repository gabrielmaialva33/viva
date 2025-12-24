defmodule VivaWeb.NimComponents do
  @moduledoc """
  Components for displaying NVIDIA NIM capabilities and demos.
  """
  use Phoenix.Component

  import VivaWeb.CoreComponents, only: [icon: 1]

  @doc """
  Renders the NVIDIA NIM capabilities grid.
  """
  attr :class, :string, default: nil

  @spec nim_grid(map()) :: Phoenix.LiveView.Rendered.t()
  def nim_grid(assigns) do
    ~H"""
    <div class={["grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-0", @class]}>
      <.nim_card
        icon="hero-chat-bubble-left-right"
        title="Linguagem"
        model="Cortex v4"
        description="Conversas complexas"
      />
      <.nim_card
        icon="hero-light-bulb"
        title="Raciocínio"
        model="Logic Core"
        description="Tomada de decisão"
      />
      <.nim_card
        icon="hero-photo"
        title="Visualização"
        model="Dream Engine"
        description="Geração de imagem"
      />
      <.nim_card
        icon="hero-eye"
        title="Percepção"
        model="Optic Nerve"
        description="Análise de vídeo"
      />
      <.nim_card icon="hero-cube" title="Spatial" model="Matrix 3D" description="Renderização 3D" />
      <.nim_card
        icon="hero-face-smile"
        title="Expressão"
        model="Face Sync"
        description="Animação facial"
      />
      <.nim_card
        icon="hero-speaker-wave"
        title="Voz"
        model="Sonic Synth"
        description="Síntese de fala"
      />
      <.nim_card
        icon="hero-microphone"
        title="Audição"
        model="Echo Sense"
        description="Reconhecimento de voz"
      />
      <.nim_card
        icon="hero-magnifying-glass"
        title="Memória"
        model="Mnemosyne"
        description="Busca semântica"
      />
      <.nim_card icon="hero-language" title="Tradução" model="Babel Core" description="Multilíngue" />
      <.nim_card
        icon="hero-shield-check"
        title="Segurança"
        model="Aegis"
        description="Barreira ética"
      />
    </div>
    """
  end

  attr :icon, :string, required: true
  attr :title, :string, required: true
  attr :model, :string, required: true
  attr :description, :string, required: true

  @spec nim_card(map()) :: Phoenix.LiveView.Rendered.t()
  def nim_card(assigns) do
    ~H"""
    <div class="group relative p-3 rounded-xl transition-all duration-300 hover:bg-white/5 cursor-default flex items-center gap-3 border border-transparent hover:border-white/5">
      <div class="w-8 h-8 rounded-lg bg-zinc-800/50 flex items-center justify-center text-zinc-500 group-hover:text-cyan-400 group-hover:bg-cyan-500/10 transition-colors">
        <.icon name={@icon} class="w-4 h-4" />
      </div>
      <div>
        <h3 class="font-medium text-xs text-zinc-300 group-hover:text-white leading-tight">{@title}</h3>
        <p class="text-[10px] text-zinc-500 uppercase tracking-wider">{@model}</p>
      </div>
    </div>
    """
  end

  attr :selected_demo, :atom, default: nil
  attr :loading, :boolean, default: false
  attr :result, :any, default: nil
  attr :avatar, :any, default: nil

  @spec demo_panel(map()) :: Phoenix.LiveView.Rendered.t()
  def demo_panel(assigns) do
    ~H"""
    <div class="w-full bg-black/20 backdrop-blur-xl border border-white/10 rounded-[2rem] overflow-hidden flex flex-col md:flex-row shadow-2xl">
      <!-- Controls Sidebar -->
      <div class="p-6 md:w-64 border-b md:border-b-0 md:border-r border-white/5 bg-white/5">
        <h2 class="text-xs font-semibold text-zinc-500 mb-6 uppercase tracking-widest pl-2">
          Controle Neural
        </h2>

        <div class="flex flex-col gap-2">
          <.demo_button
            demo={:chat}
            selected={@selected_demo}
            label="Conversa"
            icon="hero-chat-bubble-left-right"
          />
          <.demo_button
            demo={:thought}
            selected={@selected_demo}
            label="Pensamento"
            icon="hero-light-bulb"
          />
          <.demo_button demo={:image} selected={@selected_demo} label="Auto-Imagem" icon="hero-photo" />
          <.demo_button
            demo={:expression}
            selected={@selected_demo}
            label="Expressão"
            icon="hero-face-smile"
          />
          <.demo_button
            demo={:tts}
            selected={@selected_demo}
            label="Síntese de Voz"
            icon="hero-speaker-wave"
          />
          <.demo_button
            demo={:translate}
            selected={@selected_demo}
            label="Tradução"
            icon="hero-language"
          />
        </div>
      </div>
      
    <!-- Result Area -->
      <div class="flex-1 p-8 min-h-[400px] flex flex-col relative bg-[#050507]/20">
        <%= if @selected_demo do %>
          <h2 class="text-sm font-medium text-zinc-400 mb-6 flex items-center gap-2">
            <span class="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-synapse"></span>
            Output: {@selected_demo}
          </h2>
        <% end %>

        <div class="flex-1 relative">
          <%= if @loading do %>
            <div class="absolute inset-0 flex items-center justify-center">
              <div class="flex flex-col items-center gap-3">
                <span class="loading loading-spinner loading-md text-white/50"></span>
                <span class="text-xs text-zinc-500 tracking-wide font-medium">PROCESSANDO</span>
              </div>
            </div>
          <% else %>
            <%= if @result do %>
              <.demo_result demo={@selected_demo} result={@result} avatar={@avatar} />
            <% else %>
              <div class="h-full flex flex-col items-center justify-center text-zinc-600 opacity-60">
                <.icon name="hero-command-line" class="w-12 h-12 mb-4 text-white/10" />
                <p class="text-sm font-medium">Aguardando interação</p>
                <p class="text-xs mt-2 text-zinc-600">Selecione uma função no menu lateral</p>
              </div>
            <% end %>
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  @doc """
  Renders the stats bar (placeholder).
  """
  @spec stats_bar(map()) :: Phoenix.LiveView.Rendered.t()
  def stats_bar(assigns) do
    ~H"""
    <div class="hidden"></div>
    """
  end

  attr :demo, :atom, required: true
  attr :selected, :atom, default: nil
  attr :label, :string, required: true
  attr :icon, :string, required: true

  defp demo_button(assigns) do
    ~H"""
    <button
      phx-click="select_demo"
      phx-value-demo={@demo}
      class={[
        "w-full flex items-center gap-3 px-4 py-3 rounded-xl text-xs font-medium transition-all duration-300 transform",
        if(@demo == @selected,
          do: "bg-white text-black shadow-lg scale-105",
          else: "text-zinc-400 hover:bg-white/5 hover:text-white"
        )
      ]}
    >
      <.icon name={@icon} class="w-4 h-4" />
      {@label}
    </button>
    """
  end

  attr :demo, :atom, required: true
  attr :result, :any, required: true
  attr :avatar, :any, default: nil

  defp demo_result(assigns) do
    ~H"""
    <div class="text-zinc-300 text-sm leading-relaxed animate-fade-in font-sans">
      <div class="prose prose-invert max-w-none prose-p:text-zinc-300 prose-headings:text-white">
        <%= case @result do %>
          <% {:error, _reason} -> %>
            <div class="flex gap-4 items-start">
              <div class="shrink-0 w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center text-amber-500">
                <.icon name="hero-beaker" class="w-5 h-5" />
              </div>
              <div>
                <span class="text-amber-400 font-semibold block mb-1">Em Desenvolvimento</span>
                <p class="bg-amber-500/10 p-4 rounded-xl text-amber-200 text-sm">
                  Este recurso requer APIs adicionais (Imagem, TTS, Tradução)<br />
                  que serão integradas em breve.
                </p>
                <p class="text-zinc-500 text-xs mt-2">
                  <span class="text-cyan-400">✓ Funcionando:</span> Conversa, Pensamento
                </p>
              </div>
            </div>
          <% _ -> %>
            <%= case @demo do %>
              <% :chat -> %>
                <div class="flex gap-4 items-start">
                  <div class="shrink-0 w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400">
                    <.icon name="hero-chat-bubble-left-right" class="w-4 h-4" />
                  </div>
                  <div>
                    <span class="text-white font-semibold block mb-1">{@avatar.name}</span>
                    <p class="bg-white/5 p-4 rounded-r-2xl rounded-bl-2xl text-zinc-200">{@result}</p>
                  </div>
                </div>
              <% :thought -> %>
                <div class="bg-zinc-900/50 p-6 rounded-2xl border border-white/5">
                  <p class="text-[10px] text-zinc-500 uppercase tracking-wider mb-3">
                    Processo Cognitivo
                  </p>
                  <p class="text-zinc-400 italic">{@result}</p>
                </div>
              <% :image -> %>
                <img
                  src={@result}
                  class="rounded-2xl border border-white/10 w-full max-w-md shadow-2xl"
                />
              <% :expression -> %>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <%= for {exp, url} <- @result do %>
                    <div class="text-center group cursor-default">
                      <img
                        src={url}
                        class="w-full aspect-square object-cover rounded-xl border border-white/10 opacity-70 group-hover:opacity-100 transition-all duration-300 shadow-lg"
                      />
                      <span class="text-[10px] text-zinc-500 uppercase mt-2 block group-hover:text-white transition-colors">
                        {exp}
                      </span>
                    </div>
                  <% end %>
                </div>
              <% :tts -> %>
                <div class="bg-white/5 p-6 rounded-2xl border border-white/10 flex items-center gap-4">
                  <div class="w-10 h-10 rounded-full bg-white flex items-center justify-center text-black">
                    <.icon name="hero-play" class="w-4 h-4 ml-0.5" />
                  </div>
                  <div class="flex-1">
                    <audio
                      controls
                      src={@result}
                      class="w-full h-8 opacity-80 hover:opacity-100 transition-opacity"
                    />
                  </div>
                </div>
              <% :translate -> %>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div class="p-6 rounded-2xl bg-white/5 border border-white/5">
                    <span class="text-[10px] text-zinc-500 uppercase block mb-3">Input (PT-BR)</span>
                    <p class="text-white text-lg font-medium">{@result.original}</p>
                  </div>
                  <div class="p-6 rounded-2xl bg-cyan-500/10 border border-cyan-500/20">
                    <span class="text-[10px] text-cyan-400 uppercase block mb-3">
                      Output ({@result.target_lang})
                    </span>
                    <p class="text-cyan-300 text-lg font-medium">{@result.translated}</p>
                  </div>
                </div>
              <% _ -> %>
                <pre class="bg-black/30 p-4 rounded-xl text-xs font-mono">{inspect(@result, pretty: true)}</pre>
            <% end %>
        <% end %>
      </div>
    </div>
    """
  end
end
