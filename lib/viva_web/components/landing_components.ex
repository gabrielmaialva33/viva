defmodule VivaWeb.LandingComponents do
  @moduledoc """
  Components for the VIVA landing page.
  Marketing-focused components with real-time demo integration.
  """
  use Phoenix.Component

  import VivaWeb.CoreComponents, only: [icon: 1]

  # ============================================================================
  # HERO SECTION
  # ============================================================================

  @doc """
  Renders the hero section with parallax orbs, word rotation, and social proof.
  Inspired by Metronic SaaS landing.
  """
  attr :class, :string, default: nil
  attr :active_avatars, :integer, default: 0
  attr :demo_avatars, :list, default: []

  @spec hero_section(map()) :: Phoenix.LiveView.Rendered.t()
  def hero_section(assigns) do
    ~H"""
    <section
      id="hero"
      phx-hook="ParallaxOrbs"
      class={[
        "relative min-h-[90vh] flex flex-col items-center justify-center px-6 text-center overflow-hidden",
        @class
      ]}
    >
      <!-- Parallax Orbs Background -->
      <.parallax_orbs />
      
    <!-- Content -->
      <div class="relative z-10 max-w-4xl mx-auto">
        <!-- Neural Badge -->
        <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 backdrop-blur-md mb-8 animate-fade-in">
          <span class="w-2 h-2 rounded-full bg-cyan-400 animate-synapse"></span>
          <span class="text-xs uppercase tracking-widest font-medium text-cyan-300">
            Rede Neural Ativa 24/7
          </span>
        </div>
        
    <!-- Headline with Word Rotate -->
        <h1
          class="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white mb-6 animate-fade-in"
          style="animation-delay: 0.1s;"
        >
          Avatares digitais vivendo <br class="hidden sm:block" />
          <.word_rotate
            words={["pensamentos", "emoções", "conexões", "memórias"]}
            class="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-fuchsia-500 to-cyan-400"
          />
          <span class="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-fuchsia-500 to-cyan-400">
            autônomos
          </span>
        </h1>
        
    <!-- Subheadline -->
        <p
          class="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in"
          style="animation-delay: 0.2s;"
        >
          Crie avatares com personalidade única que pensam, sentem e se conectam.
          Observe vidas digitais evoluindo naturalmente 24 horas por dia.
        </p>
        
    <!-- CTA Buttons -->
        <div
          class="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in mb-12"
          style="animation-delay: 0.3s;"
        >
          <a
            href="#demo"
            class="group px-8 py-4 rounded-full bg-gradient-to-r from-cyan-500 to-fuchsia-500 text-white font-semibold text-sm tracking-wide hover:from-cyan-400 hover:to-fuchsia-400 transition-all duration-300 hover:scale-105 shadow-lg shadow-cyan-500/25 hover:shadow-xl hover:shadow-fuchsia-500/25 flex items-center gap-2"
          >
            Ver Demo ao Vivo
            <.icon
              name="hero-arrow-down"
              class="w-4 h-4 group-hover:translate-y-1 transition-transform"
            />
          </a>
          <a
            href="#features"
            class="px-8 py-4 rounded-full border border-cyan-500/30 text-white font-medium text-sm tracking-wide hover:bg-cyan-500/10 hover:border-cyan-400/50 transition-all duration-300"
          >
            Conhecer o Projeto
          </a>
        </div>
        
    <!-- Social Proof: Avatar Stack -->
        <div class="animate-fade-in" style="animation-delay: 0.4s;">
          <.avatar_stack avatars={@demo_avatars} total_count={@active_avatars} />
        </div>
      </div>
      
    <!-- Scroll indicator -->
      <div class="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <.icon name="hero-chevron-down" class="w-6 h-6 text-zinc-600" />
      </div>
    </section>
    """
  end

  @doc """
  Renders parallax floating orbs that follow the mouse cursor.
  Uses ParallaxOrbs JS hook.
  """
  def parallax_orbs(assigns) do
    assigns = assign_new(assigns, :class, fn -> nil end)

    ~H"""
    <div class={["absolute inset-0 overflow-hidden pointer-events-none", @class]}>
      <!-- Large cyan orb - top left -->
      <div
        data-parallax-orb
        data-parallax-speed="30"
        class="absolute top-[-15%] left-[5%] w-[500px] h-[500px] bg-cyan-500/20 rounded-full blur-[120px] animate-pulse-slow"
      >
      </div>
      <!-- Large fuchsia orb - bottom right -->
      <div
        data-parallax-orb
        data-parallax-speed="40"
        data-parallax-invert="true"
        class="absolute bottom-[-10%] right-[0%] w-[450px] h-[450px] bg-fuchsia-500/15 rounded-full blur-[100px] animate-pulse-slow"
        style="animation-delay: 2s;"
      >
      </div>
      <!-- Medium cyan orb - right -->
      <div
        data-parallax-orb
        data-parallax-speed="50"
        class="absolute top-[30%] right-[10%] w-[300px] h-[300px] bg-cyan-500/15 rounded-full blur-[80px] animate-pulse-slow"
        style="animation-delay: 4s;"
      >
      </div>
      <!-- Small accent orb - center left -->
      <div
        data-parallax-orb
        data-parallax-speed="60"
        data-parallax-invert="true"
        class="absolute top-[50%] left-[15%] w-[200px] h-[200px] bg-fuchsia-500/10 rounded-full blur-[60px] animate-pulse-slow"
        style="animation-delay: 3s;"
      >
      </div>
    </div>
    """
  end

  @doc """
  Renders a word that rotates through a list of words with animation.
  Uses WordRotate JS hook.
  """
  attr :words, :list, required: true
  attr :duration, :integer, default: 2500
  attr :class, :string, default: nil

  def word_rotate(assigns) do
    ~H"""
    <span
      phx-hook="WordRotate"
      id={"word-rotate-#{:erlang.phash2(@words)}"}
      data-words={Jason.encode!(@words)}
      data-duration={@duration}
      class={["inline-block", @class]}
    >
      <span data-word-display class="inline-block word-enter">{List.first(@words)}</span>
    </span>
    """
  end

  @doc """
  Renders a stack of avatar images with animated tooltips.
  Shows social proof with active avatar count.
  """
  attr :avatars, :list, default: []
  attr :total_count, :integer, default: 0
  attr :max_visible, :integer, default: 5
  attr :class, :string, default: nil

  def avatar_stack(assigns) do
    visible_avatars = Enum.take(assigns.avatars, assigns.max_visible)
    remaining = max(assigns.total_count - length(visible_avatars), 0)

    assigns =
      assigns
      |> assign(:visible_avatars, visible_avatars)
      |> assign(:remaining, remaining)

    ~H"""
    <div class={["flex flex-col items-center gap-3", @class]}>
      <!-- Avatar Stack -->
      <div class="flex items-center -space-x-3">
        <%= for {avatar, idx} <- Enum.with_index(@visible_avatars) do %>
          <div
            class="avatar-stack-item relative group"
            style={"z-index: #{length(@visible_avatars) - idx}"}
          >
            <div class="w-10 h-10 rounded-full ring-2 ring-zinc-900 bg-gradient-to-br from-cyan-500/30 to-fuchsia-500/30 flex items-center justify-center overflow-hidden">
              <%= if avatar.avatar_url do %>
                <img src={avatar.avatar_url} alt={avatar.name} class="w-full h-full object-cover" />
              <% else %>
                <span class="text-xs font-bold text-white/80">
                  {String.first(avatar.name)}
                </span>
              <% end %>
            </div>
            <!-- Tooltip -->
            <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
              <span class="text-xs font-medium text-white">{avatar.name}</span>
              <div class="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-zinc-800">
              </div>
            </div>
          </div>
        <% end %>
        <!-- Remaining count -->
        <%= if @remaining > 0 do %>
          <div class="w-10 h-10 rounded-full ring-2 ring-zinc-900 bg-zinc-800 flex items-center justify-center">
            <span class="text-xs font-bold text-zinc-400">+{@remaining}</span>
          </div>
        <% end %>
      </div>
      <!-- Stats text -->
      <div class="flex items-center gap-2 text-sm text-zinc-400">
        <%= if @total_count > 0 do %>
          <span class="text-cyan-400 font-semibold">{@total_count}</span>
          <span>avatares ativos</span>
          <span class="text-amber-400">★★★★★</span>
        <% else %>
          <span class="text-zinc-500">Avatares em espera</span>
        <% end %>
      </div>
    </div>
    """
  end

  @doc """
  Renders hero statistics with animated counters.
  """
  attr :avatar_count, :integer, default: 0
  attr :memory_count, :integer, default: 0
  attr :relationship_count, :integer, default: 0
  attr :class, :string, default: nil

  def hero_stats(assigns) do
    ~H"""
    <div class={["flex items-center justify-center gap-8 md:gap-12", @class]}>
      <.hero_stat_item value={@avatar_count} label="Avatares" />
      <.hero_stat_item value={@memory_count} label="Memórias" />
      <.hero_stat_item value={@relationship_count} label="Conexões" />
    </div>
    """
  end

  defp hero_stat_item(assigns) do
    ~H"""
    <div class="text-center">
      <div class="text-2xl md:text-3xl font-bold text-white stat-number">{@value}</div>
      <div class="text-xs text-zinc-500 uppercase tracking-wider">{@label}</div>
    </div>
    """
  end

  # ============================================================================
  # FEATURES SECTION
  # ============================================================================

  @doc """
  Renders the features grid section with stats and gradient borders.
  Inspired by Metronic's feature cards.
  """
  attr :class, :string, default: nil

  @spec features_section(map()) :: Phoenix.LiveView.Rendered.t()
  def features_section(assigns) do
    ~H"""
    <section id="features" class={["py-24 px-6 relative", @class]}>
      <div class="max-w-6xl mx-auto">
        <!-- Section Header -->
        <div class="text-center mb-16">
          <span class="inline-block px-4 py-1.5 rounded-full bg-fuchsia-500/10 border border-fuchsia-500/20 text-xs font-medium text-fuchsia-400 uppercase tracking-wider mb-4">
            Fundamentos
          </span>
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
            Psicologia digital completa
          </h2>
          <p class="text-zinc-400 max-w-2xl mx-auto">
            Cada avatar é construído com modelos psicológicos reais para comportamentos autênticos
          </p>
        </div>
        
    <!-- Features Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <.feature_card
            icon="hero-finger-print"
            title="Personalidade Big Five"
            description="Cinco traços fundamentais: Abertura, Conscienciosidade, Extroversão, Amabilidade e Neuroticismo definem cada avatar."
            color="cyan"
            stat="5"
            stat_label="Traços"
          />
          <.feature_card
            icon="hero-puzzle-piece"
            title="Eneagrama"
            description="9 tipos psicológicos que influenciam motivações profundas, medos centrais e padrões de comportamento."
            color="violet"
            stat="9"
            stat_label="Tipos"
          />
          <.feature_card
            icon="hero-heart"
            title="Emoções em Tempo Real"
            description="10 emoções que evoluem dinamicamente: alegria, tristeza, raiva, medo, surpresa, nojo, amor, solidão, curiosidade e empolgação."
            color="rose"
            stat="10"
            stat_label="Emoções"
          />
          <.feature_card
            icon="hero-users"
            title="Relacionamentos"
            description="Conexões que crescem organicamente com confiança, afeição e familiaridade. De desconhecidos a melhores amigos."
            color="blue"
            stat="∞"
            stat_label="Conexões"
          />
          <.feature_card
            icon="hero-chat-bubble-left-right"
            title="Conversas Autônomas"
            description="Avatares iniciam conversas por conta própria baseados em suas necessidades sociais e interesses."
            color="amber"
            stat="24/7"
            stat_label="Ativo"
          />
          <.feature_card
            icon="hero-sparkles"
            title="Memórias Semânticas"
            description="Sistema RAG com Qdrant armazena e recupera memórias relevantes para conversas contextualizadas."
            color="cyan"
            stat="RAG"
            stat_label="Semântico"
          />
        </div>
      </div>
    </section>
    """
  end

  @doc """
  Renders a single feature card with stats and gradient border on hover.
  """
  attr :icon, :string, required: true
  attr :title, :string, required: true
  attr :description, :string, required: true
  attr :color, :string, default: "cyan"
  attr :stat, :string, default: nil
  attr :stat_label, :string, default: nil

  @spec feature_card(map()) :: Phoenix.LiveView.Rendered.t()
  def feature_card(assigns) do
    ~H"""
    <div class="group relative p-6 rounded-2xl bg-zinc-900/50 border border-white/5 backdrop-blur-sm hover:bg-zinc-800/50 hover:border-white/10 transition-all duration-300 hover:-translate-y-1 overflow-hidden">
      <!-- Header with icon and stat -->
      <div class="flex items-start justify-between mb-4">
        <div class={[
          "w-12 h-12 rounded-xl flex items-center justify-center transition-colors",
          feature_icon_bg(@color)
        ]}>
          <.icon name={@icon} class={["w-6 h-6", feature_icon_color(@color)]} />
        </div>
        <!-- Stat badge -->
        <%= if @stat do %>
          <div class="text-right">
            <div class={["text-2xl font-bold", feature_stat_color(@color)]}>{@stat}</div>
            <div class="text-xs text-zinc-500 uppercase tracking-wider">{@stat_label}</div>
          </div>
        <% end %>
      </div>

      <h3 class="text-lg font-semibold text-white mb-2 group-hover:text-cyan-400 transition-colors">
        {@title}
      </h3>
      <p class="text-sm text-zinc-400 leading-relaxed">
        {@description}
      </p>
      
    <!-- Gradient border line on hover -->
      <div class="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500 via-fuchsia-500 to-cyan-500 gradient-border-line">
      </div>
    </div>
    """
  end

  defp feature_stat_color("cyan"), do: "text-cyan-400"
  defp feature_stat_color("fuchsia"), do: "text-fuchsia-400"
  defp feature_stat_color("violet"), do: "text-violet-400"
  defp feature_stat_color("rose"), do: "text-rose-400"
  defp feature_stat_color("blue"), do: "text-blue-400"
  defp feature_stat_color("amber"), do: "text-amber-400"
  defp feature_stat_color(_), do: "text-zinc-400"

  defp feature_icon_bg("cyan"), do: "bg-cyan-500/10 group-hover:bg-cyan-500/20"
  defp feature_icon_bg("fuchsia"), do: "bg-fuchsia-500/10 group-hover:bg-fuchsia-500/20"
  defp feature_icon_bg("violet"), do: "bg-violet-500/10 group-hover:bg-violet-500/20"
  defp feature_icon_bg("rose"), do: "bg-rose-500/10 group-hover:bg-rose-500/20"
  defp feature_icon_bg("blue"), do: "bg-blue-500/10 group-hover:bg-blue-500/20"
  defp feature_icon_bg("amber"), do: "bg-amber-500/10 group-hover:bg-amber-500/20"
  defp feature_icon_bg(_), do: "bg-zinc-500/10 group-hover:bg-zinc-500/20"

  defp feature_icon_color("cyan"), do: "text-cyan-400"
  defp feature_icon_color("fuchsia"), do: "text-fuchsia-400"
  defp feature_icon_color("violet"), do: "text-violet-400"
  defp feature_icon_color("rose"), do: "text-rose-400"
  defp feature_icon_color("blue"), do: "text-blue-400"
  defp feature_icon_color("amber"), do: "text-amber-400"
  defp feature_icon_color(_), do: "text-zinc-400"

  # ============================================================================
  # HOW IT WORKS SECTION
  # ============================================================================

  @doc """
  Renders the "How It Works" section with auto-advancing steps.
  Uses StepProgress JS hook for auto-advancement.
  """
  attr :class, :string, default: nil
  attr :step_duration, :integer, default: 5000

  def how_it_works_section(assigns) do
    ~H"""
    <section id="how-it-works" class={["py-24 px-6 bg-purple-950/20 relative", @class]}>
      <div class="max-w-6xl mx-auto">
        <!-- Section Header -->
        <div class="text-center mb-16">
          <span class="inline-block px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-xs font-medium text-cyan-400 uppercase tracking-wider mb-4">
            Processo
          </span>
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
            Como Funciona
          </h2>
          <p class="text-zinc-400 max-w-2xl mx-auto">
            Em poucos passos, crie avatares com vida própria
          </p>
        </div>
        
    <!-- Steps Container with Hook -->
        <div
          phx-hook="StepProgress"
          id="step-progress-container"
          data-step-duration={@step_duration}
          class="max-w-4xl mx-auto"
        >
          <!-- Step Tabs -->
          <div class="flex flex-wrap justify-center gap-4 md:gap-8 mb-12">
            <.step_tab number={1} title="Criar" index={0} />
            <.step_tab number={2} title="Personalizar" index={1} />
            <.step_tab number={3} title="Observar" index={2} />
            <.step_tab number={4} title="Evoluir" index={3} />
          </div>
          
    <!-- Step Contents -->
          <div class="relative">
            <.step_content
              index={0}
              icon="hero-user-plus"
              title="Crie seu Avatar"
              description="Escolha nome, gênero e idade. Deixe o sistema gerar uma personalidade única baseada em ciência psicológica real."
              features={["Nome e identidade", "Big Five gerado automaticamente", "Eneagrama aleatório"]}
            />
            <.step_content
              index={1}
              icon="hero-adjustments-horizontal"
              title="Personalize a Personalidade"
              description="Ajuste os traços de personalidade se desejar. Cada configuração afeta como o avatar pensa, sente e age."
              features={[
                "Ajuste fino dos 5 traços",
                "Escolha de tipo Eneagrama",
                "Define padrões de comportamento"
              ]}
            />
            <.step_content
              index={2}
              icon="hero-eye"
              title="Observe a Vida Autônoma"
              description="Seu avatar começa a viver. Ele pensa, sente emoções, conversa com outros avatares e forma memórias."
              features={["Pensamentos em tempo real", "Emoções dinâmicas", "Conversas autônomas"]}
            />
            <.step_content
              index={3}
              icon="hero-chart-bar"
              title="Acompanhe a Evolução"
              description="Veja relacionamentos se formarem, memórias se acumularem e a personalidade se desenvolver ao longo do tempo."
              features={["Grafo de relacionamentos", "Histórico de memórias", "Métricas de bem-estar"]}
            />
          </div>
        </div>
      </div>
    </section>
    """
  end

  # Step tab with progress bar
  attr :number, :integer, required: true
  attr :title, :string, required: true
  attr :index, :integer, required: true

  defp step_tab(assigns) do
    ~H"""
    <button
      data-step={@index}
      class="group flex flex-col items-center gap-2 cursor-pointer step-inactive transition-colors"
    >
      <div class="flex items-center gap-2">
        <span class="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-sm font-bold group-[.step-active]:bg-cyan-500 group-[.step-active]:text-black transition-colors">
          {@number}
        </span>
        <span class="text-sm font-medium hidden sm:block">{@title}</span>
      </div>
      <!-- Progress Bar -->
      <div class="w-full h-0.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          data-progress-bar
          class="h-full bg-gradient-to-r from-cyan-500 to-fuchsia-500 transition-all duration-100"
          style="width: 0%"
        >
        </div>
      </div>
    </button>
    """
  end

  # Step content panel
  attr :index, :integer, required: true
  attr :icon, :string, required: true
  attr :title, :string, required: true
  attr :description, :string, required: true
  attr :features, :list, default: []

  defp step_content(assigns) do
    ~H"""
    <div
      data-step-content={@index}
      class={["rounded-2xl bg-zinc-900/50 border border-white/5 p-8 md:p-12", @index != 0 && "hidden"]}
    >
      <div class="flex flex-col md:flex-row gap-8 items-center">
        <!-- Icon -->
        <div class="flex-shrink-0">
          <div class="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-fuchsia-500/20 flex items-center justify-center">
            <.icon name={@icon} class="w-10 h-10 text-cyan-400" />
          </div>
        </div>
        
    <!-- Content -->
        <div class="flex-1 text-center md:text-left">
          <h3 class="text-2xl font-bold text-white mb-3">{@title}</h3>
          <p class="text-zinc-400 mb-6">{@description}</p>
          
    <!-- Features list -->
          <ul class="flex flex-wrap justify-center md:justify-start gap-3">
            <%= for feature <- @features do %>
              <li class="flex items-center gap-2 text-sm text-zinc-300">
                <.icon name="hero-check-circle" class="w-4 h-4 text-cyan-400" />
                {feature}
              </li>
            <% end %>
          </ul>
        </div>
      </div>
    </div>
    """
  end

  # ============================================================================
  # DEMO SECTION HEADER
  # ============================================================================

  @doc """
  Renders the demo section header.
  """
  attr :avatar_count, :integer, default: 0
  attr :class, :string, default: nil

  @spec demo_section_header(map()) :: Phoenix.LiveView.Rendered.t()
  def demo_section_header(assigns) do
    ~H"""
    <div class={["text-center mb-12", @class]}>
      <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
        <span class="w-2 h-2 rounded-full bg-cyan-400 animate-synapse"></span>
        <span class="text-xs font-medium text-cyan-400 uppercase tracking-wider">
          Sinapses Ativas
        </span>
      </div>
      <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
        Observe avatares reais
      </h2>
      <p class="text-zinc-400 max-w-2xl mx-auto">
        Estes avatares estao vivos agora, pensando e sentindo em tempo real.
        <span :if={@avatar_count > 0} class="text-cyan-400">{@avatar_count} neuronios ativos.</span>
      </p>
    </div>
    """
  end

  # ============================================================================
  # TECHNOLOGY SECTION
  # ============================================================================

  @doc """
  Renders the technology stack section.
  """
  attr :class, :string, default: nil

  @spec tech_section(map()) :: Phoenix.LiveView.Rendered.t()
  def tech_section(assigns) do
    ~H"""
    <section class={["py-24 px-6 bg-purple-950/30 relative", @class]}>
      <div class="max-w-6xl mx-auto">
        <!-- Section Header -->
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
            Tecnologia de ponta
          </h2>
          <p class="text-zinc-400 max-w-2xl mx-auto">
            Stack moderno projetado para simular vidas digitais em escala
          </p>
        </div>
        
    <!-- Tech Stack Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
          <.tech_card name="NVIDIA NIM" description="LLM Cloud API" icon="hero-cpu-chip" />
          <.tech_card name="Phoenix LiveView" description="Real-time UI" icon="hero-bolt" />
          <.tech_card name="TimescaleDB" description="Time-series DB" icon="hero-circle-stack" />
          <.tech_card name="Qdrant" description="Vector Memory" icon="hero-sparkles" />
        </div>
        
    <!-- Divider -->
        <div class="border-t border-white/5 my-12"></div>
        
    <!-- AI Capabilities label -->
        <div class="text-center mb-8">
          <span class="text-xs uppercase tracking-widest text-zinc-500 font-medium">
            Capacidades de IA
          </span>
        </div>
      </div>
    </section>
    """
  end

  @doc """
  Renders a technology card.
  """
  attr :name, :string, required: true
  attr :description, :string, required: true
  attr :icon, :string, required: true

  @spec tech_card(map()) :: Phoenix.LiveView.Rendered.t()
  def tech_card(assigns) do
    ~H"""
    <div class="p-4 rounded-xl bg-zinc-900/50 border border-white/5 hover:border-white/10 transition-all duration-300 text-center group hover:bg-zinc-800/50">
      <div class="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center mx-auto mb-3 group-hover:bg-white/10 transition-colors">
        <.icon name={@icon} class="w-5 h-5 text-zinc-400 group-hover:text-white transition-colors" />
      </div>
      <h3 class="font-semibold text-white text-sm mb-1">{@name}</h3>
      <p class="text-xs text-zinc-500">{@description}</p>
    </div>
    """
  end

  # ============================================================================
  # FINAL CTA SECTION
  # ============================================================================

  @doc """
  Renders the final CTA section with neural grid background.
  Inspired by Metronic's background boxes.
  """
  attr :class, :string, default: nil

  @spec final_cta(map()) :: Phoenix.LiveView.Rendered.t()
  def final_cta(assigns) do
    ~H"""
    <section class={["py-32 px-6 relative overflow-hidden", @class]}>
      <!-- Neural Grid Background -->
      <.neural_grid_bg />
      
    <!-- Background glow -->
      <div class="absolute inset-0 pointer-events-none">
        <div class="absolute top-1/2 left-1/3 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-cyan-900/30 rounded-full blur-[150px]">
        </div>
        <div class="absolute top-1/2 right-1/3 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-fuchsia-900/25 rounded-full blur-[120px]">
        </div>
      </div>

      <div class="max-w-3xl mx-auto text-center relative z-10">
        <span class="inline-block px-4 py-1.5 rounded-full bg-gradient-to-r from-cyan-500/10 to-fuchsia-500/10 border border-cyan-500/20 text-xs font-medium text-cyan-400 uppercase tracking-wider mb-6">
          Em Breve
        </span>
        <h2 class="text-3xl md:text-5xl font-bold text-white mb-6">
          Pronto para criar seu avatar?
        </h2>
        <p class="text-lg text-zinc-400 mb-10 max-w-xl mx-auto">
          Crie um avatar com personalidade única e observe sua vida digital se desenvolver autonomamente.
        </p>
        <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            class="px-10 py-4 rounded-full bg-gradient-to-r from-cyan-500/50 to-fuchsia-500/50 text-white/60 font-semibold text-sm tracking-wide cursor-not-allowed"
            disabled
          >
            Começar Agora
          </button>
          <a
            href="#demo"
            class="px-8 py-4 rounded-full border border-cyan-500/30 text-white font-medium text-sm tracking-wide hover:bg-cyan-500/10 hover:border-cyan-400/50 transition-all duration-300"
          >
            Ver Demo Novamente
          </a>
        </div>
        <p class="text-xs text-zinc-600 mt-6">
          Criação de avatares disponível em breve
        </p>
      </div>
    </section>
    """
  end

  @doc """
  Renders the neural grid background pattern.
  Creates an isometric grid effect inspired by Metronic's BackgroundBoxes.
  """
  def neural_grid_bg(assigns) do
    assigns = assign_new(assigns, :class, fn -> nil end)

    ~H"""
    <div class={["neural-grid-container", @class]}>
      <div class="neural-grid">
        <%= for row <- 0..14 do %>
          <div class="neural-grid-row">
            <%= for _col <- 0..9 do %>
              <div class={[
                "neural-grid-cell",
                rem(row, 3) == 0 && "opacity-60",
                rem(row, 5) == 0 && "opacity-40"
              ]}>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  # ============================================================================
  # NAVBAR
  # ============================================================================

  @doc """
  Renders the landing page navbar.
  """
  attr :class, :string, default: nil

  @spec landing_navbar(map()) :: Phoenix.LiveView.Rendered.t()
  def landing_navbar(assigns) do
    ~H"""
    <nav class={[
      "fixed top-0 w-full z-50 px-6 py-4 flex justify-between items-center backdrop-blur-xl bg-[#0a0612]/80 border-b border-fuchsia-500/10",
      @class
    ]}>
      <div class="flex items-center gap-2">
        <div class="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.6)] animate-synapse">
        </div>
        <span class="font-bold tracking-tight text-white text-lg">VIVA</span>
      </div>

      <div class="flex items-center gap-6">
        <a
          href="#features"
          class="text-sm text-zinc-400 hover:text-white transition-colors hidden sm:block"
        >
          Features
        </a>
        <a
          href="#how-it-works"
          class="text-sm text-zinc-400 hover:text-white transition-colors hidden md:block"
        >
          Como Funciona
        </a>
        <a href="#demo" class="text-sm text-zinc-400 hover:text-white transition-colors hidden sm:block">
          Demo
        </a>
        <a href="#tech" class="text-sm text-zinc-400 hover:text-white transition-colors hidden sm:block">
          Tech
        </a>
      </div>
    </nav>
    """
  end

  # ============================================================================
  # FOOTER
  # ============================================================================

  @doc """
  Renders the landing page footer.
  """
  attr :class, :string, default: nil

  @spec landing_footer(map()) :: Phoenix.LiveView.Rendered.t()
  def landing_footer(assigns) do
    ~H"""
    <footer class={["py-16 px-6 border-t border-fuchsia-500/10 bg-[#0a0612]/60", @class]}>
      <div class="max-w-6xl mx-auto">
        <div class="flex flex-col md:flex-row justify-between items-center gap-8">
          <!-- Logo -->
          <div class="flex items-center gap-2">
            <div class="w-2 h-2 rounded-full bg-cyan-400"></div>
            <span class="font-bold text-xl text-zinc-600">VIVA</span>
          </div>
          
    <!-- Links -->
          <div class="flex items-center gap-8 text-sm text-zinc-500">
            <a href="#" class="hover:text-zinc-300 transition-colors">Privacidade</a>
            <a href="#" class="hover:text-zinc-300 transition-colors">Termos</a>
            <a href="#" class="hover:text-zinc-300 transition-colors">Contato</a>
          </div>
        </div>
        
    <!-- Bottom -->
        <div class="mt-12 pt-8 border-t border-white/5 text-center">
          <p class="text-xs text-zinc-700">
            VIVA - Observatorio de Vida Digital
          </p>
          <p class="text-xs text-zinc-800 mt-2">
            Avatares autonomos pensando, sentindo e interagindo 24/7
          </p>
        </div>
      </div>
    </footer>
    """
  end
end
