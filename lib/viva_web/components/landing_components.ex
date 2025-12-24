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
  Renders the hero section with headline, subheadline and CTA.
  """
  attr :class, :string, default: nil

  @spec hero_section(map()) :: Phoenix.LiveView.Rendered.t()
  def hero_section(assigns) do
    ~H"""
    <section class={[
      "relative min-h-[85vh] flex flex-col items-center justify-center px-6 text-center",
      @class
    ]}>
      <!-- Aurora Background Effects -->
      <div class="absolute inset-0 overflow-hidden pointer-events-none">
        <div class="absolute top-[-20%] left-[10%] w-[600px] h-[600px] bg-emerald-900/30 rounded-full blur-[150px] animate-pulse-slow">
        </div>
        <div
          class="absolute bottom-[-10%] right-[5%] w-[500px] h-[500px] bg-violet-900/20 rounded-full blur-[120px] animate-pulse-slow"
          style="animation-delay: 2s;"
        >
        </div>
        <div
          class="absolute top-[40%] right-[20%] w-[300px] h-[300px] bg-blue-900/15 rounded-full blur-[100px] animate-pulse-slow"
          style="animation-delay: 4s;"
        >
        </div>
      </div>
      
    <!-- Content -->
      <div class="relative z-10 max-w-4xl mx-auto">
        <!-- Badge -->
        <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-md mb-8 animate-fade-in">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span class="text-xs uppercase tracking-widest font-medium text-zinc-300">
            Simulacao Ativa 24/7
          </span>
        </div>
        
    <!-- Headline -->
        <h1
          class="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white mb-6 animate-fade-in"
          style="animation-delay: 0.1s;"
        >
          Avatares digitais vivendo
          <span class="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
            vidas autonomas
          </span>
        </h1>
        
    <!-- Subheadline -->
        <p
          class="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in"
          style="animation-delay: 0.2s;"
        >
          Cada avatar tem personalidade unica, emocoes em tempo real, memorias e relacionamentos que evoluem naturalmente. Observe vidas digitais se desenrolando 24 horas por dia.
        </p>
        
    <!-- CTA Button -->
        <div
          class="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in"
          style="animation-delay: 0.3s;"
        >
          <a
            href="#demo"
            class="group px-8 py-4 rounded-full bg-white text-black font-semibold text-sm tracking-wide hover:bg-zinc-100 transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-white/10 flex items-center gap-2"
          >
            Ver Demo ao Vivo
            <.icon
              name="hero-arrow-down"
              class="w-4 h-4 group-hover:translate-y-1 transition-transform"
            />
          </a>
          <a
            href="#features"
            class="px-8 py-4 rounded-full border border-white/20 text-white font-medium text-sm tracking-wide hover:bg-white/5 transition-all duration-300"
          >
            Conhecer o Projeto
          </a>
        </div>
      </div>
      
    <!-- Scroll indicator -->
      <div class="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <.icon name="hero-chevron-down" class="w-6 h-6 text-zinc-600" />
      </div>
    </section>
    """
  end

  # ============================================================================
  # FEATURES SECTION
  # ============================================================================

  @doc """
  Renders the features grid section.
  """
  attr :class, :string, default: nil

  @spec features_section(map()) :: Phoenix.LiveView.Rendered.t()
  def features_section(assigns) do
    ~H"""
    <section id="features" class={["py-24 px-6 relative", @class]}>
      <div class="max-w-6xl mx-auto">
        <!-- Section Header -->
        <div class="text-center mb-16">
          <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
            Psicologia digital completa
          </h2>
          <p class="text-zinc-400 max-w-2xl mx-auto">
            Cada avatar e construido com modelos psicologicos reais para comportamentos autenticos
          </p>
        </div>
        
    <!-- Features Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <.feature_card
            icon="hero-finger-print"
            title="Personalidade Big Five"
            description="Cinco tracos fundamentais: Abertura, Conscienciosidade, Extroversao, Amabilidade e Neuroticismo definem cada avatar."
            color="emerald"
          />
          <.feature_card
            icon="hero-puzzle-piece"
            title="Eneagrama"
            description="9 tipos psicologicos que influenciam motivacoes profundas, medos centrais e padroes de comportamento."
            color="violet"
          />
          <.feature_card
            icon="hero-heart"
            title="Emocoes em Tempo Real"
            description="10 emocoes que evoluem dinamicamente: alegria, tristeza, raiva, medo, surpresa, nojo, amor, solidao, curiosidade e empolgacao."
            color="rose"
          />
          <.feature_card
            icon="hero-users"
            title="Relacionamentos"
            description="Conexoes que crescem organicamente com confianca, afeicao e familiaridade. De desconhecidos a melhores amigos."
            color="blue"
          />
          <.feature_card
            icon="hero-chat-bubble-left-right"
            title="Conversas Autonomas"
            description="Avatares iniciam conversas por conta propria baseados em suas necessidades sociais e interesses."
            color="amber"
          />
          <.feature_card
            icon="hero-sparkles"
            title="Memorias Semanticas"
            description="Sistema RAG com Qdrant armazena e recupera memorias relevantes para conversas contextualizadas."
            color="cyan"
          />
        </div>
      </div>
    </section>
    """
  end

  @doc """
  Renders a single feature card.
  """
  attr :icon, :string, required: true
  attr :title, :string, required: true
  attr :description, :string, required: true
  attr :color, :string, default: "emerald"

  @spec feature_card(map()) :: Phoenix.LiveView.Rendered.t()
  def feature_card(assigns) do
    ~H"""
    <div class="group p-6 rounded-2xl bg-zinc-900/50 border border-white/5 backdrop-blur-sm hover:bg-zinc-800/50 hover:border-white/10 transition-all duration-300 hover:-translate-y-1">
      <div class={[
        "w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-colors",
        feature_icon_bg(@color)
      ]}>
        <.icon name={@icon} class={["w-6 h-6", feature_icon_color(@color)]} />
      </div>
      <h3 class="text-lg font-semibold text-white mb-2 group-hover:text-emerald-400 transition-colors">
        {@title}
      </h3>
      <p class="text-sm text-zinc-400 leading-relaxed">
        {@description}
      </p>
    </div>
    """
  end

  defp feature_icon_bg("emerald"), do: "bg-emerald-500/10 group-hover:bg-emerald-500/20"
  defp feature_icon_bg("violet"), do: "bg-violet-500/10 group-hover:bg-violet-500/20"
  defp feature_icon_bg("rose"), do: "bg-rose-500/10 group-hover:bg-rose-500/20"
  defp feature_icon_bg("blue"), do: "bg-blue-500/10 group-hover:bg-blue-500/20"
  defp feature_icon_bg("amber"), do: "bg-amber-500/10 group-hover:bg-amber-500/20"
  defp feature_icon_bg("cyan"), do: "bg-cyan-500/10 group-hover:bg-cyan-500/20"
  defp feature_icon_bg(_), do: "bg-zinc-500/10 group-hover:bg-zinc-500/20"

  defp feature_icon_color("emerald"), do: "text-emerald-400"
  defp feature_icon_color("violet"), do: "text-violet-400"
  defp feature_icon_color("rose"), do: "text-rose-400"
  defp feature_icon_color("blue"), do: "text-blue-400"
  defp feature_icon_color("amber"), do: "text-amber-400"
  defp feature_icon_color("cyan"), do: "text-cyan-400"
  defp feature_icon_color(_), do: "text-zinc-400"

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
      <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 mb-6">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="text-xs font-medium text-emerald-400 uppercase tracking-wider">
          Demo ao Vivo
        </span>
      </div>
      <h2 class="text-3xl md:text-4xl font-bold text-white mb-4">
        Observe avatares reais
      </h2>
      <p class="text-zinc-400 max-w-2xl mx-auto">
        Estes avatares estao vivos agora, pensando e sentindo em tempo real.
        <span :if={@avatar_count > 0} class="text-emerald-400">{@avatar_count} avatares online.</span>
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
    <section class={["py-24 px-6 bg-zinc-950/50 relative", @class]}>
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
  Renders the final CTA section.
  """
  attr :class, :string, default: nil

  @spec final_cta(map()) :: Phoenix.LiveView.Rendered.t()
  def final_cta(assigns) do
    ~H"""
    <section class={["py-24 px-6 relative overflow-hidden", @class]}>
      <!-- Background glow -->
      <div class="absolute inset-0 pointer-events-none">
        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-900/20 rounded-full blur-[150px]">
        </div>
      </div>

      <div class="max-w-3xl mx-auto text-center relative z-10">
        <h2 class="text-3xl md:text-5xl font-bold text-white mb-6">
          Pronto para criar seu avatar?
        </h2>
        <p class="text-lg text-zinc-400 mb-10 max-w-xl mx-auto">
          Crie um avatar com personalidade unica e observe sua vida digital se desenvolver autonomamente.
        </p>
        <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            class="px-10 py-4 rounded-full bg-white text-black font-semibold text-sm tracking-wide hover:bg-zinc-100 transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-white/10 cursor-not-allowed opacity-60"
            disabled
          >
            Em Breve
          </button>
          <a
            href="#demo"
            class="px-8 py-4 rounded-full border border-white/20 text-white font-medium text-sm tracking-wide hover:bg-white/5 transition-all duration-300"
          >
            Ver Demo Novamente
          </a>
        </div>
        <p class="text-xs text-zinc-600 mt-6">
          Criacao de avatares disponivel em breve
        </p>
      </div>
    </section>
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
      "fixed top-0 w-full z-50 px-6 py-4 flex justify-between items-center backdrop-blur-xl bg-black/60 border-b border-white/5",
      @class
    ]}>
      <div class="flex items-center gap-2">
        <div class="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.5)]">
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
    <footer class={["py-16 px-6 border-t border-white/5 bg-black/40", @class]}>
      <div class="max-w-6xl mx-auto">
        <div class="flex flex-col md:flex-row justify-between items-center gap-8">
          <!-- Logo -->
          <div class="flex items-center gap-2">
            <div class="w-2 h-2 rounded-full bg-emerald-400"></div>
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
