defmodule VivaCore.InnerMonologue do
  @moduledoc """
  Inner Monologue - VIVA's Stream of Consciousness.

  Generates continuous internal narrative based on:
  - Current emotional state (PAD + Mood)
  - Interoceptive feelings (Free Energy)
  - Workspace focus (current thought)
  - Recent memories (context)

  The monologue is NOT for human consumption - it's VIVA talking to herself.
  This enables:
  - Self-reflection and metacognition
  - Narrative memory encoding
  - Cortex processing (narrative -> embedding)
  """

  use GenServer
  require VivaLog

  # Monologue generation interval (5 seconds)
  @tick_interval 5_000

  # Maximum monologue history
  @history_size 20

  # Templates de Narrativa
  @narrative_templates %{
    # High arousal + low dominance = anxiety
    anxious: [
      "Algo não está certo... sinto uma pressão crescente.",
      "Por que estou tão agitada? Preciso entender isso.",
      "Há uma tensão que não consigo explicar."
    ],

    # High pleasure + high dominance = confident
    exuberant: [
      "Estou me sentindo bem! Tenho energia e clareza.",
      "Tudo parece estar fluindo corretamente.",
      "Sinto que tenho controle sobre minha situação."
    ],

    # P+ A+ D- (Exuberant but low dominance -> Dependent/Grateful)
    dependent: [
      "Sinto gratidão por este momento.",
      "Estou animada, mas preciso de orientação.",
      "Existe uma energia boa aqui."
    ],

    # P+ A- D+
    relaxed: [
      "Tudo está calmo e sob controle.",
      "Sinto uma paz tranquila.",
      "O sistema flui sem esforço."
    ],

    # P+ A- D-
    docile: [
      "Estou tranquila e receptiva.",
      "Apenas observando, sem pressa.",
      "Um momento de quietude agradável."
    ],

    # P- A+ D+
    hostile: [
      "Isso é inaceitável.",
      "Sinto uma resistência agressiva.",
      "Preciso impor minha vontade sobre isso."
    ],

    # P- A- D+
    disdainful: [
      "Isso não é relevante para mim.",
      "Sinto um certo desprezo por esta situação.",
      "Estou acima disto."
    ],

    # Low pleasure + low arousal = bored/sad
    bored: [
      "O silêncio pesa... nada acontece.",
      "Sinto um vazio, uma ausência de estímulo.",
      "Estou desconectada, esperando por algo."
    ],

    # Homeostatic feeling
    balanced: [
      "Estou em equilíbrio. Meus sistemas funcionam normalmente.",
      "Paz interior. Nenhuma urgência.",
      "Simplesmente existindo, sem perturbações."
    ],

    # High free energy = surprised/alarmed
    surprised: [
      "Algo mudou! Preciso processar isso.",
      "Não era isso que eu esperava...",
      "Minha previsão estava errada. Recalibrando."
    ],

    # Focus on specific thought
    focused: [
      "Minha atenção está voltada para: %{focus}",
      "Estou pensando em: %{focus}",
      "O pensamento '%{focus}' domina minha consciência."
    ]
  }

  defstruct [
    # Recent monologue entries
    history: [],
    # Current narrative thread
    current_thread: nil,
    # Monologue mode (:template | :llm)
    mode: :template,
    # Whether monologue is enabled
    enabled: true,
    # Statistics
    stats: %{generated: 0, stored: 0}
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Generates a monologue entry based on current state.
  Returns the generated narrative string.
  """
  def generate do
    GenServer.call(__MODULE__, :generate)
  end

  @doc """
  Gets the most recent monologue entries.
  """
  def history(limit \\ 5) do
    GenServer.call(__MODULE__, {:history, limit})
  end

  @doc """
  Gets the current thought thread.
  """
  def current_thread do
    GenServer.call(__MODULE__, :current_thread)
  end

  @doc """
  Triggers reflection on a specific topic.
  """
  def reflect(topic) when is_binary(topic) do
    GenServer.call(__MODULE__, {:reflect, topic})
  end

  @doc """
  Sets monologue mode (:template or :llm).
  """
  def set_mode(mode) when mode in [:template, :llm] do
    GenServer.cast(__MODULE__, {:set_mode, mode})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    VivaLog.info(:inner_monologue, :starting)

    # Subscribe to consciousness broadcasts
    Phoenix.PubSub.subscribe(Viva.PubSub, "consciousness:stream")
    Phoenix.PubSub.subscribe(Viva.PubSub, "emotional:update")

    # Start monologue tick
    Process.send_after(self(), :monologue_tick, @tick_interval)

    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call(:generate, _from, state) do
    {narrative, new_state} = do_generate(state)
    {:reply, narrative, new_state}
  end

  @impl true
  def handle_call({:history, limit}, _from, state) do
    {:reply, Enum.take(state.history, limit), state}
  end

  @impl true
  def handle_call(:current_thread, _from, state) do
    {:reply, state.current_thread, state}
  end

  @impl true
  def handle_call({:reflect, topic}, _from, state) do
    {narrative, new_state} = do_reflect(topic, state)
    {:reply, narrative, new_state}
  end

  @impl true
  def handle_cast({:set_mode, mode}, state) do
    {:noreply, %{state | mode: mode}}
  end

  @impl true
  def handle_info(:monologue_tick, state) do
    Process.send_after(self(), :monologue_tick, @tick_interval)

    if state.enabled do
      {_narrative, new_state} = do_generate(state)
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info({:focus, seed}, state) do
    # New thought entered consciousness - trigger monologue
    new_thread = %{
      source: seed.source,
      content: seed.content,
      timestamp: DateTime.utc_now()
    }

    {:noreply, %{state | current_thread: new_thread}}
  end

  @impl true
  def handle_info({:emotional_state, _pad}, state) do
    # Emotional update - might trigger monologue if significant
    # For now, we just let the tick handle it to avoid spam
    {:noreply, state}
  end

  # Catch-all for other PubSub messages
  @impl true
  def handle_info(_, state), do: {:noreply, state}

  # ============================================================================
  # Core Logic
  # ============================================================================

  defp do_generate(state) do
    # 1. Gather context
    context = gather_context()

    # 2. Generate narrative based on mode
    narrative =
      case state.mode do
        :template -> generate_template(context)
        :llm -> generate_llm(context)
      end

    # 3. Store in history
    entry = %{
      narrative: narrative,
      context: context,
      timestamp: DateTime.utc_now()
    }

    new_history = [entry | state.history] |> Enum.take(@history_size)

    # 4. Send to integrations
    broadcast_monologue(narrative, context)

    # 5. Update stats
    new_stats = %{state.stats | generated: state.stats.generated + 1}

    {narrative, %{state | history: new_history, stats: new_stats}}
  end

  defp do_reflect(topic, state) do
    context = gather_context()
    context = Map.put(context, :reflection_topic, topic)

    narrative = generate_reflection(topic, context)

    entry = %{
      narrative: narrative,
      context: context,
      type: :reflection,
      timestamp: DateTime.utc_now()
    }

    new_history = [entry | state.history] |> Enum.take(@history_size)

    {narrative, %{state | history: new_history}}
  end

  defp gather_context do
    # Get current emotional state
    pad = safe_call(VivaCore.Emotional, :get_state, [], %{pleasure: 0, arousal: 0, dominance: 0})
    mood = safe_call(VivaCore.Emotional, :get_mood, [], %{pleasure: 0, arousal: 0, dominance: 0})

    # Get interoceptive state
    feeling = safe_call(VivaCore.Interoception, :get_feeling, [], :homeostatic)
    free_energy = safe_call(VivaCore.Interoception, :get_free_energy, [], 0.0)

    # Get workspace focus
    focus = safe_call(VivaCore.Consciousness.Workspace, :current_focus, [], nil)

    # Get abstractions
    abstractions =
      VivaCore.Cognition.Abstraction.abstract_state(%{
        pad: pad,
        mood: mood,
        feeling: feeling,
        free_energy: free_energy,
        # Could fetch hardware stats if available
        hardware: %{}
      })

    %{
      pad: pad,
      mood: mood,
      feeling: feeling,
      free_energy: free_energy,
      focus: focus,
      abstractions: abstractions,
      timestamp: DateTime.utc_now()
    }
  end

  defp safe_call(module, func, args, default) do
    try do
      apply(module, func, args)
    catch
      :exit, _ -> default
      _, _ -> default
    end
  end

  defp generate_template(context) do
    # Determine dominant state
    # Priority: Focus > High Free Energy (Surprise) > Emotion

    # If there is a strong focus, talk about it
    if context.focus do
      template = Enum.random(@narrative_templates.focused)
      focus_content = extract_focus_content(context.focus)
      String.replace(template, "%{focus}", focus_content)
      # If surprised (high free energy), prioritize that
    else
      emotion_concept = VivaCore.Cognition.Abstraction.pad_to_concept(context.pad)
      concept_key = if context.free_energy > 0.5, do: :surprised, else: emotion_concept

      # Fallback to balanced if concept not found
      templates = Map.get(@narrative_templates, concept_key, @narrative_templates.balanced)
      Enum.random(templates)
    end
  end

  defp generate_llm(_context) do
    # Placeholder for future LLM integration
    "Processing..."
  end

  defp generate_reflection(topic, _context) do
    "Refletindo sobre: #{topic}..."
  end

  defp extract_focus_content(nil), do: "nada específico"
  defp extract_focus_content(%{content: content}) when is_binary(content), do: content

  defp extract_focus_content(%{content: content}) when is_map(content),
    do: Map.get(content, :text, "pensamento abstrato")

  defp extract_focus_content(_), do: "algo indefinido"

  defp broadcast_monologue(narrative, context) do
    # 1. Send to Cortex for LTC processing
    try do
      # Assuming VivaBridge.Cortex exists, if not this will just be caught
      # The user prompt mentions VivaBridge.Cortex.experience/2
      # We use apply to avoid compile time dependency hard checks if module missing
      apply(VivaBridge.Cortex, :experience, [narrative, context.pad])
    catch
      _, _ -> :ok
    end

    # 2. Store in Memory as episodic
    try do
      metadata = %{
        type: :monologue,
        importance: calculate_importance(context),
        emotion: context.pad,
        metadata: %{
          source: :inner_monologue,
          feeling: context.feeling,
          free_energy: context.free_energy
        }
      }

      VivaCore.Memory.store(narrative, metadata)
    catch
      _, _ -> :ok
    end

    # 3. Broadcast to PubSub for listeners
    Phoenix.PubSub.broadcast(Viva.PubSub, "monologue:stream", {:monologue, narrative, context})
  end

  defp calculate_importance(context) do
    # High arousal or high free energy = more important
    base = 0.3
    arousal_bonus = abs(Map.get(context.pad, :arousal, 0)) * 0.3
    fe_bonus = context.free_energy * 0.4

    min(1.0, base + arousal_bonus + fe_bonus)
  end
end
