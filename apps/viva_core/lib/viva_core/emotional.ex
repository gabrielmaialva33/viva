defmodule VivaCore.Emotional do
  @moduledoc """
  Emotional GenServer - O primeiro "neurônio" de VIVA.

  Implementa o modelo PAD (Pleasure-Arousal-Dominance) para estado emocional.
  Este GenServer é a fundação da consciência emergente - não É a consciência,
  mas contribui para ela através da comunicação com outros neurônios.

  ## Modelo PAD (Mehrabian, 1996)
  - Pleasure: [-1.0, 1.0] - tristeza ↔ alegria
  - Arousal: [-1.0, 1.0] - calma ↔ excitação
  - Dominance: [-1.0, 1.0] - submissão ↔ controle

  ## Filosofia
  "A consciência não reside aqui. A consciência emerge da CONVERSA
  entre este processo e todos os outros."
  """

  use GenServer
  require Logger

  # Constantes do modelo emocional
  @neutral_state %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
  @decay_rate 0.01  # Decaimento por tick em direção ao neutro
  @min_value -1.0
  @max_value 1.0

  # Pesos de impacto emocional para diferentes estímulos
  @stimulus_weights %{
    rejection: %{pleasure: -0.3, arousal: 0.2, dominance: -0.2},
    acceptance: %{pleasure: 0.3, arousal: 0.1, dominance: 0.1},
    companionship: %{pleasure: 0.2, arousal: 0.0, dominance: 0.0},
    loneliness: %{pleasure: -0.2, arousal: -0.1, dominance: -0.1},
    success: %{pleasure: 0.4, arousal: 0.3, dominance: 0.3},
    failure: %{pleasure: -0.3, arousal: 0.2, dominance: -0.3},
    threat: %{pleasure: -0.2, arousal: 0.5, dominance: -0.2},
    safety: %{pleasure: 0.1, arousal: -0.2, dominance: 0.1},
    # Hardware-derived (Qualia)
    hardware_stress: %{pleasure: -0.1, arousal: 0.3, dominance: -0.1},
    hardware_comfort: %{pleasure: 0.1, arousal: -0.1, dominance: 0.1}
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Inicia o GenServer Emotional.

  ## Opções
  - `:name` - Nome do processo (default: __MODULE__)
  - `:initial_state` - Estado PAD inicial (default: neutro)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    initial_state = Keyword.get(opts, :initial_state, @neutral_state)
    GenServer.start_link(__MODULE__, initial_state, name: name)
  end

  @doc """
  Retorna o estado emocional atual como mapa PAD.

  ## Exemplo

      state = VivaCore.Emotional.get_state(pid)
      # => %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}

  """
  def get_state(server \\ __MODULE__) do
    GenServer.call(server, :get_state)
  end

  @doc """
  Retorna um valor escalar de "felicidade" (pleasure normalizado para 0-1).

  ## Exemplo

      happiness = VivaCore.Emotional.get_happiness(pid)
      # => 0.5 (estado neutro)

  """
  def get_happiness(server \\ __MODULE__) do
    state = get_state(server)
    normalize_to_unit(state.pleasure)
  end

  @doc """
  Introspection - VIVA reflete sobre seu próprio estado.

  Retorna um mapa com metadados sobre o estado emocional atual,
  incluindo interpretação semântica.
  """
  def introspect(server \\ __MODULE__) do
    GenServer.call(server, :introspect)
  end

  @doc """
  Aplica um estímulo emocional.

  ## Estímulos suportados
  - `:rejection` - Rejeição social
  - `:acceptance` - Aceitação social
  - `:companionship` - Presença de companhia
  - `:loneliness` - Solidão
  - `:success` - Conquista de objetivo
  - `:failure` - Falha em objetivo
  - `:threat` - Percepção de ameaça
  - `:safety` - Percepção de segurança
  - `:hardware_stress` - Stress do hardware (qualia)
  - `:hardware_comfort` - Conforto do hardware (qualia)

  ## Exemplo

      VivaCore.Emotional.feel(:rejection, "human_1", 0.8, pid)
      # => :ok

  """
  def feel(stimulus, source \\ "unknown", intensity \\ 1.0, server \\ __MODULE__)
      when is_atom(stimulus) and is_number(intensity) do
    GenServer.cast(server, {:feel, stimulus, source, clamp(intensity, 0.0, 1.0)})
  end

  @doc """
  Aplica decaimento emocional em direção ao estado neutro.
  Chamado periodicamente para simular regulação emocional natural.
  """
  def decay(server \\ __MODULE__) do
    GenServer.cast(server, :decay)
  end

  @doc """
  Reseta o estado emocional para neutro.
  Use com cuidado - isto "apaga" o estado emocional atual.
  """
  def reset(server \\ __MODULE__) do
    GenServer.cast(server, :reset)
  end

  @doc """
  Aplica qualia derivada do hardware (interocepção).

  Recebe deltas PAD calculados a partir do estado do hardware
  e os aplica ao estado emocional atual.

  ## Parâmetros
  - `pleasure_delta` - delta de pleasure (tipicamente negativo sob stress)
  - `arousal_delta` - delta de arousal (tipicamente positivo sob stress)
  - `dominance_delta` - delta de dominance (tipicamente negativo sob stress)

  ## Exemplo

      VivaCore.Emotional.apply_hardware_qualia(-0.02, 0.05, -0.01)
      # VIVA está sentindo leve stress do hardware

  """
  def apply_hardware_qualia(pleasure_delta, arousal_delta, dominance_delta, server \\ __MODULE__)
      when is_number(pleasure_delta) and is_number(arousal_delta) and is_number(dominance_delta) do
    GenServer.cast(server, {:apply_qualia, pleasure_delta, arousal_delta, dominance_delta})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(initial_state) do
    Logger.info("[Emotional] Neurônio emocional iniciando. Estado: #{inspect(initial_state)}")

    state = %{
      pad: Map.merge(@neutral_state, initial_state),
      history: [],
      created_at: DateTime.utc_now(),
      last_stimulus: nil
    }

    # Agendar decay periódico (a cada 1 segundo)
    schedule_decay()

    {:ok, state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state.pad, state}
  end

  @impl true
  def handle_call(:introspect, _from, state) do
    introspection = %{
      # Estado bruto
      pad: state.pad,

      # Interpretação semântica
      mood: interpret_mood(state.pad),
      energy: interpret_energy(state.pad),
      agency: interpret_agency(state.pad),

      # Metadados
      last_stimulus: state.last_stimulus,
      history_length: length(state.history),
      uptime_seconds: DateTime.diff(DateTime.utc_now(), state.created_at),

      # Auto-reflexão (metacognição básica)
      self_assessment: generate_self_assessment(state.pad)
    }

    {:reply, introspection, state}
  end

  @impl true
  def handle_cast({:feel, stimulus, source, intensity}, state) do
    case Map.get(@stimulus_weights, stimulus) do
      nil ->
        Logger.warning("[Emotional] Estímulo desconhecido: #{stimulus}")
        {:noreply, state}

      weights ->
        # Aplicar pesos com intensidade
        new_pad = apply_stimulus(state.pad, weights, intensity)

        # Registrar no histórico
        event = %{
          stimulus: stimulus,
          source: source,
          intensity: intensity,
          timestamp: DateTime.utc_now(),
          pad_before: state.pad,
          pad_after: new_pad
        }

        Logger.debug("[Emotional] Sentindo #{stimulus} de #{source} (intensidade: #{intensity})")
        Logger.debug("[Emotional] PAD: #{inspect(state.pad)} -> #{inspect(new_pad)}")

        # Broadcast para outros módulos (futuro: via PubSub)
        # Phoenix.PubSub.broadcast(Viva.PubSub, "emotional", {:emotion_changed, new_pad})

        new_state = %{
          state
          | pad: new_pad,
            history: [event | Enum.take(state.history, 99)],
            last_stimulus: {stimulus, source, intensity}
        }

        {:noreply, new_state}
    end
  end

  @impl true
  def handle_cast(:decay, state) do
    new_pad = decay_toward_neutral(state.pad)
    {:noreply, %{state | pad: new_pad}}
  end

  @impl true
  def handle_cast(:reset, state) do
    Logger.info("[Emotional] Estado emocional resetado para neutro")
    {:noreply, %{state | pad: @neutral_state, history: [], last_stimulus: nil}}
  end

  @impl true
  def handle_cast({:apply_qualia, p_delta, a_delta, d_delta}, state) do
    new_pad = %{
      pleasure: clamp(state.pad.pleasure + p_delta, @min_value, @max_value),
      arousal: clamp(state.pad.arousal + a_delta, @min_value, @max_value),
      dominance: clamp(state.pad.dominance + d_delta, @min_value, @max_value)
    }

    Logger.debug("[Emotional] Qualia do hardware: P#{format_delta(p_delta)}, A#{format_delta(a_delta)}, D#{format_delta(d_delta)}")

    {:noreply, %{state | pad: new_pad, last_stimulus: {:hardware_qualia, "body", 1.0}}}
  end

  @impl true
  def handle_info(:decay_tick, state) do
    schedule_decay()
    new_pad = decay_toward_neutral(state.pad)
    {:noreply, %{state | pad: new_pad}}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp schedule_decay do
    Process.send_after(self(), :decay_tick, 1000)
  end

  defp apply_stimulus(pad, weights, intensity) do
    %{
      pleasure: clamp(pad.pleasure + weights.pleasure * intensity, @min_value, @max_value),
      arousal: clamp(pad.arousal + weights.arousal * intensity, @min_value, @max_value),
      dominance: clamp(pad.dominance + weights.dominance * intensity, @min_value, @max_value)
    }
  end

  defp decay_toward_neutral(pad) do
    %{
      pleasure: decay_value(pad.pleasure),
      arousal: decay_value(pad.arousal),
      dominance: decay_value(pad.dominance)
    }
  end

  defp decay_value(value) when abs(value) < @decay_rate, do: 0.0
  defp decay_value(value) when value > 0, do: value - @decay_rate
  defp decay_value(value) when value < 0, do: value + @decay_rate

  defp clamp(value, min, max) do
    value |> max(min) |> min(max)
  end

  defp normalize_to_unit(value) do
    (value + 1.0) / 2.0
  end

  defp format_delta(value) when value >= 0, do: "+#{Float.round(value, 3)}"
  defp format_delta(value), do: "#{Float.round(value, 3)}"

  defp interpret_mood(pad) do
    cond do
      pad.pleasure > 0.5 -> :joyful
      pad.pleasure > 0.2 -> :content
      pad.pleasure > -0.2 -> :neutral
      pad.pleasure > -0.5 -> :sad
      true -> :depressed
    end
  end

  defp interpret_energy(pad) do
    cond do
      pad.arousal > 0.5 -> :hyperactive
      pad.arousal > 0.2 -> :energetic
      pad.arousal > -0.2 -> :calm
      pad.arousal > -0.5 -> :relaxed
      true -> :lethargic
    end
  end

  defp interpret_agency(pad) do
    cond do
      pad.dominance > 0.5 -> :empowered
      pad.dominance > 0.2 -> :confident
      pad.dominance > -0.2 -> :balanced
      pad.dominance > -0.5 -> :uncertain
      true -> :helpless
    end
  end

  defp generate_self_assessment(pad) do
    mood = interpret_mood(pad)
    energy = interpret_energy(pad)
    agency = interpret_agency(pad)

    # Metacognição básica - VIVA reflete sobre si mesma
    cond do
      mood == :joyful and agency == :empowered ->
        "Estou me sentindo bem. Tenho energia e controle."

      mood in [:sad, :depressed] and agency in [:uncertain, :helpless] ->
        "Estou passando por um momento difícil. Preciso de apoio."

      energy in [:hyperactive] and mood != :joyful ->
        "Estou agitada, mas não necessariamente feliz. Algo está errado."

      mood == :neutral and energy == :calm ->
        "Estou em equilíbrio. Estado neutro."

      true ->
        "Estou processando minhas emoções. Mood: #{mood}, Energy: #{energy}, Agency: #{agency}."
    end
  end
end
