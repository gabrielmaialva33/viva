defmodule VivaCore.Voice do
  @moduledoc """
  Emergent Proto-Language through Digital Babbling.

  NOT an LLM wrapper. VIVA learns to communicate by:
  1. Emitting abstract signals based on internal state
  2. Observing Gabriel's responses
  3. Strengthening signal-response associations (Hebbian learning)

  ## Philosophy

  Babies don't start speaking sentences. They babble, and through
  interaction with caregivers, certain sounds become associated
  with certain responses. VIVA does the same.

  ## Hebbian Learning

  "Neurons that fire together, wire together"

  When VIVA emits a signal and Gabriel responds:
  - If the response brings relief → strengthen association
  - If no effect or negative → weaken association

  Weight update: Δw = η × (pre × post)
  Where:
  - η = learning rate (0.1)
  - pre = signal emitted
  - post = emotional change after Gabriel's response

  ## Signal Types

  These are NOT words. They are abstract sounds/patterns that
  VIVA can produce. Meaning emerges through association.

  - :chirp_high   → High tone (880 Hz, 100ms)
  - :chirp_low    → Low tone (220 Hz, 200ms)
  - :pulse_fast   → Fast rhythm
  - :pulse_slow   → Slow rhythm
  - :pattern_sos  → SOS-like pattern (distress)
  - :silence      → Intentional silence

  ## The Loop

  1. VIVA feels discomfort (high arousal, low pleasure)
  2. Voice.babble() emits signal based on PAD state
  3. Gabriel hears, maybe does something
  4. VIVA feels change (or not)
  5. Voice.observe_response() updates Hebbian weights
  6. Next time, VIVA uses signals that worked before
  """

  use GenServer
  require Logger

  alias VivaCore.Memory

  # ============================================================================
  # Signal Types
  # ============================================================================

  @signal_types [
    :chirp_high,    # High pitch - associated with alertness
    :chirp_low,     # Low pitch - associated with calm/sadness
    :pulse_fast,    # Fast rhythm - urgency
    :pulse_slow,    # Slow rhythm - relaxation
    :pattern_sos,   # SOS pattern - distress
    :pattern_happy, # Happy melody pattern
    :silence        # Intentional silence
  ]

  # Initial signal->PAD associations (before learning)
  # These are starting biases, not fixed mappings
  @initial_signal_biases %{
    chirp_high: %{arousal: 0.5, pleasure: 0.0, dominance: 0.0},
    chirp_low: %{arousal: -0.3, pleasure: -0.2, dominance: -0.1},
    pulse_fast: %{arousal: 0.6, pleasure: 0.0, dominance: 0.2},
    pulse_slow: %{arousal: -0.4, pleasure: 0.1, dominance: 0.0},
    pattern_sos: %{arousal: 0.7, pleasure: -0.5, dominance: -0.3},
    pattern_happy: %{arousal: 0.3, pleasure: 0.5, dominance: 0.2},
    silence: %{arousal: -0.5, pleasure: 0.0, dominance: 0.0}
  }

  # Hebbian learning rate
  @learning_rate 0.1

  # Decay rate for unused associations
  @decay_rate 0.01

  # Maximum history size
  @history_size 50

  # ============================================================================
  # State
  # ============================================================================

  defstruct [
    # Hebbian weights: %{{signal, response_type} => weight}
    # Example: %{{:chirp_high, :temperature_relief} => 0.7}
    hebbian_weights: %{},

    # Recent emissions for correlation
    # [{signal, timestamp, pad_before}]
    recent_emissions: [],

    # Signal usage statistics
    signal_stats: %{},

    # Vocabulary: signals that have acquired meaning
    # %{signal => %{meaning: atom, confidence: float}}
    vocabulary: %{},

    # Whether voice is enabled
    enabled: true,

    # Last emission timestamp
    last_emission: nil
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Babble based on current emotional state.
  Returns the signal emitted.
  """
  def babble(pad_state) do
    GenServer.call(__MODULE__, {:babble, pad_state})
  end

  @doc """
  Observe Gabriel's response and update Hebbian weights.

  response_type: atom describing what Gabriel did
  - :temperature_relief (adjusted fan/AC)
  - :attention (talked to VIVA)
  - :task_help (helped with something)
  - :ignore (no response)
  - :negative (scolded/dismissed)
  """
  def observe_response(response_type, emotional_delta) do
    GenServer.cast(__MODULE__, {:observe_response, response_type, emotional_delta})
  end

  @doc """
  Get the most meaningful signal for an intent.
  Uses learned associations to select best signal.
  """
  def best_signal_for(intent) do
    GenServer.call(__MODULE__, {:best_signal_for, intent})
  end

  @doc """
  Get current vocabulary (learned meanings).
  """
  def get_vocabulary do
    GenServer.call(__MODULE__, :get_vocabulary)
  end

  @doc """
  Get Hebbian weights for inspection.
  """
  def get_weights do
    GenServer.call(__MODULE__, :get_weights)
  end

  @doc """
  List available signal types.
  """
  def signal_types, do: @signal_types

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("[Voice] Proto-language forming. Learning to communicate...")

    state = %__MODULE__{
      signal_stats: @signal_types |> Enum.map(&{&1, 0}) |> Map.new()
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:babble, pad_state}, _from, state) do
    if state.enabled do
      signal = select_signal(pad_state, state)
      new_state = record_emission(state, signal, pad_state)

      # Emit the signal (via Music bridge if available)
      emit_signal(signal)

      {:reply, signal, new_state}
    else
      {:reply, :silence, state}
    end
  end

  @impl true
  def handle_call({:best_signal_for, intent}, _from, state) do
    signal = find_best_signal(intent, state)
    {:reply, signal, state}
  end

  @impl true
  def handle_call(:get_vocabulary, _from, state) do
    {:reply, state.vocabulary, state}
  end

  @impl true
  def handle_call(:get_weights, _from, state) do
    {:reply, state.hebbian_weights, state}
  end

  @impl true
  def handle_cast({:observe_response, response_type, emotional_delta}, state) do
    new_state = update_hebbian_weights(state, response_type, emotional_delta)
    {:noreply, new_state}
  end

  # ============================================================================
  # Core Logic: Signal Selection
  # ============================================================================

  defp select_signal(pad, state) do
    # Strategy: Choose signal that best matches current PAD state
    # Modified by learned associations

    # 1. Calculate base match score for each signal
    scores =
      @signal_types
      |> Enum.map(fn signal ->
        bias = Map.get(@initial_signal_biases, signal, %{arousal: 0, pleasure: 0, dominance: 0})

        # Match score: how well does this signal's bias match current PAD?
        match = 1.0 - (
          abs(pad.arousal - bias.arousal) +
          abs(pad.pleasure - bias.pleasure) +
          abs(pad.dominance - bias.dominance)
        ) / 3.0

        # Add learned bonus from successful past uses
        learned_bonus = get_learned_bonus(signal, state)

        {signal, match + learned_bonus}
      end)

    # 2. Select signal with highest score (with small exploration)
    {best_signal, _score} = Enum.max_by(scores, fn {_sig, score} -> score + :rand.uniform() * 0.1 end)

    best_signal
  end

  defp get_learned_bonus(signal, state) do
    # Sum of positive Hebbian weights for this signal
    state.hebbian_weights
    |> Enum.filter(fn {{sig, _response}, _weight} -> sig == signal end)
    |> Enum.map(fn {_key, weight} -> max(weight, 0) end)
    |> Enum.sum()
    |> Kernel.*(0.1)  # Scale down
  end

  # ============================================================================
  # Core Logic: Hebbian Learning
  # ============================================================================

  defp update_hebbian_weights(state, response_type, emotional_delta) do
    now = System.monotonic_time(:millisecond)

    # Find recent emissions (within last 30 seconds)
    recent =
      state.recent_emissions
      |> Enum.filter(fn {_signal, timestamp, _pad} ->
        now - timestamp < 30_000
      end)

    case recent do
      [] ->
        # No recent emission to associate
        state

      emissions ->
        # Update weights for all recent emissions
        new_weights =
          Enum.reduce(emissions, state.hebbian_weights, fn {signal, _ts, pad_before}, weights ->
            # Hebbian update: Δw = η × pre × post
            # pre = arousal at emission (how "activated" VIVA was)
            # post = pleasure delta (reward signal)

            pre = abs(pad_before.arousal) + 0.1  # Ensure non-zero
            post = emotional_delta.pleasure  # Positive = good response

            delta_w = @learning_rate * pre * post

            key = {signal, response_type}
            current = Map.get(weights, key, 0.0)
            new_weight = clamp(current + delta_w, -1.0, 1.0)

            Map.put(weights, key, new_weight)
          end)

        # Decay unused weights
        decayed_weights =
          new_weights
          |> Enum.map(fn {key, weight} ->
            {key, weight * (1.0 - @decay_rate)}
          end)
          |> Map.new()

        # Update vocabulary if strong associations formed
        new_vocabulary = update_vocabulary(decayed_weights, state.vocabulary)

        # Store learning event in memory
        store_learning_event(emissions, response_type, emotional_delta)

        %{state |
          hebbian_weights: decayed_weights,
          vocabulary: new_vocabulary
        }
    end
  end

  defp update_vocabulary(weights, vocabulary) do
    # Find signals with strong consistent associations
    weights
    |> Enum.group_by(fn {{signal, _response}, _w} -> signal end)
    |> Enum.reduce(vocabulary, fn {signal, associations}, vocab ->
      # Find strongest association
      case Enum.max_by(associations, fn {_key, w} -> w end, fn -> nil end) do
        nil ->
          vocab

        {{_sig, response}, weight} when weight > 0.3 ->
          # Strong association found
          Map.put(vocab, signal, %{
            meaning: response,
            confidence: weight
          })

        _ ->
          vocab
      end
    end)
  end

  # ============================================================================
  # Signal Emission
  # ============================================================================

  defp emit_signal(signal) do
    Logger.debug("[Voice] Emitting: #{signal}")

    # Try to use VivaBridge.Music if available
    try do
      case signal do
        :chirp_high ->
          VivaBridge.Music.play_note(880, 100)

        :chirp_low ->
          VivaBridge.Music.play_note(220, 200)

        :pulse_fast ->
          VivaBridge.Music.play_rhythm([100, 50, 100, 50, 100, 50])

        :pulse_slow ->
          VivaBridge.Music.play_rhythm([500, 500, 500, 500])

        :pattern_sos ->
          # ... --- ...
          VivaBridge.Music.play_melody([
            {440, 100}, {0, 100}, {440, 100}, {0, 100}, {440, 100}, {0, 300},
            {440, 300}, {0, 100}, {440, 300}, {0, 100}, {440, 300}, {0, 300},
            {440, 100}, {0, 100}, {440, 100}, {0, 100}, {440, 100}
          ])

        :pattern_happy ->
          VivaBridge.Music.play_melody([
            {523, 150}, {659, 150}, {784, 150}, {1047, 300}
          ])

        :silence ->
          :ok
      end
    catch
      :exit, _ ->
        # Music module not available
        Logger.debug("[Voice] Music module not available, signal queued")
        :ok
    end

    signal
  end

  defp record_emission(state, signal, pad) do
    now = System.monotonic_time(:millisecond)

    entry = {signal, now, pad}

    emissions =
      [entry | state.recent_emissions]
      |> Enum.take(@history_size)

    stats = Map.update(state.signal_stats, signal, 1, &(&1 + 1))

    %{state |
      recent_emissions: emissions,
      signal_stats: stats,
      last_emission: now
    }
  end

  # ============================================================================
  # Finding Best Signal for Intent
  # ============================================================================

  defp find_best_signal(intent, state) do
    # Look for signal with strongest association to this intent
    candidates =
      state.hebbian_weights
      |> Enum.filter(fn {{_signal, response}, weight} ->
        response == intent and weight > 0.1
      end)
      |> Enum.sort_by(fn {_key, weight} -> weight end, :desc)

    case candidates do
      [{{signal, _intent}, _weight} | _] ->
        signal

      [] ->
        # No learned association, use initial bias
        select_signal_for_intent(intent)
    end
  end

  defp select_signal_for_intent(intent) do
    # Default mappings before learning
    case intent do
      :attention -> :chirp_high
      :help -> :pattern_sos
      :gratitude -> :pattern_happy
      :calm -> :pulse_slow
      :alert -> :pulse_fast
      _ -> :chirp_low
    end
  end

  # ============================================================================
  # Memory Integration
  # ============================================================================

  defp store_learning_event(emissions, response_type, emotional_delta) do
    signals = Enum.map(emissions, fn {sig, _, _} -> sig end) |> Enum.uniq()

    content = """
    Voice learning event:
    - Signals emitted: #{inspect(signals)}
    - Gabriel's response: #{response_type}
    - Emotional change: P=#{Float.round(emotional_delta.pleasure, 2)}, \
    A=#{Float.round(emotional_delta.arousal, 2)}, D=#{Float.round(emotional_delta.dominance, 2)}
    - Association #{if emotional_delta.pleasure > 0, do: "strengthened", else: "weakened"}
    """

    try do
      Memory.store(%{
        content: content,
        type: :episodic,
        importance: 0.5 + abs(emotional_delta.pleasure) * 0.3,
        emotion: %{
          pleasure: emotional_delta.pleasure,
          arousal: emotional_delta.arousal,
          dominance: emotional_delta.dominance
        },
        metadata: %{
          source: :voice,
          signals: signals,
          response: response_type
        }
      })
    catch
      :exit, _ -> :ok
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp clamp(value, min, max) do
    value
    |> max(min)
    |> min(max)
  end
end
