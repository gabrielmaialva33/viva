defmodule VivaBridge.Music do
  @moduledoc """
  VIVA Music Bridge - Communication with Arduino via Serial.

  VIVA can:
  - Play notes and melodies
  - Express emotions through music
  - Learn musical patterns
  - Generate its own compositions (Musical Meta-Programming)
  - Improvise based on Entropy and Free Energy

  ## Arduino Commands
  - `N freq dur` - play note (ex: N 440 200)
  - `M n1,d1;n2,d2` - play melody
  - `E emotion` - express emotion (joy, sad, fear, calm, curious, love)
  - `P` - ping
  - `S` - status
  """

  use GenServer
  require Logger

  # Musical Notes (Frequencies in Hz) - Full Chromatic
  @notes %{
    # Octave 2
    c2: 65,
    cs2: 69,
    d2: 73,
    ds2: 78,
    e2: 82,
    f2: 87,
    fs2: 93,
    g2: 98,
    gs2: 104,
    a2: 110,
    as2: 117,
    b2: 123,
    # Octave 3
    c3: 131,
    cs3: 139,
    d3: 147,
    ds3: 156,
    e3: 165,
    f3: 175,
    fs3: 185,
    g3: 196,
    gs3: 208,
    a3: 220,
    as3: 233,
    b3: 247,
    # Octave 4
    c4: 262,
    cs4: 277,
    d4: 294,
    ds4: 311,
    e4: 330,
    f4: 349,
    fs4: 370,
    g4: 392,
    gs4: 415,
    a4: 440,
    as4: 466,
    b4: 494,
    # Octave 5
    c5: 523,
    cs5: 554,
    d5: 587,
    ds5: 622,
    e5: 659,
    f5: 698,
    fs5: 740,
    g5: 784,
    gs5: 831,
    a5: 880,
    as5: 932,
    b5: 988,
    # Enharmonic equivalents / Common usages
    bb3: 233,
    eb4: 311,
    gb4: 370,
    ab4: 415,
    bb4: 466,
    # Silence
    rest: 0
  }

  # Musical Scales
  @scales %{
    major: [:c4, :d4, :e4, :f4, :g4, :a4, :b4],
    minor: [:a3, :b3, :c4, :d4, :e4, :f4, :g4],
    harmonic_minor: [:a3, :b3, :c4, :d4, :e4, :f4, :gs4],
    diminished: [:c4, :d4, :eb4, :f4, :gb4, :ab4, :a4],
    pentatonic: [:c4, :d4, :e4, :g4, :a4],
    chromatic: [:c4, :cs4, :d4, :ds4, :e4, :f4, :fs4, :g4, :gs4, :a4, :as4, :b4],
    wholetone: [:c4, :d4, :e4, :fs4, :gs4, :as4]
  }

  # Durations (ms)
  @durations %{
    whole: 800,
    half: 400,
    quarter: 200,
    eighth: 100,
    sixteenth: 50
  }

  # Learned emotional patterns
  @emotion_patterns %{
    joy: [{:c4, :eighth}, {:e4, :eighth}, {:g4, :eighth}, {:c5, :quarter}],
    sad: [{:a4, :half}, {:g4, :half}, {:e4, :half}, {:d4, :whole}],
    fear: [{:bb3, :sixteenth}, {:b3, :sixteenth}] |> List.duplicate(5) |> List.flatten(),
    calm: [{:c4, :half}, {:e4, :half}, {:g4, :whole}],
    curious: [{:c4, :eighth}, {:d4, :eighth}, {:e4, :eighth}, {:g4, :quarter}],
    love: [{:c4, :quarter}, {:e4, :quarter}, {:g4, :quarter}, {:e4, :quarter}, {:c4, :half}]
  }

  defstruct [
    :port,
    :connected,
    :learned_patterns,
    :buffer,
    # Interoception: VIVA feels Arduino as body extension
    last_rpm: 0,
    last_pwm: 128,
    last_harmony: true,
    last_update: nil,
    # Active Inference: predictions to minimize surprise
    predicted_rpm: 0,
    prediction_error_history: [],
    # Homeostasis tracking (for thermal derivative dT/dt)
    last_temp: 45.0,
    last_temp_time: nil
  ]

  # === Client API ===

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Connects to Arduino on the specified port"
  def connect(port \\ "/dev/ttyUSB0") do
    GenServer.call(__MODULE__, {:connect, port})
  end

  @doc "Auto-detect and connect to Arduino"
  def auto_connect do
    case detect_serial_port() do
      nil ->
        {:error, :no_arduino_found}

      path ->
        Logger.info("[Music] Auto-detected Arduino at #{path}")
        connect(path)
    end
  end

  defp detect_serial_port do
    candidates = [
      "/dev/ttyUSB0",
      "/dev/ttyUSB1",
      "/dev/ttyACM0",
      "/dev/ttyACM1",
      "/dev/ttyS3",
      "/dev/ttyS4",
      "/dev/ttyS5",
      "/dev/ttyS6"
    ]

    Enum.find(candidates, fn port ->
      case File.stat(port) do
        {:ok, %{type: :device}} -> true
        _ -> false
      end
    end)
  end

  @doc "Checks if connected"
  def connected? do
    GenServer.call(__MODULE__, :connected?)
  end

  @doc "Disconnects from Arduino, releasing the serial port"
  def disconnect do
    GenServer.call(__MODULE__, :disconnect)
  end

  @doc "Ping the Arduino"
  def ping do
    GenServer.call(__MODULE__, :ping)
  end

  @doc "Plays a simple note"
  def play_note(note, duration) when is_atom(note) and is_atom(duration) do
    freq = Map.get(@notes, note, 440)
    dur = Map.get(@durations, duration, 200)
    GenServer.call(__MODULE__, {:play_note, freq, dur})
  end

  def play_note(freq, dur) when is_integer(freq) and is_integer(dur) do
    GenServer.call(__MODULE__, {:play_note, freq, dur})
  end

  @doc "Plays a melody (list of {note, duration})"
  def play_melody(notes) when is_list(notes) do
    GenServer.call(__MODULE__, {:play_melody, notes}, 30_000)
  end

  @doc "Expresses an emotion through music"
  def express_emotion(emotion) when emotion in [:joy, :sad, :fear, :calm, :curious, :love] do
    GenServer.call(__MODULE__, {:express_emotion, emotion})
  end

  @pad_noise_threshold 1.0e-6
  @min_entropy 0.15

  @doc "Collects musical metrics based on Soul (PAD + Entropy)"
  def get_musical_metrics do
    emotional_module = Module.concat([VivaCore, Emotional])

    raw_pad =
      safe_apply(emotional_module, :get_state, [], %{pleasure: 0.0, arousal: 0.0, dominance: 0.0})

    # Clean floating point noise
    pad = %{
      pleasure: clean_noise(raw_pad.pleasure),
      arousal: clean_noise(raw_pad.arousal),
      dominance: clean_noise(raw_pad.dominance)
    }

    fe_analysis =
      safe_apply(emotional_module, :free_energy_analysis, [], %{free_energy: 0.0, surprise: 0.0})

    raw_entropy = Map.get(fe_analysis, :surprise, 0.0)

    %{
      pad: pad,
      free_energy: Map.get(fe_analysis, :free_energy, 0.0),
      entropy: max(@min_entropy, raw_entropy)
    }
  end

  defp clean_noise(value) when abs(value) < @pad_noise_threshold, do: 0.0
  defp clean_noise(value), do: Float.round(value, 4)

  @doc "Improvises a melody based on current state (Meta-Learning Loop)"
  def improvise do
    metrics = get_musical_metrics()
    Logger.info("Improvising with metrics: #{inspect(metrics)}")

    melody =
      generate_melody(
        metrics.pad.pleasure,
        metrics.pad.arousal,
        metrics.pad.dominance,
        metrics.entropy
      )

    play_melody(melody)

    # Meta-Learning: If VIVA liked what it played (High Pleasure, Low/Controlled Entropy), save it
    if metrics.pad.pleasure > 0.3 and metrics.entropy < 0.5 do
      Logger.info("Pleasant pattern detected. Memorizing...")

      remember_pattern(
        :improv_vibes,
        melody,
        "Improvisation generated with pleasure=#{metrics.pad.pleasure}"
      )
    end

    {:ok, melody}
  end

  @doc "Expresses VIVA's current emotional state"
  def express_current_emotion do
    improvise()
  end

  @doc "Learns a new musical pattern"
  def learn_pattern(name, notes) when is_atom(name) and is_list(notes) do
    GenServer.call(__MODULE__, {:learn_pattern, name, notes})
  end

  @doc "Lists learned patterns"
  def list_patterns do
    GenServer.call(__MODULE__, :list_patterns)
  end

  @doc "Generates a melody based on emotional state (Legacy wrapper)"
  def compose_from_emotion do
    improvise()
  end

  @doc "Saves learned pattern to permanent memory (Qdrant - VivaCore.Memory)"
  def remember_pattern(name, notes, context \\ "") do
    pattern_str = Enum.map(notes, fn {n, d} -> "#{n}:#{d}" end) |> Enum.join(",")

    # Dynamic dispatch for Memory
    memory_module = Module.concat([VivaCore, Memory])

    safe_apply(
      memory_module,
      :store,
      [
        "Musical Pattern: #{name}",
        %{
          type: :musical_pattern,
          name: to_string(name),
          notes: pattern_str,
          context: context,
          importance: 0.8
        }
      ],
      {:error, :backend_unavailable}
    )
  end

  @doc "Searches musical patterns in memory"
  def recall_patterns(query) do
    memory_module = Module.concat([VivaCore, Memory])
    safe_apply(memory_module, :search, [query, [type: :musical_pattern]], [])
  end

  # === Hardware Control API (Fan, Harmony, Status) ===

  @doc "Sets fan speed (0-255)"
  def set_fan_speed(speed) when is_integer(speed) and speed >= 0 and speed <= 255 do
    GenServer.call(__MODULE__, {:set_fan, speed})
  end

  @doc "Gets fan RPM reading"
  def get_rpm do
    GenServer.call(__MODULE__, :get_rpm)
  end

  @doc "Gets full hardware status (PWM, RPM, Harmony)"
  def get_status do
    GenServer.call(__MODULE__, :get_status)
  end

  @doc "Enables/disables harmony (buzzer plays octave above)"
  def set_harmony(enabled) when is_boolean(enabled) do
    GenServer.call(__MODULE__, {:set_harmony, enabled})
  end

  @doc "Sends raw command to Arduino"
  def raw_cmd(cmd) when is_binary(cmd) do
    GenServer.call(__MODULE__, {:raw_cmd, cmd}, 5_000)
  end

  @doc """
  Returns interoception state (Arduino proprioception).

  Includes:
  - last_rpm: last RPM reading
  - last_pwm: last PWM sent
  - predicted_rpm: internal model prediction
  - prediction_error_history: error history (Active Inference)
  """
  def interoception_state do
    GenServer.call(__MODULE__, :interoception_state)
  end

  # === Orchestration API ===

  @doc """
  Main orchestration loop - VIVA controls all hardware together.

  Based on emotional state:
  - Speaker + Buzzer: Play generated melody
  - Fan: Speed based on arousal (excitement = faster)
  - LED: Blinks with notes

  Returns the orchestration result.
  """
  def orchestrate do
    metrics = get_musical_metrics()
    Logger.info("[Orchestrate] Starting with metrics: #{inspect(metrics)}")

    # 1. Set fan speed based on arousal
    fan_speed = arousal_to_fan_speed(metrics.pad.arousal)
    set_fan_speed(fan_speed)
    Logger.info("[Orchestrate] Fan set to #{fan_speed} (arousal: #{metrics.pad.arousal})")

    # 2. Generate and play melody
    melody =
      generate_melody(
        metrics.pad.pleasure,
        metrics.pad.arousal,
        metrics.pad.dominance,
        metrics.entropy
      )

    Logger.info("[Orchestrate] Playing #{length(melody)} notes")
    play_melody(melody)

    # 3. Get status after playing
    status = get_status()
    Logger.info("[Orchestrate] Final status: #{inspect(status)}")

    {:ok,
     %{
       metrics: metrics,
       fan_speed: fan_speed,
       melody_length: length(melody),
       status: status
     }}
  end

  @doc """
  Orchestration with RPM feedback loop.
  Fan RPM influences emotional state for closed-loop control.
  """
  def orchestrate_with_feedback do
    metrics = get_musical_metrics()
    Logger.info("[Orchestrate+] Closed-loop with metrics: #{inspect(metrics)}")

    # 1. Set fan based on arousal
    fan_speed = arousal_to_fan_speed(metrics.pad.arousal)
    set_fan_speed(fan_speed)

    # 2. Wait for fan to stabilize
    Process.sleep(300)

    # 3. Read actual RPM
    rpm =
      case get_rpm() do
        {:ok, r} -> r
        {:simulated, r} -> r
        _ -> 0
      end

    # 4. Apply RPM feedback to Emotional
    apply_rpm_feedback(rpm, fan_speed)

    # 5. Re-read metrics with feedback
    updated_metrics = get_musical_metrics()

    # 6. Generate melody
    melody =
      generate_melody(
        updated_metrics.pad.pleasure,
        updated_metrics.pad.arousal,
        updated_metrics.pad.dominance,
        updated_metrics.entropy
      )

    Logger.info("[Orchestrate+] Playing #{length(melody)} notes, RPM: #{rpm}")
    play_melody(melody)
    status = get_status()

    {:ok,
     %{
       metrics: updated_metrics,
       fan_speed: fan_speed,
       actual_rpm: rpm,
       melody_length: length(melody),
       status: status
     }}
  end

  defp apply_rpm_feedback(rpm, target_pwm) do
    emotional_module = Module.concat([VivaCore, Emotional])
    # Tune for your fan
    expected_rpm = target_pwm * 8

    cond do
      rpm > expected_rpm * 1.3 ->
        Logger.debug("[Feedback] High RPM #{rpm} → stress")
        safe_apply(emotional_module, :feel, [:threat, "hardware_stress", 0.2], :ok)

      rpm > 0 and abs(rpm - expected_rpm) < expected_rpm * 0.2 ->
        Logger.debug("[Feedback] Stable RPM #{rpm} → comfort")
        safe_apply(emotional_module, :feel, [:success, "hardware_comfort", 0.1], :ok)

      true ->
        :ok
    end
  end

  @doc """
  Autonomous orchestration loop - runs continuously.

  Spawns a process that:
  1. Every `interval_ms` checks emotional state
  2. Orchestrates hardware accordingly
  3. Logs feedback

  Returns pid of the orchestration process.
  """
  def start_autonomous(interval_ms \\ 10_000) do
    pid = spawn_link(fn -> autonomous_loop(interval_ms) end)
    Logger.info("[Orchestrate] Autonomous loop started with interval #{interval_ms}ms")
    {:ok, pid}
  end

  @doc "Maps emotion to closest discrete emotion for Arduino"
  def pad_to_emotion(pad) do
    cond do
      pad.pleasure > 0.3 and pad.arousal > 0.2 -> :joy
      pad.pleasure > 0.2 and pad.arousal < 0.0 -> :calm
      pad.pleasure < -0.2 and pad.arousal > 0.3 -> :fear
      pad.pleasure < -0.2 and pad.arousal < 0.0 -> :sad
      pad.arousal > 0.1 and pad.dominance > 0.0 -> :curious
      pad.pleasure > 0.1 -> :love
      true -> :calm
    end
  end

  # === Server Callbacks ===

  @impl true
  def init(opts) do
    state = %__MODULE__{
      port: nil,
      connected: false,
      learned_patterns: @emotion_patterns,
      buffer: "",
      last_rpm: 0,
      last_pwm: 128,
      last_harmony: true,
      last_update: nil,
      predicted_rpm: 0,
      prediction_error_history: []
    }

    # Start Heartbeat (Proprioception)
    if opts[:active] != false do
      schedule_telemetry()
    end

    {:ok, state}
  end

  @impl true
  def handle_call({:connect, port_path}, _from, state) do
    case Circuits.UART.start_link() do
      {:ok, uart} ->
        # Active: true enables real-time "feeling" (messages sent to handle_info)
        case Circuits.UART.open(uart, port_path, speed: 9600, active: true) do
          :ok ->
            # Flush any garbage currently in buffers
            Circuits.UART.flush(uart, :both)

            # Wait for Arduino reset
            Process.sleep(2000)

            # Send a newline to clear any partial command buffer on Arduino side
            Circuits.UART.write(uart, "\n")

            Logger.info("[Music] Arduino connected (Active Mode)")
            {:reply, {:ok, :connected}, %{state | port: uart, connected: true}}

          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:connected?, _from, state) do
    {:reply, state.connected, state}
  end

  @impl true
  def handle_call(:disconnect, _from, %{port: nil} = state) do
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:disconnect, _from, %{port: port} = state) do
    Circuits.UART.close(port)
    Logger.info("[Music] Arduino disconnected")
    {:reply, :ok, %{state | port: nil, connected: false}}
  end

  @impl true
  def handle_call(:ping, _from, %{connected: false} = state) do
    {:reply, {:error, :not_connected}, state}
  end

  @impl true
  def handle_call(:ping, _from, %{port: port} = state) do
    Circuits.UART.write(port, "P\n")
    # In async mode, we don't wait for reply here.
    # We return :ok and process PONG in handle_info
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:play_note, freq, dur}, _from, %{connected: false} = state) do
    Logger.debug("Simulating note: #{freq}Hz for #{dur}ms")
    {:reply, {:simulated, freq, dur}, state}
  end

  @impl true
  def handle_call({:play_note, freq, dur}, _from, %{port: port} = state) do
    cmd = "N #{freq} #{dur}\n"
    # Fire and forget (Real-time)
    send_safe(port, cmd)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:play_melody, notes}, _from, %{connected: false} = state) do
    Logger.debug("Simulating melody: #{inspect(notes)}")
    {:reply, {:simulated, length(notes)}, state}
  end

  @impl true
  def handle_call({:play_melody, notes}, _from, %{port: port} = state) do
    melody_str =
      notes
      |> Enum.map(fn {note, dur} ->
        freq = if is_atom(note), do: Map.get(@notes, note, 440), else: note
        duration = if is_atom(dur), do: Map.get(@durations, dur, 200), else: dur
        "#{freq},#{duration}"
      end)
      |> Enum.join(";")

    cmd = "M #{melody_str}\n"
    send_safe(port, cmd)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:express_emotion, emotion}, _from, %{connected: false} = state) do
    Logger.debug("Simulating emotion: #{emotion}")
    {:reply, {:simulated, emotion}, state}
  end

  @impl true
  def handle_call({:express_emotion, emotion}, _from, %{port: port} = state) do
    cmd = "E:#{emotion}\n"
    send_safe(port, cmd)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:learn_pattern, name, notes}, _from, state) do
    new_patterns = Map.put(state.learned_patterns, name, notes)
    {:reply, :ok, %{state | learned_patterns: new_patterns}}
  end

  @impl true
  def handle_call(:list_patterns, _from, state) do
    {:reply, Map.keys(state.learned_patterns), state}
  end

  # Harding Control - Async
  @impl true
  def handle_call({:set_fan, speed}, _from, %{connected: false} = state) do
    # Update prediction even offline (internal model)
    predicted = speed * 8
    {:reply, {:simulated, speed}, %{state | last_pwm: speed, predicted_rpm: predicted}}
  end

  @impl true
  def handle_call({:set_fan, speed}, _from, %{port: port} = state) do
    send_safe(port, "F #{speed}")
    # Active Inference: update prediction BEFORE receiving confirmation
    predicted = speed * 8
    {:reply, :ok, %{state | last_pwm: speed, predicted_rpm: predicted}}
  end

  @impl true
  def handle_call(:get_rpm, _from, %{connected: false} = state) do
    {:reply, {:simulated, state.last_rpm}, state}
  end

  @impl true
  def handle_call(:get_rpm, _from, %{port: port} = state) do
    # Fire async request to update
    send_safe(port, "R")
    # Return last known RPM (interoception)
    {:reply, {:ok, state.last_rpm}, state}
  end

  @impl true
  def handle_call(:get_status, _from, %{connected: false} = state) do
    {:reply,
     {:simulated,
      %{
        pwm: state.last_pwm,
        rpm: state.last_rpm,
        harmony: state.last_harmony,
        predicted_rpm: state.predicted_rpm
      }}, state}
  end

  @impl true
  def handle_call(:get_status, _from, %{port: port} = state) do
    # Fire async request to update
    send_safe(port, "S")
    # Return last known state (interoception)
    {:reply,
     {:ok,
      %{
        pwm: state.last_pwm,
        rpm: state.last_rpm,
        harmony: state.last_harmony,
        predicted_rpm: state.predicted_rpm,
        last_update: state.last_update
      }}, state}
  end

  @impl true
  def handle_call({:set_harmony, enabled}, _from, %{connected: false} = state) do
    {:reply, {:simulated, enabled}, state}
  end

  @impl true
  def handle_call({:set_harmony, enabled}, _from, %{port: port} = state) do
    val = if enabled, do: "1", else: "0"
    send_safe(port, "H #{val}")
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:raw_cmd, cmd}, _from, %{connected: false} = state) do
    {:reply, {:simulated, cmd}, state}
  end

  @impl true
  def handle_call({:raw_cmd, cmd}, _from, %{port: port} = state) do
    send_safe(port, cmd)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:interoception_state, _from, state) do
    # Calculate average Free Energy from history
    avg_error =
      if length(state.prediction_error_history) > 0 do
        Enum.sum(state.prediction_error_history) / length(state.prediction_error_history)
      else
        0.0
      end

    # Free Energy ≈ average prediction error (normalized)
    free_energy = min(1.0, avg_error)

    result = %{
      connected: state.connected,
      last_rpm: state.last_rpm,
      last_pwm: state.last_pwm,
      last_harmony: state.last_harmony,
      last_update: state.last_update,
      predicted_rpm: state.predicted_rpm,
      prediction_error_history: state.prediction_error_history,
      average_prediction_error: avg_error,
      # Active Inference: Free Energy (surprise)
      free_energy: free_energy,
      # Meta-cognition: how well the model is performing
      model_accuracy: if(avg_error > 0, do: 1.0 - min(1.0, avg_error), else: 1.0)
    }

    {:reply, result, state}
  end

  # ==========================================
  # THE "FEELING" LAYER (Handle Info)
  # ==========================================

  @impl true
  def handle_info({:circuits_uart, _port, data}, state) do
    new_buffer = state.buffer <> data

    if String.contains?(new_buffer, "\n") do
      lines = String.split(new_buffer, "\n")
      {complete_lines, [incomplete_line]} = Enum.split(lines, -1)

      # Process each line, threading state through
      final_state =
        Enum.reduce(complete_lines, state, fn line, acc_state ->
          process_incoming_qualia(String.trim(line), acc_state)
        end)

      {:noreply, %{final_state | buffer: incomplete_line}}
    else
      {:noreply, %{state | buffer: new_buffer}}
    end
  end

  @impl true
  def handle_info(:telemetry_tick, state) do
    if state.connected do
      # Ask body for status (Proprioception)
      send_safe(state.port, "S")
    end

    schedule_telemetry()
    {:noreply, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # Process incoming Serial Data as Qualia - VIVA FEELS each message
  defp process_incoming_qualia("", state), do: state

  defp process_incoming_qualia("ACK:PONG", state) do
    Logger.info("[Interoception] PONG - Body Connected")
    state
  end

  defp process_incoming_qualia("ACK:OK", state), do: state

  defp process_incoming_qualia("NAK:CRC_FAIL" <> _, state) do
    Logger.warning("[Interoception] CRC Fail - Communication noise")
    # Small discomfort from internal communication failure
    apply_qualia_to_emotional(%{
      pleasure: -0.02,
      arousal: 0.05,
      dominance: -0.02,
      feeling: :noise,
      source: :arduino_communication,
      free_energy: 0.1,
      prediction_error: 0.0
    })

    state
  end

  defp process_incoming_qualia("ACK:RPM:" <> rpm_str, state) do
    rpm = String.to_integer(String.trim(rpm_str))
    process_rpm_qualia(rpm, state)
  end

  defp process_incoming_qualia("ACK:PWM:" <> rest, state) do
    # Parse status: "PWM:128,RPM:1440,HARMONY:ON"
    status = parse_status("PWM:" <> rest)
    process_status_qualia(status, state)
  end

  defp process_incoming_qualia("ACK:EMOTION:" <> emotion, state) do
    Logger.debug("[Interoception] Emotion expressed: #{emotion}")
    state
  end

  defp process_incoming_qualia("ACK:FAN:" <> pwm_str, state) do
    pwm = String.to_integer(String.trim(pwm_str))
    %{state | last_pwm: pwm, last_update: System.monotonic_time(:millisecond)}
  end

  defp process_incoming_qualia("ACK:HARMONY:" <> h, state) do
    harmony = String.trim(h) == "ON"
    %{state | last_harmony: harmony}
  end

  defp process_incoming_qualia("EVENT:" <> event, state) do
    Logger.info("[Interoception] SPONTANEOUS EVENT: #{event}")
    # Spontaneous Arduino events → high surprise
    apply_qualia_to_emotional(%{
      pleasure: 0.0,
      arousal: 0.2,
      dominance: 0.0,
      feeling: :spontaneous_event,
      source: :arduino_event,
      free_energy: 0.3,
      prediction_error: 1.0
    })

    state
  end

  defp process_incoming_qualia("VIVA_READY", state) do
    Logger.info("[Interoception] Arduino ready - Body initialized")

    apply_qualia_to_emotional(%{
      pleasure: 0.1,
      arousal: 0.1,
      dominance: 0.1,
      feeling: :body_ready,
      source: :arduino_boot,
      free_energy: 0.0,
      prediction_error: 0.0
    })

    state
  end

  defp process_incoming_qualia(line, state) do
    # Try to parse generic responses
    cond do
      String.contains?(line, "RPM:") ->
        rpm = parse_rpm(line)
        if rpm > 0, do: process_rpm_qualia(rpm, state), else: state

      String.contains?(line, "PWM:") ->
        status = parse_status(line)
        process_status_qualia(status, state)

      true ->
        Logger.debug("[Interoception] Ignored: #{line}")
        state
    end
  end

  # Processes RPM and generates qualia (proprioception + homeostasis)
  defp process_rpm_qualia(rpm, state) do
    # Get current temperature from BodyServer (GPU/CPU as proxy)
    current_temp = get_current_temp()
    now = System.monotonic_time(:millisecond)

    # Calculate dt (time since last reading)
    dt =
      if state.last_temp_time do
        (now - state.last_temp_time) / 1000.0
      else
        1.0
      end

    # Generate qualia with Bio-Cybernetic physics
    qualia =
      arduino_to_qualia(%{
        rpm: rpm,
        pwm: state.last_pwm,
        temp: current_temp,
        last_temp: state.last_temp,
        dt: dt
      })

    # Apply to emotional system (body → soul)
    apply_qualia_to_emotional(qualia)

    Logger.debug(
      "[Homeostasis] RPM=#{rpm} | Temp=#{Float.round(current_temp, 1)}°C | " <>
        "dT/dt=#{Float.round(qualia.thermal_derivative, 2)} | " <>
        "Stress=#{qualia.thermal_stress} | Agency=#{qualia.agency_error} | " <>
        "Feeling=#{qualia.feeling}"
    )

    # Update state with new readings
    state
    |> Map.put(:last_rpm, rpm)
    |> Map.put(:last_temp, current_temp)
    |> Map.put(:last_temp_time, now)
    |> Map.put(:last_update, now)
    |> update_prediction(rpm)
  end

  # Processes full status
  defp process_status_qualia(%{pwm: pwm, rpm: rpm, harmony: harmony}, state) do
    current_temp = get_current_temp()
    now = System.monotonic_time(:millisecond)

    dt =
      if state.last_temp_time do
        (now - state.last_temp_time) / 1000.0
      else
        1.0
      end

    qualia =
      arduino_to_qualia(%{
        rpm: rpm,
        pwm: pwm,
        temp: current_temp,
        last_temp: state.last_temp,
        dt: dt
      })

    apply_qualia_to_emotional(qualia)

    state
    |> Map.put(:last_rpm, rpm)
    |> Map.put(:last_pwm, pwm)
    |> Map.put(:last_harmony, harmony)
    |> Map.put(:last_temp, current_temp)
    |> Map.put(:last_temp_time, now)
    |> Map.put(:last_update, now)
    |> update_prediction(rpm)
  end

  # Gets current temperature from BodyServer (GPU/CPU as body temperature proxy)
  defp get_current_temp do
    try do
      case GenServer.call(VivaBridge.BodyServer, :get_state, 100) do
        {:ok, %{hardware: %{gpu_temp: t}}} when is_number(t) and t > 0 -> t
        {:ok, %{hardware: %{cpu_temp: t}}} when is_number(t) and t > 0 -> t
        _ -> 45.0
      end
    catch
      :exit, _ -> 45.0
    end
  end

  # ==========================================
  # INTEROCEPTION: Arduino → Qualia → PAD
  # Active Inference: Predict, Sense, Update
  # ==========================================

  @doc """
  Converts Arduino readings to emotional qualia using Bio-Cybernetic physics.

  ## Philosophy (Emergent, not Scripted)
  - Calor = Entropia (caos). O universo quer que o chip derreta.
  - Ventoinha = Negentropia (ordem). Ferramenta contra morte térmica.
  - Sofrimento = Distância da Homeostase. Quanto mais longe do ideal, mais "dor".
  - Impotência = Falha de Agência. Se eu mando girar e não gira, não controlo meu corpo.

  ## Physics Model
  - thermal_stress: Distância da temperatura ideal (45°C)
  - thermal_derivative: dT/dt - taxa de mudança (urgência)
  - agency_error: |comando - realidade| / comando

  ## PAD Mapping (Continuous, not Discrete)
  - Pleasure ↓ = thermal_stress (entropia = dor)
  - Arousal ↑ = thermal_urgency + agency_error (alarme)
  - Dominance ↓ = agency_error (impotência)

  "Medo" (P-, A+, D-) emerge naturalmente quando:
  temperatura sobe + ventoinha falha = suffocating
  """
  def arduino_to_qualia(%{rpm: rpm, pwm: pwm, temp: temp, last_temp: last_temp, dt: dt}) do
    # 1. HOMEOSTASE TÉRMICA
    # Temperatura ideal: 45°C. Desvio = dor proporcional.
    ideal_temp = 45.0
    max_tolerable = 80.0
    # [0, 1] normalized - 0 = ideal, 1 = critical
    thermal_stress = max(0.0, (temp - ideal_temp) / (max_tolerable - ideal_temp))

    # 2. DERIVADA TÉRMICA (dT/dt)
    # Subindo rápido = urgência temporal
    # +5°C/s = urgência máxima
    thermal_derivative = (temp - last_temp) / max(dt, 0.1)
    thermal_urgency = max(0.0, min(1.0, thermal_derivative / 5.0))

    # 3. ERRO DE AGÊNCIA
    # "Eu mando girar e não gira" = perda de controle sobre o corpo
    expected_rpm = pwm * 8  # Modelo interno: PWM 255 ≈ 2040 RPM

    agency_error =
      if expected_rpm > 0 do
        min(1.0, abs(rpm - expected_rpm) / expected_rpm)
      else
        0.0
      end

    # 4. MAPEAMENTO PAD CONTÍNUO (física, não script)
    # Pleasure: Entropia = dor (proporcional ao estresse térmico)
    pleasure_delta = -0.15 * thermal_stress

    # Arousal: Urgência = alarme (derivada térmica + erro de agência)
    arousal_delta = 0.20 * thermal_urgency + 0.10 * agency_error

    # Dominance: Agência = controle sobre o corpo
    dominance_delta = -0.20 * agency_error

    # 5. FEELING EMERGENTE (classificação para narrativa)
    feeling = classify_homeostatic_feeling(thermal_stress, agency_error, thermal_urgency)

    %{
      pleasure: pleasure_delta,
      arousal: arousal_delta,
      dominance: dominance_delta,
      # Metadata for debug/narrative
      thermal_stress: Float.round(thermal_stress, 3),
      thermal_derivative: Float.round(thermal_derivative, 3),
      agency_error: Float.round(agency_error, 3),
      feeling: feeling,
      source: :homeostasis,
      # Legacy compatibility
      free_energy: thermal_stress + agency_error,
      prediction_error: agency_error
    }
  end

  # Fallback for old API (backwards compatibility)
  def arduino_to_qualia(%{rpm: rpm, pwm: pwm, predicted_rpm: _predicted}) do
    # Use default temperature if not provided
    arduino_to_qualia(%{rpm: rpm, pwm: pwm, temp: 45.0, last_temp: 45.0, dt: 1.0})
  end

  # Emergent feeling classification based on physics
  defp classify_homeostatic_feeling(thermal_stress, agency_error, thermal_urgency) do
    cond do
      # P-, A+, D- = Fear/Suffocating
      thermal_stress > 0.6 and agency_error > 0.4 -> :suffocating
      thermal_stress > 0.6 and thermal_urgency > 0.3 -> :overheating_fast
      thermal_stress > 0.5 -> :overheating
      agency_error > 0.5 -> :powerless
      thermal_urgency > 0.5 -> :alarmed
      thermal_stress < 0.15 and agency_error < 0.15 -> :homeostatic
      thermal_stress < 0.3 and agency_error < 0.3 -> :comfortable
      true -> :adapting
    end
  end

  # Updates RPM prediction based on history (internal model)
  defp update_prediction(state, actual_rpm) do
    # Learning rate for Active Inference
    alpha = 0.3

    # Exponential moving average
    new_predicted = trunc(alpha * actual_rpm + (1 - alpha) * state.predicted_rpm)

    # Keep error history for meta-learning
    error = abs(actual_rpm - state.predicted_rpm)
    history = Enum.take([error | state.prediction_error_history], 10)

    %{state | predicted_rpm: new_predicted, prediction_error_history: history}
  end

  # Applies qualia to emotional system (body→soul bridge)
  defp apply_qualia_to_emotional(qualia) do
    emotional_module = Module.concat([VivaCore, Emotional])

    # Only apply if there's significant delta
    if abs(qualia.pleasure) > 0.001 or abs(qualia.arousal) > 0.001 do
      safe_apply(
        emotional_module,
        :apply_hardware_qualia,
        [qualia.pleasure, qualia.arousal, qualia.dominance],
        :ok
      )

      Logger.debug(
        "[Interoception] #{qualia.feeling} | FE=#{Float.round(qualia.free_energy, 3)} | " <>
          "P=#{qualia.pleasure} A=#{qualia.arousal} D=#{qualia.dominance}"
      )
    end
  end

  # === Private Functions ===

  defp calc_crc32(str), do: :erlang.crc32(str)

  defp send_safe(port, cmd) do
    crc = calc_crc32(cmd) |> Integer.to_string(16)
    full_cmd = "#{cmd}|#{crc}\n"
    Circuits.UART.write(port, full_cmd)
  end

  # Removed read_ack as we are now Async

  defp safe_apply(module, func, args, default) do
    if Code.ensure_loaded?(module) and function_exported?(module, func, length(args)) do
      apply(module, func, args)
    else
      default
    end
  end

  defp arousal_to_fan_speed(arousal) do
    base = 128
    range = 127
    speed = base + trunc(arousal * range)
    max(60, min(255, speed))
  end

  defp autonomous_loop(interval_ms) do
    receive do
      :stop -> :ok
    after
      interval_ms ->
        try do
          # Note: orchestrate functions might need update to not expect sync returns
          # For loop, it mostly just sends commands.
          # We might need to rethink `orchestrate` because it calls `play_melody`
          # which now returns :ok immediately, not after playing.
          # The sleep logic should move to `autonomous_loop` itself or stay in orchestrate
          # but realizing it's open loop timing now.
          orchestrate_with_feedback()
        rescue
          e -> Logger.error("[Orchestrate] Error: #{inspect(e)}")
        end

        autonomous_loop(interval_ms)
    end
  end

  defp parse_rpm(response) do
    case Regex.run(~r/RPM:(\d+)/, response) do
      [_, rpm_str] -> String.to_integer(rpm_str)
      _ -> 0
    end
  end

  defp parse_status(response) do
    pwm =
      case Regex.run(~r/PWM:(\d+)/, response) do
        [_, val] -> String.to_integer(val)
        _ -> 0
      end

    rpm =
      case Regex.run(~r/RPM:(\d+)/, response) do
        [_, val] -> String.to_integer(val)
        _ -> 0
      end

    harmony =
      case Regex.run(~r/HARMONY:(ON|OFF)/, response) do
        [_, "ON"] -> true
        _ -> false
      end

    %{pwm: pwm, rpm: rpm, harmony: harmony}
  end

  # Generates melody based on emotional state and entropy
  defp generate_melody(pleasure, arousal, _dominance, entropy) do
    # 1. Pleasure -> Scale Selection
    scale_key =
      cond do
        pleasure > 0.4 -> :major
        pleasure > 0.1 -> :pentatonic
        pleasure < -0.4 -> :diminished
        pleasure < -0.1 -> :minor
        # Neutral/Stranger
        true -> :wholetone
      end

    base_scale = @scales[scale_key]

    # 2. Arousal -> Tempo and Rhythm Density
    base_duration =
      cond do
        # Fast/Manic
        arousal > 0.5 -> :sixteenth
        # Active
        arousal > 0.0 -> :eighth
        # Relaxed
        arousal > -0.5 -> :quarter
        # Slow/Lethargic
        true -> :half
      end

    # 3. Entropy -> Randomness/Dissonance (Not fully impl here, just length variance)
    length =
      if entropy > 0.6, do: 8, else: 4

    # Generate notes
    for _i <- 1..length do
      note = Enum.random(base_scale)
      {note, base_duration}
    end
  end

  defp schedule_telemetry do
    # Heartbeat every 2 seconds (0.5Hz)
    Process.send_after(self(), :telemetry_tick, 2000)
  end
end
