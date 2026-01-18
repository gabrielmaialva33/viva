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

  defstruct [:port, :connected, :learned_patterns]

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
  def init(_opts) do
    state = %__MODULE__{
      port: nil,
      connected: false,
      learned_patterns: @emotion_patterns
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:connect, port_path}, _from, state) do
    # On WSL, must use Windows COM port
    # Ex: /dev/ttyS5 for COM5
    case Circuits.UART.start_link() do
      {:ok, uart} ->
        # Arduino Serial usually 9600 or 115200. .ino code says 9600.
        case Circuits.UART.open(uart, port_path, speed: 9600, active: false) do
          :ok ->
            # Wait for Arduino reset
            Process.sleep(2000)

            # Read ready message
            case Circuits.UART.read(uart, 1000) do
              {:ok, data} ->
                Logger.info("Arduino connected: #{inspect(data)}")
                {:reply, {:ok, :connected}, %{state | port: uart, connected: true}}

              {:error, reason} ->
                Logger.warning("Arduino did not respond: #{inspect(reason)}")
                {:reply, {:ok, :connected_no_response}, %{state | port: uart, connected: true}}
            end

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
  def handle_call(:ping, _from, %{connected: false} = state) do
    {:reply, {:error, :not_connected}, state}
  end

  @impl true
  def handle_call(:ping, _from, %{port: port} = state) do
    Circuits.UART.write(port, "P\n")
    Process.sleep(100)

    case Circuits.UART.read(port, 500) do
      {:ok, "PONG\r\n"} -> {:reply, :pong, state}
      {:ok, data} -> {:reply, {:ok, data}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:play_note, freq, dur}, _from, %{connected: false} = state) do
    Logger.debug("Simulating note: #{freq}Hz for #{dur}ms")
    {:reply, {:simulated, freq, dur}, state}
  end

  @impl true
  def handle_call({:play_note, freq, dur}, _from, %{port: port} = state) do
    cmd = "N #{freq} #{dur}\n"
    Circuits.UART.write(port, cmd)
    Process.sleep(dur + 50)

    case Circuits.UART.read(port, 500) do
      {:ok, response} -> {:reply, {:ok, String.trim(response)}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
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
    Circuits.UART.write(port, cmd)

    # Calculate total time
    total_time =
      notes
      |> Enum.map(fn {_, dur} ->
        if is_atom(dur), do: Map.get(@durations, dur, 200), else: dur
      end)
      |> Enum.sum()

    # Wait for completion + margin
    Process.sleep(total_time + 500)

    case Circuits.UART.read(port, 1000) do
      {:ok, response} -> {:reply, {:ok, String.trim(response)}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:express_emotion, emotion}, _from, %{connected: false} = state) do
    Logger.debug("Simulating emotion: #{emotion}")
    {:reply, {:simulated, emotion}, state}
  end

  @impl true
  def handle_call({:express_emotion, emotion}, _from, %{port: port} = state) do
    cmd = "E #{emotion}\n"
    Circuits.UART.write(port, cmd)
    Process.sleep(2000)

    case Circuits.UART.read(port, 1000) do
      {:ok, response} -> {:reply, {:ok, String.trim(response)}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
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

  # === Hardware Control Handlers ===

  @impl true
  def handle_call({:set_fan, speed}, _from, %{connected: false} = state) do
    Logger.debug("Simulating fan speed: #{speed}")
    {:reply, {:simulated, speed}, state}
  end

  @impl true
  def handle_call({:set_fan, speed}, _from, %{port: port} = state) do
    send_safe(port, "F #{speed}")

    case read_ack(port) do
      {:ok, response} -> {:reply, {:ok, response}, state}
      error -> {:reply, error, state}
    end
  end

  @impl true
  def handle_call(:get_rpm, _from, %{connected: false} = state) do
    {:reply, {:simulated, 0}, state}
  end

  @impl true
  def handle_call(:get_rpm, _from, %{port: port} = state) do
    send_safe(port, "R")

    case read_ack(port) do
      {:ok, response} ->
        rpm = parse_rpm(response)
        {:reply, {:ok, rpm}, state}

      error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call(:get_status, _from, %{connected: false} = state) do
    {:reply, {:simulated, %{pwm: 0, rpm: 0, harmony: true}}, state}
  end

  @impl true
  def handle_call(:get_status, _from, %{port: port} = state) do
    send_safe(port, "S")

    case read_ack(port) do
      {:ok, response} ->
        status = parse_status(response)
        {:reply, {:ok, status}, state}

      error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:set_harmony, enabled}, _from, %{connected: false} = state) do
    Logger.debug("Simulating harmony: #{enabled}")
    {:reply, {:simulated, enabled}, state}
  end

  @impl true
  def handle_call({:set_harmony, enabled}, _from, %{port: port} = state) do
    val = if enabled, do: "1", else: "0"
    send_safe(port, "H #{val}")

    case read_ack(port) do
      {:ok, response} -> {:reply, {:ok, response}, state}
      error -> {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:raw_cmd, cmd}, _from, %{connected: false} = state) do
    Logger.debug("Simulating raw cmd: #{cmd}")
    {:reply, {:simulated, cmd}, state}
  end

  @impl true
  def handle_call({:raw_cmd, cmd}, _from, %{port: port} = state) do
    send_safe(port, cmd)

    case read_ack(port, 1000) do
      {:ok, response} -> {:reply, {:ok, response}, state}
      error -> {:reply, error, state}
    end
  end

  # === Private Functions ===

  # Calculates CRC32 of a string using Erlang's zlib
  defp calc_crc32(str) do
    :erlang.crc32(str)
  end

  # Sends command wrapped with CRC32
  # Ex: "F 255" -> "F 255|A1B2C3D4\n"
  defp send_safe(port, cmd) do
    crc = calc_crc32(cmd) |> Integer.to_string(16)
    full_cmd = "#{cmd}|#{crc}\n"
    Circuits.UART.write(port, full_cmd)
  end

  # Reads response and checks for ACK/NAK
  defp read_ack(port, timeout \\ 500) do
    # Give Arduino processing time
    Process.sleep(50)

    case Circuits.UART.read(port, timeout) do
      {:ok, raw_response} ->
        response = String.trim(raw_response)

        cond do
          String.starts_with?(response, "ACK:") ->
            {:ok, String.replace_prefix(response, "ACK:", "")}

          String.starts_with?(response, "NAK:") ->
            Logger.error("[Music] NAK received: #{response}")
            {:error, :nak_received}

          true ->
            # Fallback for mixed output
            {:ok, response}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Helper for safe dynamic dispatch
  defp safe_apply(module, func, args, default) do
    if Code.ensure_loaded?(module) and function_exported?(module, func, length(args)) do
      apply(module, func, args)
    else
      # Logger.warning("Module #{module} or function #{func} not available - using default")
      default
    end
  end

  # Converts arousal [-1, 1] to fan speed [60, 255]
  defp arousal_to_fan_speed(arousal) do
    # Map arousal to fan speed
    # -1.0 (calm) -> 60 (gentle breeze)
    # 0.0 (neutral) -> 128
    # 1.0 (excited) -> 255 (full blast)
    base = 128
    range = 127
    speed = base + trunc(arousal * range)
    max(60, min(255, speed))
  end

  # Autonomous orchestration loop with feedback
  defp autonomous_loop(interval_ms) do
    receive do
      :stop -> :ok
    after
      interval_ms ->
        try do
          orchestrate_with_feedback()
        rescue
          e -> Logger.error("[Orchestrate] Error: #{inspect(e)}")
        end

        autonomous_loop(interval_ms)
    end
  end

  # Parses "RPM:1234" response
  defp parse_rpm(response) do
    case Regex.run(~r/RPM:(\d+)/, response) do
      [_, rpm_str] -> String.to_integer(rpm_str)
      _ -> 0
    end
  end

  # Parses "PWM:128,RPM:1200,HARMONY:ON" response
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
        # Frenetic
        arousal > 0.5 -> :sixteenth
        # Animated
        arousal > 0.0 -> :eighth
        # Drone/Ambient
        arousal < -0.5 -> :whole
        # Moderate
        true -> :quarter
      end

    # 3. Dominance -> (Future: Volume/Octave)

    # 4. Entropy (Jazz Factor) -> Length and Variation
    note_count = max(4, trunc(8 + arousal * 4))
    # 0.0 to 1.0+
    jazz_factor = entropy

    # Generator
    1..note_count
    |> Enum.map(fn i ->
      # Choose note from scale
      # If high entropy, chance to pick random chromatic note
      note =
        if :rand.uniform() < jazz_factor * 0.3 do
          Enum.random(@scales[:chromatic])
        else
          idx = rem(i, length(base_scale))
          # Simple octave shift logic if needed
          Enum.at(base_scale, idx)
        end

      # Duration variation based on entropy
      duration =
        if :rand.uniform() < jazz_factor * 0.5 do
          Enum.random([:eighth, :quarter, :sixteenth])
        else
          if rem(i, 4) == 0, do: :half, else: base_duration
        end

      {note, duration}
    end)
  end
end
