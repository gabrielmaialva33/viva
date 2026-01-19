defmodule VivaCore.BodySchema do
  @moduledoc """
  The "Self-Image" of the VIVA cell.

  This module defines the boundaries and capabilities of the current instance (Cell).
  It implements "Motor Babbling" to discover available hardware organs (Fan, GPU, Network)
  and "Network Sense" to perceive the environment where other cells might exist.
  """

  use GenServer
  require Logger

  # ============================================================================
  # Struct Definition
  # ============================================================================
  defstruct [
    # Hardware Discovery (The "Organelles")
    local_hardware: %{
      # %{cores: 8, model: "..."}
      cpu: nil,
      # %{name: "RTX 4090", vram_gb: 24}
      gpu: nil,
      # Total RAM
      memory_gb: nil,
      # Is an Arduino connected?
      fan_control: false,
      # :working | :broken | :absent
      fan_status: :unknown,
      # nil = desktop, %{...} = laptop
      battery: nil
    },

    # Network Environment (The "Tissue")
    network_env: %{
      gateway: nil,
      local_ip: nil,
      public_ip: nil,
      # Other VIVA cells or devices
      peers: [],
      # :isolated | :local | :internet
      isolation_level: :unknown
    },

    # Meta-Cognition
    # 0-1 (Certainty about self)
    confidence: 0.0,
    # 0-1 (Desire to divide/spread)
    replication_drive: 0.0
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def get_schema do
    GenServer.call(__MODULE__, :get_schema)
  end

  def probe do
    GenServer.cast(__MODULE__, :probe)
  end

  def probe_network do
    GenServer.cast(__MODULE__, :probe_network)
  end

  def has_capability?(cap) do
    GenServer.call(__MODULE__, {:has_capability, cap})
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("[BodySchema] Cell membrane forming. Identity unknown.")

    # Start self-discovery immediately
    {:ok, %__MODULE__{}, {:continue, :motor_babbling}}
  end

  @impl true
  def handle_continue(:motor_babbling, state) do
    Logger.info("[BodySchema] Motor Babbling: Testing limbs and senses...")

    # 1. Probe CPU/Memory (Introspection)
    sys_info = probe_system_info()

    # 2. Probe GPU (via BodyServer)
    gpu_info = probe_gpu()

    # 3. Probe Fan (Motor Action)
    # This is async, will trigger a cast later or block shortly
    {fan_control, fan_status} = probe_fan_action()

    new_hardware = %{
      state.local_hardware
      | cpu: sys_info.cpu,
        memory_gb: sys_info.memory,
        gpu: gpu_info,
        fan_control: fan_control,
        fan_status: fan_status
    }

    # 4. Schedule Network Probe (The Environment)
    Process.send_after(self(), :probe_network_tick, 2000)

    # Calculate initial confidence based on what we found
    confidence = calculate_confidence(new_hardware)

    Logger.info("[BodySchema] Self-Discovery complete. Confidence: #{confidence}")
    if fan_status == :working, do: Logger.info("[BodySchema] Active cooling (Lungs) DETECTED.")

    if fan_status == :absent,
      do: Logger.info("[BodySchema] Active cooling ABSENT. Adapting metabolism.")

    new_state = %{state | local_hardware: new_hardware, confidence: confidence}

    # Notify Emotional module about body capabilities
    # This allows it to disable distress for absent organs
    notify_emotional(new_state)

    {:noreply, new_state}
  end

  @impl true
  def handle_call(:get_schema, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call({:has_capability, :fan}, _from, state) do
    {:reply, state.local_hardware.fan_status == :working, state}
  end

  @impl true
  def handle_call({:has_capability, :gpu}, _from, state) do
    {:reply, state.local_hardware.gpu != nil, state}
  end

  @impl true
  def handle_cast(:probe, state) do
    {:noreply, state, {:continue, :motor_babbling}}
  end

  @impl true
  def handle_info(:probe_network_tick, state) do
    # Async network scan
    probe_network_async()
    {:noreply, state}
  end

  @impl true
  def handle_info({:network_result, env}, state) do
    Logger.info(
      "[BodySchema] Environment Scanned. Gateway: #{env.gateway || "None"}. Peers: #{length(env.peers)}"
    )

    {:noreply, %{state | network_env: env}}
  end

  # ============================================================================
  # Internal Probes
  # ============================================================================

  defp probe_system_info do
    # Basic Erlang system info
    schedulers = System.schedulers_online()
    mem_bytes = :erlang.memory(:total)

    %{
      cpu: %{cores: schedulers},
      memory: Float.round(mem_bytes / 1024 / 1024 / 1024, 2)
    }
  end

  defp probe_gpu do
    # Try to ask BodyServer for GPU name
    try do
      case VivaBridge.BodyServer.get_state() do
        %{hardware: %{gpu_name: name}} when is_binary(name) ->
          %{name: name, type: :dedicated}

        _ ->
          nil
      end
    catch
      :exit, _ -> nil
    end
  end

  defp probe_fan_action do
    # The Motor Babbling: Try to move the limb
    # Check if Music module is connected
    if Code.ensure_loaded?(VivaBridge.Music) and VivaBridge.Music.connected?() do
      # Connected, now test response
      # Send pulse
      VivaBridge.Music.set_fan_speed(150)
      # Wait for spin up
      Process.sleep(1000)

      # Check sensation
      rpm =
        case VivaBridge.Music.get_rpm() do
          {:ok, val} -> val
          _ -> 0
        end

      # Reset
      # Maintain idle
      VivaBridge.Music.set_fan_speed(50)

      if rpm > 0 do
        {true, :working}
      else
        # Connected but no RPM? Maybe fan is broken or just 2-pin
        # or :absent if we assume 0 means no fan
        {true, :broken}
      end
    else
      {false, :absent}
    end
  end

  defp probe_network_async do
    pid = self()

    Task.start(fn ->
      env = %{
        gateway: get_gateway(),
        local_ip: get_local_ip(),
        public_ip: nil,
        peers: scan_arp(),
        isolation_level: :local
      }

      send(pid, {:network_result, env})
    end)
  end

  defp get_gateway do
    # Simple heuristic for Linux
    case System.cmd("ip", ["route"]) do
      {output, 0} ->
        output
        |> String.split("\n")
        |> Enum.find(&String.starts_with?(&1, "default"))
        |> case do
          nil -> nil
          line -> line |> String.split() |> Enum.at(2)
        end

      _ ->
        nil
    end
  end

  defp get_local_ip do
    case :inet.getif() do
      {:ok, list} ->
        case Enum.find(list, fn {ip, _, _} -> ip != {127, 0, 0, 1} end) do
          {ip, _, _} -> ip |> Tuple.to_list() |> Enum.join(".")
          nil -> nil
        end

      _ ->
        nil
    end
  end

  defp scan_arp do
    # Try different methods for ARP scanning
    # WSL2 doesn't have arp, so fallback to ip neigh
    cond do
      System.find_executable("arp") ->
        case System.cmd("arp", ["-a"]) do
          {output, 0} ->
            parse_arp_output(output)

          _ ->
            []
        end

      System.find_executable("ip") ->
        # Fallback for WSL2/minimal systems
        case System.cmd("ip", ["neigh"]) do
          {output, 0} ->
            parse_ip_neigh_output(output)

          _ ->
            []
        end

      true ->
        []
    end
  end

  defp parse_arp_output(output) do
    output
    |> String.split("\n")
    |> Enum.map(fn line ->
      parts = String.split(line)
      if length(parts) > 1, do: Enum.at(parts, 1) |> String.trim("()"), else: nil
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp parse_ip_neigh_output(output) do
    # ip neigh format: "192.168.1.1 dev eth0 lladdr aa:bb:cc:dd:ee:ff REACHABLE"
    output
    |> String.split("\n")
    |> Enum.map(fn line ->
      case String.split(line) do
        [ip | _rest] when ip != "" -> ip
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)
  end

  defp calculate_confidence(hw) do
    # How much do I know about myself?
    score = 0.0
    score = score + if hw.cpu, do: 0.2, else: 0.0
    score = score + if hw.fan_status != :unknown, do: 0.3, else: 0.0
    score = score + if hw.gpu, do: 0.2, else: 0.0
    score
  end

  defp notify_emotional(state) do
    # Notify Emotional about body capabilities so it can adjust weights
    # e.g., disable fan-related distress if no fan exists
    try do
      VivaCore.Emotional.configure_body_schema(state)
    catch
      :exit, _ ->
        # Emotional not started yet, that's fine
        :ok
    end
  end
end
