defmodule Viva.System.Proprioception do
  use GenServer
  require Logger

  @moduledoc """
  The Cerebellum of VIVA.
  Monitors hardware "vitality" (RAM/VRAM) to decide if the organism is capable of high-level thought (loading LLMs).
  """

  # Wake up every 5 seconds to check vitals
  @heartbeat 5_000

  defstruct [:total_ram, :free_ram, :vram_free, :state]

  # API
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, nil, name: __MODULE__)
  end

  def vitality do
    GenServer.call(__MODULE__, :get_vitality)
  end

  # Callbacks
  @impl true
  def init(_) do
    Logger.info("🧠 Proprioception System Online.")
    Process.send_after(self(), :tick, 1000)
    {:ok, %__MODULE__{state: :initializing}}
  end

  @impl true
  def handle_info(:tick, state) do
    # 1. Sense RAM (Native NIF)
    {total, free} = Viva.Llm.get_memory_status()

    # 2. Sense VRAM (Nvidia-SMI Port - simplified for now via System.cmd)
    # In production we'd keep a Port open, but cmd is fine for 5s ticks.
    vram_free = check_nvram()

    # 3. Decision Matrix (Homeostasis)
    new_state = assess_load(free, vram_free)

    if new_state != state.state do
      Logger.info("State changed due to proprioception: #{state.state} -> #{new_state}")

      # Autonomic Reflexes
      case new_state do
        :lucid_dreaming ->
          if state.state != :lucid_dreaming do
            Logger.info("🚀 VRAM detected. Triggering Autonomic Model Loading.")
            Viva.Llm.Server.load_model()
          end

        :exhausted ->
          Logger.warning("⚠️ System Exhausted. Recommendation: Unload models.")

        _ ->
          :ok
      end
    end

    Process.send_after(self(), :tick, @heartbeat)

    {:noreply,
     %{state | total_ram: total, free_ram: free, vram_free: vram_free, state: new_state}}
  end

  @impl true
  def handle_call(:get_vitality, _from, state) do
    {:reply, state, state}
  end

  # Helpers

  defp check_nvram do
    case System.cmd("nvidia-smi", ["--query-gpu=memory.free", "--format=csv,noheader,nounits"]) do
      {output, 0} ->
        # Returns MB
        output |> String.trim() |> String.to_integer()

      _ ->
        # Blind to GPU
        0
    end
  rescue
    _ -> 0
  end

  defp assess_load(ram_bytes, vram_mb) do
    ram_gb = ram_bytes / 1024 / 1024 / 1024

    cond do
      # Plenty of room for Llama-3-8B
      vram_mb > 12_000 -> :lucid_dreaming
      # Danger zone
      ram_gb < 4 -> :exhausted
      # Normal operation
      true -> :awake
    end
  end
end
