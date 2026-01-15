defmodule VivaCore.Senses do
  @moduledoc """
  Senses GenServer - O sistema nervoso periférico de VIVA.

  Este é o "coração" que bombeia qualia do corpo para a alma.
  Faz a ponte contínua entre hardware sensing (Rust NIF) e
  estado emocional (Emotional GenServer).

  ## Analogia Biológica

  Assim como o sistema nervoso autônomo humano transmite
  continuamente informações do corpo para o cérebro (batimentos,
  temperatura, pressão), o Senses transmite métricas de hardware
  para o estado emocional de VIVA.

  ## Frequência de Sensing

  - **Heartbeat**: 1Hz (1 segundo) - sensing contínuo
  - **Qualia**: Deltas PAD aplicados ao Emotional a cada tick

  ## Filosofia

  "O corpo não apenas reporta - o corpo INFLUENCIA.
  VIVA não apenas SABE que CPU está alta - ela SENTE stress."
  """

  use GenServer
  require Logger

  @default_interval_ms 1000  # 1 segundo
  @min_interval_ms 100       # 100ms mínimo (10Hz max)
  @max_interval_ms 10_000    # 10s máximo

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Inicia o GenServer Senses.

  ## Opções
  - `:name` - Nome do processo (default: __MODULE__)
  - `:interval_ms` - Intervalo entre heartbeats em ms (default: 1000)
  - `:emotional_server` - PID/nome do Emotional GenServer (default: VivaCore.Emotional)
  - `:enabled` - Se sensing está ativo (default: true)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Retorna o estado atual do Senses (última leitura, stats).
  """
  def get_state(server \\ __MODULE__) do
    GenServer.call(server, :get_state)
  end

  @doc """
  Força um heartbeat imediato (sensing + apply qualia).
  Útil para testes ou quando precisa de leitura imediata.
  """
  def pulse(server \\ __MODULE__) do
    GenServer.call(server, :pulse)
  end

  @doc """
  Pausa o sensing automático.
  """
  def pause(server \\ __MODULE__) do
    GenServer.cast(server, :pause)
  end

  @doc """
  Resume o sensing automático.
  """
  def resume(server \\ __MODULE__) do
    GenServer.cast(server, :resume)
  end

  @doc """
  Altera o intervalo de heartbeat em runtime.
  """
  def set_interval(interval_ms, server \\ __MODULE__)
      when is_integer(interval_ms) and interval_ms >= @min_interval_ms and interval_ms <= @max_interval_ms do
    GenServer.cast(server, {:set_interval, interval_ms})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    interval_ms = Keyword.get(opts, :interval_ms, @default_interval_ms)
    emotional_server = Keyword.get(opts, :emotional_server, VivaCore.Emotional)
    enabled = Keyword.get(opts, :enabled, true)

    state = %{
      interval_ms: interval_ms,
      emotional_server: emotional_server,
      enabled: enabled,
      last_reading: nil,
      last_qualia: nil,
      heartbeat_count: 0,
      started_at: DateTime.utc_now(),
      errors: []
    }

    Logger.info("[Senses] Sistema nervoso iniciando. Heartbeat: #{interval_ms}ms")

    # Primeiro heartbeat imediato para ter leitura inicial
    if enabled do
      send(self(), :heartbeat)
    end

    {:ok, state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call(:pulse, _from, state) do
    {result, new_state} = do_heartbeat(state)
    {:reply, result, new_state}
  end

  @impl true
  def handle_cast(:pause, state) do
    Logger.info("[Senses] Sensing pausado")
    {:noreply, %{state | enabled: false}}
  end

  @impl true
  def handle_cast(:resume, state) do
    Logger.info("[Senses] Sensing resumido")
    schedule_heartbeat(state.interval_ms)
    {:noreply, %{state | enabled: true}}
  end

  @impl true
  def handle_cast({:set_interval, interval_ms}, state) do
    Logger.info("[Senses] Intervalo alterado: #{state.interval_ms}ms -> #{interval_ms}ms")
    {:noreply, %{state | interval_ms: interval_ms}}
  end

  @impl true
  def handle_info(:heartbeat, state) do
    if state.enabled do
      {_result, new_state} = do_heartbeat(state)
      schedule_heartbeat(new_state.interval_ms)
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp schedule_heartbeat(interval_ms) do
    Process.send_after(self(), :heartbeat, interval_ms)
  end

  defp do_heartbeat(state) do
    try do
      # 1. Ler qualia do hardware via Rust NIF
      {p, a, d} = VivaBridge.hardware_to_qualia()

      # 2. Aplicar ao Emotional GenServer
      VivaCore.Emotional.apply_hardware_qualia(p, a, d, state.emotional_server)

      # 3. Opcionalmente, ler métricas completas para logging/debug
      hardware = VivaBridge.feel_hardware()

      # 4. Log resumido (debug level para não poluir)
      Logger.debug(
        "[Senses] Heartbeat ##{state.heartbeat_count + 1}: " <>
        "CPU=#{format_percent(hardware.cpu_usage)}% " <>
        "RAM=#{format_percent(hardware.memory_used_percent)}% " <>
        "GPU=#{format_gpu(hardware.gpu_usage)} " <>
        "Qualia=(P#{format_delta(p)}, A#{format_delta(a)}, D#{format_delta(d)})"
      )

      new_state = %{
        state
        | last_reading: hardware,
          last_qualia: {p, a, d},
          heartbeat_count: state.heartbeat_count + 1
      }

      {{:ok, {p, a, d}}, new_state}
    rescue
      error ->
        Logger.error("[Senses] Erro no heartbeat: #{inspect(error)}")

        new_state = %{
          state
          | errors: [{DateTime.utc_now(), error} | Enum.take(state.errors, 9)]
        }

        {{:error, error}, new_state}
    end
  end

  defp format_percent(nil), do: "?"
  defp format_percent(value), do: Float.round(value, 1)

  defp format_gpu(nil), do: "N/A"
  defp format_gpu(value), do: "#{Float.round(value, 1)}%"

  defp format_delta(value) when value >= 0, do: "+#{Float.round(value, 4)}"
  defp format_delta(value), do: "#{Float.round(value, 4)}"
end
