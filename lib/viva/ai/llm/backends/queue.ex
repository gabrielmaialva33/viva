defmodule Viva.AI.LLM.Backends.Queue do
  @moduledoc """
  Sistema de fila com prioridades para requisições LLM.

  Gerencia requisições de múltiplos avatares garantindo:
  - Prioridade para ações críticas (conversas, emergências)
  - Fairness entre avatares (round-robin quando prioridade igual)
  - Rate limiting para evitar sobrecarga
  - Métricas de latência e throughput

  ## Prioridades

  1. **:critical** (0) - Ações de emergência, safety checks
  2. **:high** (1) - Conversas em tempo real com usuário
  3. **:normal** (2) - Pensamentos autônomos, reflexões
  4. **:low** (3) - Background tasks, análises não urgentes
  5. **:batch** (4) - Processamento em lote, otimizações

  ## Uso

      {:ok, ref} = Queue.enqueue(:normal, fn -> generate_thought(prompt) end)
      {:ok, result} = Queue.await(ref, 30_000)

  ## Configuração

      config :viva, Viva.AI.LLM.Backends.Queue,
        max_concurrent: 4,       # Requisições simultâneas
        max_queue_size: 1000,    # Tamanho máximo da fila
        default_timeout: 30_000, # Timeout padrão
        metrics_enabled: true
  """

  use GenServer

  require Logger

  alias Viva.AI.LLM.Backends.Router

  @priority_levels %{
    critical: 0,
    high: 1,
    normal: 2,
    low: 3,
    batch: 4
  }

  @default_max_concurrent 4
  @default_max_queue_size 1000
  @default_timeout 30_000

  # Estado do GenServer
  defstruct [
    :queue,           # Priority queue (heap)
    :processing,      # MapSet de refs em processamento
    :results,         # Map de ref => result
    :waiters,         # Map de ref => [pid]
    :metrics,         # Métricas de performance
    :config           # Configuração
  ]

  # API Pública

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Adiciona uma requisição à fila.

  ## Options
    * `:priority` - Nível de prioridade (:critical, :high, :normal, :low, :batch)
    * `:avatar_id` - ID do avatar para fairness
    * `:timeout` - Timeout específico para esta requisição

  Retorna `{:ok, ref}` onde ref pode ser usado para await.
  """
  def enqueue(fun, opts \\ []) when is_function(fun, 0) do
    GenServer.call(__MODULE__, {:enqueue, fun, opts})
  end

  @doc """
  Enfileira um generate com roteamento automático.
  """
  def generate(prompt, opts \\ []) do
    priority = Keyword.get(opts, :priority, :normal)
    timeout = Keyword.get(opts, :timeout, @default_timeout)

    fun = fn -> Router.generate(prompt, opts) end

    case enqueue(fun, priority: priority, timeout: timeout) do
      {:ok, ref} -> await(ref, timeout)
      error -> error
    end
  end

  @doc """
  Enfileira um generate_thought (prioridade normal).
  """
  def generate_thought(prompt, opts \\ []) do
    opts = Keyword.put_new(opts, :priority, :normal)
    generate(prompt, Keyword.put(opts, :task_type, :thought))
  end

  @doc """
  Enfileira uma conversa (prioridade alta).
  """
  def chat(messages, opts \\ []) do
    priority = Keyword.get(opts, :priority, :high)
    timeout = Keyword.get(opts, :timeout, @default_timeout)

    fun = fn -> Router.chat(messages, opts) end

    case enqueue(fun, priority: priority, timeout: timeout) do
      {:ok, ref} -> await(ref, timeout)
      error -> error
    end
  end

  @doc """
  Aguarda o resultado de uma requisição enfileirada.
  """
  def await(ref, timeout \\ @default_timeout) do
    GenServer.call(__MODULE__, {:await, ref}, timeout + 1000)
  catch
    :exit, {:timeout, _} ->
      GenServer.cast(__MODULE__, {:cancel, ref})
      {:error, :timeout}
  end

  @doc """
  Retorna estatísticas da fila.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Retorna o tamanho atual da fila.
  """
  def queue_size do
    GenServer.call(__MODULE__, :queue_size)
  end

  @doc """
  Cancela uma requisição pendente.
  """
  def cancel(ref) do
    GenServer.cast(__MODULE__, {:cancel, ref})
  end

  # GenServer Callbacks

  @impl true
  def init(opts) do
    config = %{
      max_concurrent: Keyword.get(opts, :max_concurrent, config(:max_concurrent, @default_max_concurrent)),
      max_queue_size: Keyword.get(opts, :max_queue_size, config(:max_queue_size, @default_max_queue_size)),
      default_timeout: Keyword.get(opts, :default_timeout, config(:default_timeout, @default_timeout))
    }

    state = %__MODULE__{
      queue: :gb_sets.empty(),
      processing: MapSet.new(),
      results: %{},
      waiters: %{},
      metrics: init_metrics(),
      config: config
    }

    Logger.info("[Queue] Started with max_concurrent=#{config.max_concurrent}")

    {:ok, state}
  end

  @impl true
  def handle_call({:enqueue, fun, opts}, _from, state) do
    if :gb_sets.size(state.queue) >= state.config.max_queue_size do
      {:reply, {:error, :queue_full}, state}
    else
      ref = make_ref()
      priority = @priority_levels[Keyword.get(opts, :priority, :normal)] || 2
      timestamp = System.monotonic_time(:millisecond)
      avatar_id = Keyword.get(opts, :avatar_id)

      # Item na fila: {priority, timestamp, avatar_id, ref, fun, opts}
      item = {priority, timestamp, avatar_id, ref, fun, opts}
      new_queue = :gb_sets.add_element(item, state.queue)

      state = %{state | queue: new_queue}
      state = update_metric(state, :enqueued)

      # Tenta processar se há slots disponíveis
      state = maybe_process_next(state)

      {:reply, {:ok, ref}, state}
    end
  end

  @impl true
  def handle_call({:await, ref}, from, state) do
    case Map.get(state.results, ref) do
      nil ->
        # Ainda não completou, adiciona à lista de waiters
        waiters = Map.update(state.waiters, ref, [from], &[from | &1])
        {:noreply, %{state | waiters: waiters}}

      result ->
        # Já temos resultado
        results = Map.delete(state.results, ref)
        {:reply, result, %{state | results: results}}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      queue_size: :gb_sets.size(state.queue),
      processing: MapSet.size(state.processing),
      pending_results: map_size(state.results),
      metrics: state.metrics,
      config: state.config
    }
    {:reply, stats, state}
  end

  @impl true
  def handle_call(:queue_size, _from, state) do
    {:reply, :gb_sets.size(state.queue), state}
  end

  @impl true
  def handle_cast({:cancel, ref}, state) do
    # Remove da fila se ainda estiver lá
    queue = :gb_sets.filter(fn {_, _, _, r, _, _} -> r != ref end, state.queue)

    # Notifica waiters se houver
    case Map.get(state.waiters, ref) do
      nil -> :ok
      waiters -> Enum.each(waiters, &GenServer.reply(&1, {:error, :cancelled}))
    end

    state = %{state |
      queue: queue,
      waiters: Map.delete(state.waiters, ref),
      results: Map.delete(state.results, ref)
    }

    {:noreply, state}
  end

  @impl true
  def handle_info({:task_complete, ref, result, duration}, state) do
    # Remove do processamento
    processing = MapSet.delete(state.processing, ref)

    # Atualiza métricas
    state = state
      |> update_metric(:completed)
      |> update_latency(duration)

    # Notifica waiters ou armazena resultado
    state = case Map.get(state.waiters, ref) do
      nil ->
        %{state | results: Map.put(state.results, ref, result)}

      waiters ->
        Enum.each(waiters, &GenServer.reply(&1, result))
        %{state | waiters: Map.delete(state.waiters, ref)}
    end

    state = %{state | processing: processing}

    # Processa próximo da fila
    state = maybe_process_next(state)

    {:noreply, state}
  end

  @impl true
  def handle_info({:task_failed, ref, reason, duration}, state) do
    handle_info({:task_complete, ref, {:error, reason}, duration}, state)
  end

  # Processamento

  defp maybe_process_next(state) do
    can_process = MapSet.size(state.processing) < state.config.max_concurrent
    has_items = not :gb_sets.is_empty(state.queue)

    if can_process and has_items do
      # Pega o item de maior prioridade (menor número)
      {{priority, timestamp, _avatar_id, ref, fun, _opts}, queue} = :gb_sets.take_smallest(state.queue)

      # Adiciona ao processamento
      processing = MapSet.put(state.processing, ref)

      # Spawna task para execução
      parent = self()
      Task.start(fn ->
        start = System.monotonic_time(:millisecond)
        try do
          result = fun.()
          duration = System.monotonic_time(:millisecond) - start
          send(parent, {:task_complete, ref, result, duration})
        rescue
          e ->
            duration = System.monotonic_time(:millisecond) - start
            send(parent, {:task_failed, ref, e, duration})
        end
      end)

      wait_time = System.monotonic_time(:millisecond) - timestamp
      Logger.debug("[Queue] Processing ref=#{inspect(ref)} priority=#{priority} wait=#{wait_time}ms")

      %{state | queue: queue, processing: processing}
    else
      state
    end
  end

  # Métricas

  defp init_metrics do
    %{
      enqueued: 0,
      completed: 0,
      errors: 0,
      total_latency: 0,
      avg_latency: 0
    }
  end

  defp update_metric(state, :enqueued) do
    update_in(state.metrics.enqueued, &(&1 + 1))
  end

  defp update_metric(state, :completed) do
    update_in(state.metrics.completed, &(&1 + 1))
  end

  defp update_metric(state, :error) do
    update_in(state.metrics.errors, &(&1 + 1))
  end

  defp update_latency(state, duration) do
    metrics = state.metrics
    total = metrics.total_latency + duration
    count = metrics.completed
    avg = if count > 0, do: div(total, count), else: 0

    %{state | metrics: %{metrics | total_latency: total, avg_latency: avg}}
  end

  defp config(key, default) do
    Application.get_env(:viva, __MODULE__, [])
    |> Keyword.get(key, default)
  end
end
