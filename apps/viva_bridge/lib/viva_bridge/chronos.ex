defmodule VivaBridge.Chronos do
  @moduledoc """
  Hardware-accelerated Oracle.
  Manages the Python Port for the Amazon Chronos-T5 model.
  """

  use GenServer
  require Logger

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Predict the next value in a sequence.
  Returns {:ok, prediction, range} or {:error, reason}.
  """
  def predict(history, metric \\ "metric") do
    # Long timeout for inference
    GenServer.call(__MODULE__, {:predict, history, metric}, 10_000)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  defmodule State do
    defstruct [:port, :queue]
  end

  @impl true
  def init(_opts) do
    Logger.info("[Chronos] Summoning the Time Lord (Python)...")

    # Path to script
    script = Application.app_dir(:viva_bridge, "priv/python/chronos_service.py")

    # Check if python exists (assuming global path from verification)
    python_cmd = "python3"

    port = Port.open({:spawn, "#{python_cmd} -u #{script}"}, [:binary, :line, :exit_status])

    {:ok, %State{port: port, queue: :queue.new()}}
  end

  @impl true
  def handle_call({:predict, history, metric}, from, state) do
    payload = Jason.encode!(%{history: history, metric: metric})
    Port.command(state.port, "#{payload}\n")

    # Enqueue the caller to wait for the result
    new_queue = :queue.in(from, state.queue)

    {:noreply, %{state | queue: new_queue}}
  end

  @impl true
  def handle_info({_port, {:data, {:eol, line}}}, state) do
    # Decouple the response
    case Jason.decode(line) do
      {:ok, response} ->
        # Dequeue the waiting client
        {{:value, client}, new_queue} = :queue.out(state.queue)

        if Map.has_key?(response, "error") do
          GenServer.reply(client, {:error, response["error"]})
        else
          pred = response["prediction"]
          range = response["range"]
          GenServer.reply(client, {:ok, pred, range})
        end

        {:noreply, %{state | queue: new_queue}}

      {:error, _} ->
        # Log STDERR lines (which come as lines without valid JSON usually,
        # but Port line mode might mix them if not careful.
        # Actually standard python bridge logic sends stderr elsewhere usually,
        # but here we capturing stout. If it's just a log msg, ignore json error)
        # However, for simplicity, we assume robust JSON.
        Logger.debug("[Chronos][Py] #{line}")
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({_port, {:exit_status, status}}, state) do
    Logger.error("[Chronos] Python process died with status: #{status}")
    {:stop, :port_died, state}
  end

  @impl true
  def terminate(_reason, state) do
    if state.port, do: Port.close(state.port)
  end
end
