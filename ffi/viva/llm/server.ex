defmodule Viva.Llm.Server do
  use GenServer
  require Logger

  @moduledoc """
  The "Stomach" of VIVA.
  Holds the reference to the digested LLM Model (Native Resource).
  Executes inference requests.
  """

  @default_model "models/Llama-3-8B-Instruct.Q4_K_M.gguf"

  defstruct [:resource, :model_path, :status]
  # status: :unloaded | :loading | :ready | :error

  # API
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, nil, name: __MODULE__)
  end

  def load_model(path \\ @default_model) do
    GenServer.cast(__MODULE__, {:load, path})
  end

  def predict(prompt) do
    # Increased timeout for inference
    GenServer.call(__MODULE__, {:predict, prompt}, 60_000)
  end

  def status do
    GenServer.call(__MODULE__, :status)
  end

  # Callbacks
  @impl true
  def init(_) do
    {:ok, %__MODULE__{status: :unloaded}}
  end

  @impl true
  def handle_cast({:load, path}, state) do
    if state.status == :loading or state.status == :ready do
      Logger.info("Model already loaded or loading.")
      {:noreply, state}
    else
      Logger.info("🧂 Stomach: Ingesting model #{path}...")
      {:noreply, %{state | status: :loading, model_path: path}, {:continue, :do_load}}
    end
  end

  @impl true
  def handle_continue(:do_load, state) do
    # 99 layers on GPU if available (we know it is)
    case Viva.Llm.load_model(state.model_path, 99) do
      {:ok, resource} ->
        Logger.info("✨ Stomach: Model digested successfully. Ready for synthesis.")
        {:noreply, %{state | resource: resource, status: :ready}}

      {:error, reason} ->
        Logger.error("🤢 Stomach: Failed to digest model: #{inspect(reason)}")
        {:noreply, %{state | status: :error}}
    end
  end

  @impl true
  def handle_call({:predict, prompt}, _from, %{status: :ready, resource: resource} = state) do
    Logger.debug("Stomach process: #{prompt}")

    case Viva.Llm.predict(resource, prompt) do
      {:ok, result} -> {:reply, {:ok, result}, state}
      error -> {:reply, error, state}
    end
  end

  def handle_call({:predict, _}, _from, state) do
    {:reply, {:error, :model_not_ready}, state}
  end

  def handle_call(:status, _from, state) do
    {:reply, state.status, state}
  end
end
