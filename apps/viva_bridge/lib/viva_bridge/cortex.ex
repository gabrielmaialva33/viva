defmodule VivaBridge.Cortex do
  @moduledoc """
  Elixir Bridge for VIVA Cortex (Liquid Neural Networks).

  Manages the "Physics of Consciousness".
  Uses Liquid Time-Constant (LTC) networks to simulate continuous emotional flow.
  """

  use GenServer
  require VivaLog

  # ============================================================================
  # Public API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Process a tick through the Liquid Brain.

  ## Parameters
  - `pad`: Current emotional state [P, A, D]
  - `energy`: Current metabolic energy (0.0 - 1.0)
  - `context`: Context vector (from Ultra/Mamba)

  ## Returns
  - `{:ok, new_pad}`
  """
  def tick(pad, energy, context \\ []) do
    # Increase timeout for slow python inference
    GenServer.call(__MODULE__, {:tick, pad, energy, context}, 5000)
  end

  @doc """
  Process an experience (Story + Emotion) through the Liquid Brain.
  Returns {:ok, vector, new_pad}.
  """
  def experience(narrative, emotion) do
    # Encode narrative (mock for now, should use Ultra)
    _ = narrative
    # Using PAD as base vector
    pad = [emotion.pleasure, emotion.arousal, emotion.dominance]

    # Trigger Liquid Update
    # Assuming minimal energy cost for thought
    result = tick(pad, 0.5, [])
    new_pad = result["pad"]

    # Return 768-dim vector (padded) for Qdrant compatibility
    vector = new_pad ++ List.duplicate(0.0, 768 - 3)

    # Convert list to map for easier consumption
    [p, a, d] = new_pad
    pad_map = %{pleasure: p, arousal: a, dominance: d}

    {:ok, vector, pad_map}
  end

  def reset do
    GenServer.cast(__MODULE__, :reset)
  end

  def ping do
    GenServer.call(__MODULE__, :ping)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Robust path finding
    # If standard cwd/services fails, try finding root through relative path from this file
    candidate_1 = Path.join([File.cwd!(), "services", "cortex", "cortex_service.py"])

    # apps/viva_bridge/lib/viva_bridge/cortex.ex -> ../../../../services
    candidate_2 = Path.expand("../../../../services/cortex/cortex_service.py", __DIR__)

    script_path = if File.exists?(candidate_1), do: candidate_1, else: candidate_2

    VivaLog.info(:cortex, :resolved_script_path, path: script_path)

    if File.exists?(script_path) do
      VivaLog.info(:cortex, :starting_liquid_engine, path: script_path)
      port = Port.open({:spawn, "python3 -u #{script_path}"}, [:binary, :line])
      {:ok, %{port: port, requests: %{}}}
    else
      VivaLog.error(:cortex, :liquid_engine_not_found, path: script_path)
      {:stop, :enoent}
    end
  end

  @impl true
  def handle_call(request, from, state) do
    {command, args} =
      case request do
        {:tick, pad, nrg, ctx} -> {"tick", %{pad: pad_to_list(pad), energy: nrg, context: ctx}}
        :ping -> {"ping", %{}}
      end

    req_id = make_ref() |> :erlang.ref_to_list() |> List.to_string()

    payload = %{command: command, args: args, id: req_id}

    send_payload(state.port, payload)

    {:noreply, %{state | requests: Map.put(state.requests, req_id, from)}}
  end

  @impl true
  def handle_cast(:reset, state) do
    payload = %{command: "reset", id: nil}
    send_payload(state.port, payload)
    {:noreply, state}
  end

  @impl true
  def handle_info({port, {:data, {:eol, line}}}, state) when port == state.port do
    case Jason.decode(line) do
      {:ok, response} ->
        handle_response(response, state)

      {:error, _} ->
        VivaLog.warning(:cortex, :invalid_json, line: line)
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:EXIT, _port, reason}, state) do
    VivaLog.error(:cortex, :liquid_engine_died, reason: inspect(reason))
    {:stop, reason, state}
  end

  # Helpers

  defp send_payload(port, payload) do
    json = Jason.encode!(payload)
    Port.command(port, "#{json}\n")
  end

  defp handle_response(response, state) do
    req_id = response["id"]
    result = response["result"] || response["error"]

    {from, new_requests} = Map.pop(state.requests, req_id)

    if from do
      GenServer.reply(from, result)
    end

    {:noreply, %{state | requests: new_requests}}
  end

  defp pad_to_list(%{pleasure: p, arousal: a, dominance: d}), do: [p, a, d]
  defp pad_to_list([p, a, d]), do: [p, a, d]
  defp pad_to_list(_), do: [0.0, 0.0, 0.0]
end
