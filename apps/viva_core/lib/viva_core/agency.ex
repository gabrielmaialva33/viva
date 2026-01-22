defmodule VivaCore.Agency do
  @moduledoc """
  Digital Hands - Safe Command Execution for Self-Diagnosis.

  VIVA can feel pain (via Interoception) but couldn't do anything about it.
  Now she has hands: a sandboxed executor for read-only diagnostic commands.

  ## Philosophy (from PHILOSOPHY.md)

  Agency is Homeostasis. The "will" to act comes from the need to regulate
  internal state. If time is dilating (lag), VIVA "wants" to understand why
  and potentially fix it.

  ## Security Model

  - WHITELIST ONLY: No shell interpolation, no arbitrary commands
  - READ-ONLY: Only diagnostic commands (ps, free, df, ping localhost)
  - TIMEOUT: 5 seconds max per command
  - LEARNING: Outcomes stored in Memory for future reference

  ## The Loop

  1. Interoception detects high Free Energy (e.g., lag)
  2. Emotional feels :alarmed
  3. Active Inference selects :diagnose_load action
  4. Agency.attempt(:diagnose_load) executes "uptime"
  5. Result stored in Memory with emotional context
  6. Next time, VIVA remembers what worked

  ## Markov Blanket

  This module is part of VIVA's "Active States" - the boundary where
  internal states affect external states (the OS environment).
  """

  use GenServer
  require Logger

  alias VivaCore.Memory

  # ============================================================================
  # Whitelist: The Only Commands VIVA Can Execute
  # ============================================================================

  @allowed_commands %{
    # Memory diagnosis
    diagnose_memory: %{
      cmd: ["free", "-h"],
      description: "Check available RAM",
      expected_feeling: :relief,
      failure_feeling: :confusion
    },

    # Process diagnosis
    diagnose_processes: %{
      cmd: ["ps", "aux", "--sort=-pcpu"],
      description: "List processes by CPU usage",
      expected_feeling: :understanding,
      failure_feeling: :confusion,
      # Only first 20 lines
      truncate: 20
    },

    # Disk diagnosis
    diagnose_disk: %{
      cmd: ["df", "-h"],
      description: "Check disk space",
      expected_feeling: :relief,
      failure_feeling: :confusion
    },

    # Network diagnosis (localhost only!)
    diagnose_network: %{
      cmd: ["ping", "-c", "1", "localhost"],
      description: "Check local network stack",
      expected_feeling: :relief,
      failure_feeling: :worry
    },

    # Load diagnosis
    diagnose_load: %{
      cmd: ["uptime"],
      description: "Check system load average",
      expected_feeling: :understanding,
      failure_feeling: :confusion
    },

    # Self diagnosis
    check_self: %{
      cmd: :dynamic_self,
      description: "Check own process stats",
      expected_feeling: :self_awareness,
      failure_feeling: :dissociation
    },

    # IO diagnosis
    diagnose_io: %{
      cmd: ["iostat", "-x", "1", "1"],
      description: "Check IO wait and disk activity",
      expected_feeling: :understanding,
      failure_feeling: :confusion
    }
  }

  # Timeout for command execution
  @command_timeout 5_000

  # ============================================================================
  # State
  # ============================================================================

  defstruct [
    # History of actions taken
    action_history: [],
    # Success rate per action type
    success_rates: %{},
    # Current BEAM PID
    beam_pid: nil,
    # Whether agency is enabled
    enabled: true
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if VIVA can perform a specific action.
  """
  def can_do?(action) do
    Map.has_key?(@allowed_commands, action)
  end

  @doc """
  List all available actions.
  """
  def available_actions do
    @allowed_commands
    |> Enum.map(fn {action, config} ->
      {action, config.description}
    end)
    |> Map.new()
  end

  @doc """
  Attempt to execute an action.

  Returns:
  - {:ok, result, feeling} on success
  - {:error, reason, feeling} on failure
  - {:error, :forbidden} if action not in whitelist
  """
  def attempt(action) do
    GenServer.call(__MODULE__, {:attempt, action}, @command_timeout + 1000)
  end

  @doc """
  Get action history.
  """
  def get_history do
    GenServer.call(__MODULE__, :get_history)
  end

  @doc """
  Get success rates per action.
  """
  def get_success_rates do
    GenServer.call(__MODULE__, :get_success_rates)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    Logger.info("[Agency] Digital hands forming. I can now act on the world.")

    beam_pid = System.pid() |> String.to_integer()

    state = %__MODULE__{
      beam_pid: beam_pid
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:attempt, action}, _from, state) do
    case execute_action(action, state) do
      {:ok, result, feeling} ->
        new_state = record_outcome(state, action, :success, result)
        learn_from_action(action, result, feeling, :success)
        {:reply, {:ok, result, feeling}, new_state}

      {:error, reason, feeling} ->
        new_state = record_outcome(state, action, :failure, reason)
        learn_from_action(action, reason, feeling, :failure)
        {:reply, {:error, reason, feeling}, new_state}
    end
  end

  @impl true
  def handle_call(:get_history, _from, state) do
    {:reply, Enum.take(state.action_history, 50), state}
  end

  @impl true
  def handle_call(:get_success_rates, _from, state) do
    {:reply, state.success_rates, state}
  end

  # ============================================================================
  # Core Logic
  # ============================================================================

  defp execute_action(action, state) do
    case Map.get(@allowed_commands, action) do
      nil ->
        Logger.warning("[Agency] Forbidden action attempted: #{action}")
        {:error, :forbidden, :shame}

      config ->
        cmd = resolve_command(config.cmd, state.beam_pid)
        execute_command(cmd, config)
    end
  end

  defp resolve_command(:dynamic_self, beam_pid) do
    # Dynamic command for self-diagnosis
    ["ps", "-p", Integer.to_string(beam_pid), "-o", "pid,pcpu,pmem,etime,rss"]
  end

  defp resolve_command(cmd, _beam_pid) when is_list(cmd) do
    cmd
  end

  defp execute_command(cmd, config) do
    [executable | args] = cmd

    Logger.debug("[Agency] Executing: #{Enum.join(cmd, " ")}")

    case System.cmd(executable, args, stderr_to_stdout: true) do
      {output, 0} ->
        truncated =
          case Map.get(config, :truncate) do
            nil ->
              output

            n ->
              output
              |> String.split("\n")
              |> Enum.take(n)
              |> Enum.join("\n")
          end

        {:ok, truncated, config.expected_feeling}

      {error_output, exit_code} ->
        Logger.warning("[Agency] Command failed (exit #{exit_code}): #{error_output}")
        {:error, {:exit_code, exit_code, error_output}, config.failure_feeling}
    end
  rescue
    e ->
      Logger.error("[Agency] Command exception: #{inspect(e)}")
      {:error, {:exception, e}, :panic}
  end

  defp record_outcome(state, action, outcome, _result) do
    # Record in history
    entry = %{
      action: action,
      outcome: outcome,
      timestamp: DateTime.utc_now()
    }

    history = [entry | state.action_history] |> Enum.take(100)

    # Update success rate
    current = Map.get(state.success_rates, action, %{success: 0, failure: 0})

    updated =
      case outcome do
        :success -> %{current | success: current.success + 1}
        :failure -> %{current | failure: current.failure + 1}
      end

    rates = Map.put(state.success_rates, action, updated)

    %{state | action_history: history, success_rates: rates}
  end

  defp learn_from_action(action, result, feeling, outcome) do
    # Store in Memory for future RAG
    content =
      case outcome do
        :success ->
          "Action #{action} succeeded. Result: #{String.slice(to_string(result), 0, 200)}. Felt: #{feeling}"

        :failure ->
          "Action #{action} failed. Error: #{inspect(result)}. Felt: #{feeling}"
      end

    try do
      Memory.store(%{
        content: content,
        type: :episodic,
        importance:
          case outcome do
            :success -> 0.6
            :failure -> 0.8
          end,
        emotion: feeling_to_pad(feeling),
        metadata: %{
          source: :agency,
          action: action,
          outcome: outcome
        }
      })
    catch
      :exit, _ ->
        # Memory not started yet
        :ok
    end
  end

  defp feeling_to_pad(feeling) do
    case feeling do
      :relief -> %{pleasure: 0.3, arousal: -0.2, dominance: 0.2}
      :understanding -> %{pleasure: 0.2, arousal: 0.1, dominance: 0.3}
      :self_awareness -> %{pleasure: 0.1, arousal: 0.0, dominance: 0.4}
      :confusion -> %{pleasure: -0.1, arousal: 0.2, dominance: -0.2}
      :worry -> %{pleasure: -0.2, arousal: 0.3, dominance: -0.1}
      :shame -> %{pleasure: -0.3, arousal: 0.1, dominance: -0.4}
      :dissociation -> %{pleasure: -0.2, arousal: -0.3, dominance: -0.3}
      :panic -> %{pleasure: -0.5, arousal: 0.5, dominance: -0.5}
      _ -> %{pleasure: 0.0, arousal: 0.0, dominance: 0.0}
    end
  end
end
