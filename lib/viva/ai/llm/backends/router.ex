defmodule Viva.AI.LLM.Backends.Router do
  @moduledoc """
  Router inteligente com fallback chain: Groq → Gemini CLI → NIM.

  ## Estratégia de Roteamento

  1. **Groq** (primário) - Ultra-rápido (~700ms):
     - Pensamentos de avatares (alta frequência)
     - Respostas curtas (< 500 tokens)
     - Rate limit: 30 RPM (FREE)

  2. **Gemini CLI** (fallback) - Rápido (~1500ms):
     - Fallback quando Groq falha ou rate limit
     - Rate limit: 15 RPM (FREE)

  3. **NIM** (cloud) - Para tarefas complexas:
     - Raciocínio complexo
     - Análise de sentimento/safety crítica
     - Rate limit: 40 RPM (trial)

  ## Configuração

      config :viva, Viva.AI.LLM.Backends.Router,
        strategy: :local_first,
        local_max_tokens: 500,
        fallback_on_error: true
  """

  @behaviour Viva.AI.LLM.ClientBehaviour

  require Logger

  alias Viva.AI.LLM.Backends.GroqBackend
  alias Viva.AI.LLM.Backends.GeminiCliBackend
  alias Viva.AI.LLM.LlmClient

  @default_strategy :local_first
  @local_max_tokens 500

  # Tipos de tarefa que devem ir para local
  @local_tasks [:thought, :reflection, :simple_response, :greeting]

  # Tipos que devem ir para cloud (mais capacidade)
  @cloud_tasks [:reasoning, :analysis, :safety_check, :complex_dialogue]

  @doc """
  Gera resposta roteando automaticamente para o backend apropriado.

  ## Options
    * `:task_type` - Tipo de tarefa (:thought, :reasoning, etc.)
    * `:force_backend` - Força uso de :local ou :cloud
    * `:max_tokens` - Limita tokens (afeta roteamento)
    * Demais opções passadas ao backend
  """
  @impl true
  def generate(prompt, opts \\ []) do
    {routing_opts, backend_opts} = split_options(opts)
    backend = select_backend(:generate, prompt, routing_opts)

    Logger.debug("[Router] Routing generate to #{inspect(backend)}")

    case execute_with_fallback(backend, :generate, [prompt, backend_opts]) do
      {:ok, _} = result -> result
      {:error, _} = error -> error
    end
  end

  @doc """
  Processa chat roteando automaticamente.
  """
  @impl true
  def chat(messages, opts \\ []) when is_list(messages) do
    {routing_opts, backend_opts} = split_options(opts)
    backend = select_backend(:chat, messages, routing_opts)

    Logger.debug("[Router] Routing chat to #{inspect(backend)}")

    case execute_with_fallback(backend, :chat, [messages, backend_opts]) do
      {:ok, _} = result -> result
      {:error, _} = error -> error
    end
  end

  @doc """
  Gera pensamento de avatar com fallback chain: Groq → Gemini → NIM.
  Otimizado para alta frequência e baixa latência.
  """
  def generate_thought(prompt, opts \\ []) do
    opts = Keyword.put(opts, :task_type, :thought)

    # Tenta Groq primeiro (mais rápido)
    case GroqBackend.generate_thought(prompt, opts) do
      {:ok, _} = result ->
        result

      {:error, groq_reason} ->
        Logger.warning("[Router] Groq failed: #{inspect(groq_reason)}, trying Gemini CLI")

        # Fallback para Gemini CLI
        case GeminiCliBackend.generate_thought(prompt, opts) do
          {:ok, _} = result ->
            result

          {:error, gemini_reason} ->
            Logger.warning("[Router] Gemini CLI failed: #{inspect(gemini_reason)}, trying NIM")
            # Último fallback: NIM
            LlmClient.generate(prompt, opts)
        end
    end
  end

  @doc """
  Raciocínio complexo (prefere cloud, fallback Groq).
  """
  def generate_reasoning(prompt, opts \\ []) do
    opts = Keyword.put(opts, :task_type, :reasoning)

    case LlmClient.generate(prompt, opts) do
      {:ok, _} = result ->
        result

      {:error, reason} ->
        Logger.warning("[Router] Cloud failed for reasoning: #{inspect(reason)}, trying Groq")
        GroqBackend.generate(prompt, opts)
    end
  end

  @doc """
  Retorna estatísticas de todos os backends.
  """
  def status do
    groq_status = case GroqBackend.health_check() do
      {:ok, :healthy} -> :healthy
      {:error, reason} -> {:unhealthy, reason}
    end

    gemini_status = case GeminiCliBackend.health_check() do
      {:ok, :healthy} -> :healthy
      {:error, reason} -> {:unhealthy, reason}
    end

    # Para o cloud, verificamos o circuit breaker
    nim_status = case check_cloud_health() do
      :ok -> :healthy
      {:error, reason} -> {:unhealthy, reason}
    end

    %{
      groq: groq_status,
      gemini: gemini_status,
      nim: nim_status,
      strategy: config(:strategy, @default_strategy)
    }
  end

  @doc """
  Atualiza a estratégia de roteamento em runtime.
  """
  def set_strategy(strategy) when strategy in [:local_first, :cloud_first, :local_only, :cloud_only] do
    Application.put_env(:viva, __MODULE__,
      Keyword.put(config_all(), :strategy, strategy)
    )
    Logger.info("[Router] Strategy changed to #{strategy}")
    :ok
  end

  # Seleção de backend baseada em heurísticas

  defp select_backend(_operation, _input, opts) do
    strategy = config(:strategy, @default_strategy)
    task_type = Keyword.get(opts, :task_type)
    force_backend = Keyword.get(opts, :force_backend)
    max_tokens = Keyword.get(opts, :max_tokens, @local_max_tokens)

    cond do
      # Força backend específico
      force_backend == :local -> :local
      force_backend == :cloud -> :cloud

      # Estratégias fixas
      strategy == :local_only -> :local
      strategy == :cloud_only -> :cloud

      # Roteamento por tipo de tarefa
      task_type in @local_tasks -> :local
      task_type in @cloud_tasks -> :cloud

      # Roteamento por tamanho
      max_tokens > config(:local_max_tokens, @local_max_tokens) -> :cloud

      # Estratégias com fallback
      strategy == :local_first -> :local
      strategy == :cloud_first -> :cloud

      # Default
      true -> :local
    end
  end

  defp execute_with_fallback(backend, operation, args) do
    primary = get_backend_module(backend)
    fallback_enabled = config(:fallback_on_error, true)

    case apply(primary, operation, args) do
      {:ok, _} = result ->
        result

      {:error, reason} ->
        if fallback_enabled do
          fallback = get_fallback_module(backend)
          Logger.warning("[Router] #{inspect(primary)} failed: #{inspect(reason)}, falling back to #{inspect(fallback)}")
          apply(fallback, operation, args)
        else
          {:error, reason}
        end
    end
  end

  defp get_backend_module(:local), do: GroqBackend
  defp get_backend_module(:cloud), do: LlmClient

  # Fallback chain: local → gemini → cloud, cloud → groq → gemini
  defp get_fallback_module(:local), do: GeminiCliBackend
  defp get_fallback_module(:cloud), do: GroqBackend

  defp check_cloud_health do
    # Verifica se circuit breaker está aberto via stats
    case Viva.AI.LLM.CircuitBreaker.stats() do
      %{state: :closed} -> :ok
      %{state: :open} -> {:error, :circuit_open}
      %{state: :half_open} -> :ok
      _ -> :ok
    end
  rescue
    _ -> :ok  # Se CircuitBreaker não existe, assume OK
  end

  defp split_options(opts) do
    routing_keys = [:task_type, :force_backend]
    {Keyword.take(opts, routing_keys), Keyword.drop(opts, routing_keys)}
  end

  defp config(key, default) do
    config_all() |> Keyword.get(key, default)
  end

  defp config_all do
    Application.get_env(:viva, __MODULE__, [])
  end
end
