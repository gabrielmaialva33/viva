defmodule Viva.AI.LLM.Backends.GroqBackend do
  @moduledoc """
  Backend cloud para LLM usando Groq API.

  Groq oferece inferência ultra-rápida (~100ms) com modelos de alta qualidade.
  Free tier: 30 RPM, 14400 RPD, modelos Llama 3.3 70B.

  Configuração via config:
    config :viva, Viva.AI.LLM.Backends.GroqBackend,
      api_key: System.get_env("GROQ_API_KEY"),
      model: "llama-3.3-70b-versatile",
      timeout: 30_000
  """

  @behaviour Viva.AI.LLM.ClientBehaviour

  require Logger

  @base_url "https://api.groq.com/openai/v1"
  @default_model "llama-3.3-70b-versatile"
  @default_timeout 30_000

  # Avatar thought generation settings
  @thought_settings %{
    temperature: 0.8,
    top_p: 0.9,
    max_tokens: 150
  }

  @doc """
  Gera uma resposta para um prompt simples.

  ## Options
    * `:model` - Modelo Groq (default: llama-3.3-70b-versatile)
    * `:temperature` - Criatividade (0.0-2.0, default: 0.8)
    * `:max_tokens` - Máximo de tokens na resposta
    * `:timeout` - Timeout em ms (default: 30_000)
  """
  @impl true
  def generate(prompt, opts \\ []) do
    messages = [%{role: "user", content: prompt}]
    chat(messages, opts)
  end

  @doc """
  Processa uma conversa com múltiplas mensagens.

  ## Options
    * `:model` - Modelo Groq a usar
    * `:temperature` - Criatividade (0.0-2.0)
    * `:max_tokens` - Máximo de tokens na resposta
    * `:system` - System prompt opcional
  """
  @impl true
  def chat(messages, opts \\ []) when is_list(messages) do
    model = opts[:model] || config(:model, @default_model)
    timeout = opts[:timeout] || config(:timeout, @default_timeout)

    # Add system prompt if provided
    messages = case Keyword.get(opts, :system) do
      nil -> messages
      system -> [%{role: "system", content: system} | messages]
    end

    # Normalize message format
    normalized_messages = Enum.map(messages, fn
      %{role: role, content: content} -> %{role: to_string(role), content: content}
      %{"role" => role, "content" => content} -> %{role: role, content: content}
    end)

    body = %{
      model: model,
      messages: normalized_messages,
      temperature: Keyword.get(opts, :temperature, 0.7),
      max_tokens: Keyword.get(opts, :max_tokens, 1024),
      stream: false
    }

    case make_request("/chat/completions", body, timeout) do
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        {:ok, String.trim(content)}

      {:ok, response} ->
        Logger.warning("[GroqBackend] Unexpected response: #{inspect(response)}")
        {:error, :invalid_response}

      {:error, reason} = error ->
        Logger.error("[GroqBackend] Chat failed: #{inspect(reason)}")
        error
    end
  end

  @doc """
  Gera um pensamento de avatar otimizado para velocidade.
  """
  def generate_thought(prompt, opts \\ []) do
    merged_opts = Keyword.merge([
      temperature: @thought_settings.temperature,
      top_p: @thought_settings.top_p,
      max_tokens: @thought_settings.max_tokens
    ], opts)

    generate(prompt, merged_opts)
  end

  @doc """
  Verifica se o backend Groq está disponível.
  """
  def health_check do
    case config(:api_key) do
      nil ->
        {:error, :no_api_key}

      api_key when is_binary(api_key) ->
        url = "#{@base_url}/models"
        headers = [{"Authorization", "Bearer #{api_key}"}]

        case Req.get(url, headers: headers, receive_timeout: 5_000) do
          {:ok, %Req.Response{status: 200}} ->
            {:ok, :healthy}

          {:ok, %Req.Response{status: 401}} ->
            {:error, :invalid_api_key}

          {:ok, %Req.Response{status: status}} ->
            {:error, {:http_error, status}}

          {:error, reason} ->
            {:error, {:connection_failed, reason}}
        end
    end
  end

  @doc """
  Lista modelos disponíveis no Groq.
  """
  def list_models do
    api_key = config(:api_key)
    url = "#{@base_url}/models"
    headers = [{"Authorization", "Bearer #{api_key}"}]

    case Req.get(url, headers: headers, receive_timeout: 5_000) do
      {:ok, %Req.Response{status: 200, body: %{"data" => models}}} ->
        {:ok, Enum.map(models, & &1["id"])}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Private functions

  defp make_request(endpoint, body, timeout) do
    api_key = config(:api_key)

    if is_nil(api_key) do
      {:error, :no_api_key}
    else
      url = "#{@base_url}#{endpoint}"
      headers = [
        {"Authorization", "Bearer #{api_key}"},
        {"Content-Type", "application/json"}
      ]

      start = System.monotonic_time(:millisecond)

      result = Req.post(url,
        json: body,
        headers: headers,
        receive_timeout: timeout,
        connect_options: [timeout: 5_000]
      )

      elapsed = System.monotonic_time(:millisecond) - start

      case result do
        {:ok, %Req.Response{status: 200, body: response_body}} ->
          Logger.debug("[GroqBackend] Request completed in #{elapsed}ms")
          {:ok, response_body}

        {:ok, %Req.Response{status: 429, body: body}} ->
          retry_after = get_retry_after(body)
          Logger.warning("[GroqBackend] Rate limited, retry after #{retry_after}s")
          {:error, {:rate_limited, retry_after}}

        {:ok, %Req.Response{status: status, body: body}} ->
          {:error, {:http_error, status, body}}

        {:error, %Req.TransportError{reason: :timeout}} ->
          {:error, :timeout}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp get_retry_after(%{"error" => %{"message" => msg}}) do
    # Parse "Please try again in Xs" from error message
    case Regex.run(~r/try again in (\d+)/, msg) do
      [_, seconds] -> String.to_integer(seconds)
      _ -> 60
    end
  end
  defp get_retry_after(_), do: 60

  defp config(key, default \\ nil) do
    Application.get_env(:viva, __MODULE__, [])
    |> Keyword.get(key, default)
  end
end
