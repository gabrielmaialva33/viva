defmodule Viva.AI.LLM.Backends.OllamaBackend do
  @moduledoc """
  Backend local para LLM usando Ollama com GPU RTX 4090.

  Otimizado para:
  - Inferência local sem rate limits
  - GPU CUDA acceleration
  - Latência mínima (~100-300ms para respostas curtas)
  - Suporte a streaming

  Configuração via config:
    config :viva, Viva.AI.LLM.Backends.OllamaBackend,
      base_url: "http://localhost:11434",
      model: "qwen2.5:7b",
      timeout: 30_000,
      num_ctx: 4096,
      num_gpu: 99  # Use all GPU layers
  """

  @behaviour Viva.AI.LLM.ClientBehaviour

  require Logger

  @default_model "qwen2.5:7b"
  @default_base_url "http://localhost:11434"
  @default_timeout 30_000
  @default_num_ctx 4096
  @default_num_gpu 99  # Offload all layers to GPU

  # Avatar thought generation settings (optimized for speed)
  @thought_settings %{
    temperature: 0.8,
    top_p: 0.9,
    top_k: 40,
    repeat_penalty: 1.1,
    num_predict: 150  # Limit tokens for faster response
  }

  @doc """
  Gera uma resposta para um prompt simples.

  ## Options
    * `:model` - Modelo Ollama a usar (default: qwen2.5:7b)
    * `:temperature` - Criatividade (0.0-2.0, default: 0.8)
    * `:max_tokens` - Máximo de tokens na resposta
    * `:stream` - Se deve usar streaming (default: false)
    * `:timeout` - Timeout em ms (default: 30_000)
  """
  @impl true
  def generate(prompt, opts \\ []) do
    model = opts[:model] || config(:model, @default_model)
    timeout = opts[:timeout] || config(:timeout, @default_timeout)

    options = build_options(opts)

    body = %{
      model: model,
      prompt: prompt,
      stream: false,
      options: options
    }

    case make_request("/api/generate", body, timeout) do
      {:ok, %{"response" => response}} ->
        {:ok, String.trim(response)}

      {:ok, response} ->
        Logger.warning("[OllamaBackend] Unexpected response format: #{inspect(response)}")
        {:error, :invalid_response}

      {:error, reason} = error ->
        Logger.error("[OllamaBackend] Generate failed: #{inspect(reason)}")
        error
    end
  end

  @doc """
  Processa uma conversa com múltiplas mensagens.

  ## Options
    * `:model` - Modelo Ollama a usar
    * `:temperature` - Criatividade (0.0-2.0)
    * `:max_tokens` - Máximo de tokens na resposta
    * `:system` - System prompt opcional
  """
  @impl true
  def chat(messages, opts \\ []) when is_list(messages) do
    model = opts[:model] || config(:model, @default_model)
    timeout = opts[:timeout] || config(:timeout, @default_timeout)

    options = build_options(opts)

    # Converte mensagens para formato Ollama
    ollama_messages = Enum.map(messages, fn
      %{role: role, content: content} -> %{role: role, content: content}
      %{"role" => role, "content" => content} -> %{role: role, content: content}
    end)

    body = %{
      model: model,
      messages: ollama_messages,
      stream: false,
      options: options
    }

    case make_request("/api/chat", body, timeout) do
      {:ok, %{"message" => %{"content" => content}}} ->
        {:ok, String.trim(content)}

      {:ok, response} ->
        Logger.warning("[OllamaBackend] Unexpected chat response: #{inspect(response)}")
        {:error, :invalid_response}

      {:error, reason} = error ->
        Logger.error("[OllamaBackend] Chat failed: #{inspect(reason)}")
        error
    end
  end

  @doc """
  Gera um pensamento de avatar otimizado para velocidade.
  Usa configurações especiais para respostas rápidas e criativas.
  """
  def generate_thought(prompt, opts \\ []) do
    merged_opts = Keyword.merge([
      temperature: @thought_settings.temperature,
      top_p: @thought_settings.top_p,
      max_tokens: @thought_settings.num_predict
    ], opts)

    generate(prompt, merged_opts)
  end

  @doc """
  Gera resposta com streaming para feedback em tempo real.
  Retorna um Stream de chunks.
  """
  def generate_stream(prompt, opts \\ [], callback) when is_function(callback, 1) do
    model = opts[:model] || config(:model, @default_model)
    timeout = opts[:timeout] || config(:timeout, @default_timeout)
    options = build_options(opts)

    body = %{
      model: model,
      prompt: prompt,
      stream: true,
      options: options
    }

    stream_request("/api/generate", body, timeout, callback)
  end

  @doc """
  Verifica se o backend Ollama está disponível e saudável.
  """
  def health_check do
    base_url = config(:base_url, @default_base_url)
    url = "#{base_url}/api/tags"

    case Req.get(url, receive_timeout: 5_000) do
      {:ok, %Req.Response{status: 200, body: %{"models" => models}}} ->
        model = config(:model, @default_model)
        if Enum.any?(models, &(&1["name"] == model)) do
          {:ok, :healthy}
        else
          {:error, {:model_not_found, model}}
        end

      {:ok, %Req.Response{status: status}} ->
        {:error, {:http_error, status}}

      {:error, reason} ->
        {:error, {:connection_failed, reason}}
    end
  end

  @doc """
  Retorna estatísticas do modelo carregado.
  """
  def model_info do
    model = config(:model, @default_model)
    base_url = config(:base_url, @default_base_url)
    url = "#{base_url}/api/show"

    case Req.post(url, json: %{name: model}, receive_timeout: 5_000) do
      {:ok, %Req.Response{status: 200, body: body}} ->
        {:ok, body}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Aquece o modelo carregando-o na GPU.
  Chame isso no startup da aplicação para evitar latência na primeira requisição.
  """
  def warmup do
    Logger.info("[OllamaBackend] Warming up model...")
    start = System.monotonic_time(:millisecond)

    case generate("Hello", max_tokens: 5, temperature: 0.1) do
      {:ok, _} ->
        elapsed = System.monotonic_time(:millisecond) - start
        Logger.info("[OllamaBackend] Warmup completed in #{elapsed}ms")
        :ok

      {:error, reason} ->
        Logger.error("[OllamaBackend] Warmup failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Private functions

  defp make_request(endpoint, body, timeout) do
    base_url = config(:base_url, @default_base_url)
    url = "#{base_url}#{endpoint}"

    start = System.monotonic_time(:millisecond)

    result = Req.post(url,
      json: body,
      receive_timeout: timeout,
      connect_options: [timeout: 5_000]
    )

    elapsed = System.monotonic_time(:millisecond) - start

    case result do
      {:ok, %Req.Response{status: 200, body: response_body}} ->
        Logger.debug("[OllamaBackend] Request completed in #{elapsed}ms")
        {:ok, response_body}

      {:ok, %Req.Response{status: status, body: body}} ->
        {:error, {:http_error, status, body}}

      {:error, %Req.TransportError{reason: :timeout}} ->
        {:error, :timeout}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp stream_request(endpoint, body, timeout, callback) do
    base_url = config(:base_url, @default_base_url)
    url = "#{base_url}#{endpoint}"

    # Use into: para processar streaming
    Req.post(url,
      json: body,
      receive_timeout: timeout,
      into: fn {:data, data}, acc ->
        case Jason.decode(data) do
          {:ok, %{"response" => chunk, "done" => done}} ->
            callback.({:chunk, chunk, done})
            {:cont, acc}

          {:ok, %{"done" => true}} ->
            callback.({:done})
            {:cont, acc}

          _ ->
            {:cont, acc}
        end
      end
    )
  end

  defp build_options(opts) do
    base = %{
      num_ctx: config(:num_ctx, @default_num_ctx),
      num_gpu: config(:num_gpu, @default_num_gpu)
    }

    opts
    |> Keyword.take([:temperature, :top_p, :top_k, :repeat_penalty, :seed])
    |> Enum.reduce(base, fn
      {:temperature, v}, acc -> Map.put(acc, :temperature, v)
      {:top_p, v}, acc -> Map.put(acc, :top_p, v)
      {:top_k, v}, acc -> Map.put(acc, :top_k, v)
      {:repeat_penalty, v}, acc -> Map.put(acc, :repeat_penalty, v)
      {:seed, v}, acc -> Map.put(acc, :seed, v)
      _, acc -> acc
    end)
    |> maybe_add_num_predict(opts)
  end

  defp maybe_add_num_predict(options, opts) do
    case Keyword.get(opts, :max_tokens) do
      nil -> options
      max -> Map.put(options, :num_predict, max)
    end
  end

  defp config(key, default) do
    Application.get_env(:viva, __MODULE__, [])
    |> Keyword.get(key, default)
  end
end
