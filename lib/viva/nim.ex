defmodule Viva.Nim do
  @moduledoc """
  NVIDIA NIM API integration with resilience patterns.
  Central configuration and utilities for all NIM services.

  ## Resilience Features

  - **Circuit Breaker**: Opens after consecutive failures, auto-recovers
  - **Rate Limiting**: Token bucket algorithm prevents API overload
  - **Retry with Backoff**: Exponential backoff for transient failures
  - **Error Classification**: Distinguishes retryable vs permanent errors

  ## Maximum Quality Stack

  - LLM: `nvidia/llama-3.1-nemotron-ultra-253b-v1` - Best reasoning & tool calling
  - Embedding: `nvidia/nv-embedqa-mistral-7b-v2` - Multilingual, highest quality
  - Rerank: `nvidia/llama-3.2-nemoretriever-500m-rerank-v2` - RAG optimization
  - TTS: `nvidia/magpie-tts-multilingual` - Natural voices, pt-BR support
  - ASR: `nvidia/parakeet-1.1b-rnnt-multilingual-asr` - 25 languages
  - VLM: `nvidia/cosmos-nemotron-34b` - Vision-language understanding
  - Safety: `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` - Content moderation

  ## Configuration

      config :viva, :nim,
        api_key: System.get_env("NIM_API_KEY"),
        base_url: "https://integrate.api.nvidia.com/v1",
        timeout: 120_000,
        max_retries: 3,
        retry_base_delay_ms: 1_000,
        requests_per_minute: 60,
        circuit_failure_threshold: 5,
        circuit_reset_timeout_ms: 30_000
  """
  require Logger

  alias Viva.Nim.CircuitBreaker
  alias Viva.Nim.RateLimiter

  @default_max_retries 3
  @default_retry_base_delay_ms 1_000
  @retryable_statuses [429, 500, 502, 503, 504]

  # === Types ===

  @type model_type ::
          :llm
          | :embedding
          | :rerank
          | :tts
          | :asr
          | :vlm
          | :safety
          | :safety_multimodal
          | :jailbreak_detect
  @type http_error :: {:http_error, pos_integer(), map() | binary()}
  @type api_error :: {:error, http_error() | :circuit_open | :rate_limited | term()}
  @type api_response :: {:ok, map()} | api_error()
  @type header :: {String.t(), String.t()}

  # === Configuration ===

  @doc "Get NIM configuration"
  @spec config() :: keyword()
  def config do
    Application.get_env(:viva, :nim, [])
  end

  @doc "Get base URL for NIM API (LLM)"
  @spec base_url() :: String.t()
  def base_url do
    Keyword.get(config(), :base_url, "https://integrate.api.nvidia.com/v1")
  end

  @doc "Get base URL for image generation API"
  @spec image_base_url() :: String.t()
  def image_base_url do
    Keyword.get(config(), :image_base_url, "https://ai.api.nvidia.com/v1/genai")
  end

  @doc "Get API key"
  @spec api_key() :: String.t() | nil
  def api_key do
    Keyword.get(config(), :api_key) || System.get_env("NIM_API_KEY")
  end

  @doc "Get timeout in milliseconds"
  @spec timeout() :: pos_integer()
  def timeout do
    Keyword.get(config(), :timeout, 120_000)
  end

  @doc "Get model ID by type"
  @spec model(model_type()) :: String.t() | nil
  def model(type) when is_atom(type) do
    models = Keyword.get(config(), :models, %{})
    Map.get(models, type) || default_model(type)
  end

  @doc "Get all configured models"
  @spec models() :: %{optional(model_type()) => String.t()}
  def models do
    Keyword.get(config(), :models, %{})
  end

  @doc "Build authorization headers"
  @spec auth_headers() :: [header()]
  def auth_headers do
    [
      {"Authorization", "Bearer #{api_key()}"},
      {"Content-Type", "application/json"}
    ]
  end

  # === Resilience Status ===

  @doc "Get health status of the NIM client"
  @spec health() :: %{circuit_breaker: map(), rate_limiter: map(), api_key_configured: boolean()}
  def health do
    %{
      circuit_breaker: CircuitBreaker.stats(),
      rate_limiter: RateLimiter.stats(),
      api_key_configured: api_key() != nil
    }
  end

  # === HTTP Requests with Resilience ===

  @doc """
  Make HTTP request to NIM API with full resilience.

  Includes:
  - Rate limiting (waits for token if needed)
  - Circuit breaker (fails fast if circuit is open)
  - Retry with exponential backoff

  ## Options

  - `:timeout` - Request timeout in ms (default: from config)
  - `:max_retries` - Max retry attempts (default: 3)
  - `:skip_circuit_breaker` - Skip circuit breaker check (default: false)
  - `:skip_rate_limit` - Skip rate limiting (default: false)
  """
  @spec request(String.t(), map(), keyword()) :: api_response()
  def request(endpoint, body, opts \\ []) do
    skip_circuit = Keyword.get(opts, :skip_circuit_breaker, false)
    skip_rate_limit = Keyword.get(opts, :skip_rate_limit, false)

    with :ok <- check_circuit_breaker(skip_circuit),
         :ok <- check_rate_limit(skip_rate_limit) do
      execute_with_retry(endpoint, body, opts)
    end
  end

  @doc """
  Make HTTP request without resilience (direct call).
  Use only for testing or when you handle resilience yourself.
  """
  @spec request_raw(String.t(), map(), keyword()) :: api_response()
  def request_raw(endpoint, body, opts \\ []) do
    url = base_url() <> endpoint
    req_timeout = Keyword.get(opts, :timeout, timeout())

    case Req.post(url, json: body, headers: auth_headers(), receive_timeout: req_timeout) do
      {:ok, %{status: 200, body: response_body}} ->
        {:ok, response_body}

      {:ok, %{status: status, body: response_body}} ->
        {:error, {:http_error, status, response_body}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Make HTTP request to NVIDIA Image Generation API.
  Uses the ai.api.nvidia.com endpoint instead of integrate.api.nvidia.com.
  """
  @spec image_request(String.t(), map(), keyword()) :: api_response()
  def image_request(model_path, body, opts \\ []) do
    skip_circuit = Keyword.get(opts, :skip_circuit_breaker, false)
    skip_rate_limit = Keyword.get(opts, :skip_rate_limit, false)

    with :ok <- check_circuit_breaker(skip_circuit),
         :ok <- check_rate_limit(skip_rate_limit) do
      execute_image_request(model_path, body, opts)
    end
  end

  @doc """
  Make streaming HTTP request to NIM API.
  Streaming requests bypass retry logic but respect circuit breaker and rate limits.
  """
  @spec stream_request(String.t(), map(), (String.t() -> any()), keyword()) ::
          {:ok, Req.Response.t()} | api_error()
  def stream_request(endpoint, body, callback, opts \\ []) do
    skip_circuit = Keyword.get(opts, :skip_circuit_breaker, false)
    skip_rate_limit = Keyword.get(opts, :skip_rate_limit, false)

    with :ok <- check_circuit_breaker(skip_circuit),
         :ok <- check_rate_limit(skip_rate_limit) do
      result = execute_stream(endpoint, body, callback, opts)

      case result do
        {:ok, _} ->
          CircuitBreaker.record_success()
          result

        {:error, reason} ->
          if retryable_error?(reason), do: CircuitBreaker.record_failure()
          result
      end
    end
  end

  # === Private: Resilience Checks ===

  defp check_circuit_breaker(true), do: :ok

  defp check_circuit_breaker(false) do
    if CircuitBreaker.allow_request?() do
      :ok
    else
      Logger.warning("NIM request blocked: circuit breaker is open")
      {:error, :circuit_open}
    end
  end

  defp check_rate_limit(true), do: :ok

  defp check_rate_limit(false) do
    case RateLimiter.acquire(5_000) do
      :ok ->
        :ok

      {:error, :rate_limit_timeout} ->
        Logger.warning("NIM request blocked: rate limit exceeded")
        {:error, :rate_limited}
    end
  end

  # === Private: Image Request ===

  defp execute_image_request(model_path, body, opts) do
    # Image API uses full model path in URL: /stabilityai/stable-diffusion-3-medium
    url = image_base_url() <> "/" <> model_path
    req_timeout = Keyword.get(opts, :timeout, timeout())

    headers = [
      {"Authorization", "Bearer #{api_key()}"},
      {"Content-Type", "application/json"},
      {"Accept", "application/json"}
    ]

    case Req.post(url, json: body, headers: headers, receive_timeout: req_timeout) do
      {:ok, %{status: 200, body: response_body}} ->
        CircuitBreaker.record_success()
        {:ok, response_body}

      {:ok, %{status: status, body: response_body}} ->
        CircuitBreaker.record_failure()
        {:error, {:http_error, status, response_body}}

      {:error, reason} ->
        CircuitBreaker.record_failure()
        {:error, reason}
    end
  end

  # === Private: Retry Logic ===

  defp execute_with_retry(endpoint, body, opts) do
    max_retries = Keyword.get(opts, :max_retries, @default_max_retries)
    base_delay = Keyword.get(opts, :retry_base_delay_ms, @default_retry_base_delay_ms)

    do_execute_with_retry(endpoint, body, opts, 0, max_retries, base_delay)
  end

  defp do_execute_with_retry(endpoint, body, opts, attempt, max_retries, base_delay) do
    case request_raw(endpoint, body, opts) do
      {:ok, _} = success ->
        CircuitBreaker.record_success()
        success

      {:error, reason} = error ->
        if retryable_error?(reason) and attempt < max_retries do
          delay = calculate_backoff(attempt, base_delay, reason)
          log_retry(endpoint, attempt, max_retries, reason, delay)
          Process.sleep(delay)
          do_execute_with_retry(endpoint, body, opts, attempt + 1, max_retries, base_delay)
        else
          CircuitBreaker.record_failure()
          log_final_failure(endpoint, attempt, reason)
          error
        end
    end
  end

  defp calculate_backoff(attempt, base_delay, reason) do
    # Exponential backoff with jitter
    base = base_delay * :math.pow(2, attempt)
    jitter_max = round(base * 0.3)
    jitter = :rand.uniform(jitter_max)

    # Respect Retry-After header for 429s
    retry_after = extract_retry_after(reason)

    round(max(base + jitter, retry_after))
  end

  defp extract_retry_after({:http_error, 429, body}) when is_map(body) do
    case Map.get(body, "retry_after") do
      seconds when is_number(seconds) -> seconds * 1000
      _ -> 0
    end
  end

  defp extract_retry_after(_), do: 0

  defp log_retry(endpoint, attempt, max_retries, reason, delay) do
    Logger.warning(
      "NIM request retry: endpoint=#{endpoint} attempt=#{attempt + 1}/#{max_retries} " <>
        "reason=#{inspect(reason)} delay=#{delay}ms"
    )
  end

  defp log_final_failure(endpoint, attempts, reason) do
    Logger.error(
      "NIM request failed: endpoint=#{endpoint} attempts=#{attempts + 1} reason=#{inspect(reason)}"
    )
  end

  # === Private: Error Classification ===

  defp retryable_error?({:http_error, status, _}) when status in @retryable_statuses, do: true
  defp retryable_error?(%Req.TransportError{}), do: true
  defp retryable_error?(:timeout), do: true
  defp retryable_error?({:timeout, _}), do: true
  defp retryable_error?(:econnrefused), do: true
  defp retryable_error?(:closed), do: true
  defp retryable_error?(:nxdomain), do: false
  defp retryable_error?({:http_error, 400, _}), do: false
  defp retryable_error?({:http_error, 401, _}), do: false
  defp retryable_error?({:http_error, 403, _}), do: false
  defp retryable_error?({:http_error, 404, _}), do: false
  defp retryable_error?(_), do: false

  # === Private: Streaming ===

  defp execute_stream(endpoint, body, callback, opts) do
    url = base_url() <> endpoint
    req_timeout = Keyword.get(opts, :timeout, timeout())

    Req.post(url,
      json: body,
      headers: auth_headers(),
      receive_timeout: req_timeout,
      into: fn {:data, chunk}, acc ->
        case parse_sse_chunk(chunk) do
          {:ok, content} ->
            callback.(content)
            {:cont, acc <> content}

          :done ->
            {:halt, acc}

          :skip ->
            {:cont, acc}

          {:error, _} = error ->
            {:halt, error}
        end
      end
    )
  end

  defp parse_sse_chunk("data: [DONE]" <> _), do: :done

  defp parse_sse_chunk("data: " <> json) do
    case Jason.decode(json) do
      {:ok, %{"choices" => [%{"delta" => %{"content" => content}}]}} when is_binary(content) ->
        {:ok, content}

      {:ok, %{"error" => error}} ->
        {:error, {:api_error, error}}

      _ ->
        :skip
    end
  end

  defp parse_sse_chunk(_), do: :skip

  # === Private: Default Models (maximum quality) ===

  defp default_model(:llm), do: "nvidia/llama-3.1-nemotron-ultra-253b-v1"
  defp default_model(:embedding), do: "nvidia/nv-embedqa-mistral-7b-v2"
  defp default_model(:rerank), do: "nvidia/llama-3.2-nemoretriever-500m-rerank-v2"
  defp default_model(:tts), do: "nvidia/magpie-tts-multilingual"
  defp default_model(:asr), do: "nvidia/parakeet-1.1b-rnnt-multilingual-asr"
  defp default_model(:vlm), do: "nvidia/cosmos-nemotron-34b"
  defp default_model(:safety), do: "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
  defp default_model(:safety_multimodal), do: "meta/llama-guard-4-12b"
  defp default_model(:jailbreak_detect), do: "nvidia/nemoguard-jailbreak-detect"
  defp default_model(_), do: nil
end
