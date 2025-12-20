defmodule Viva.Nim.LlmClient do
  @moduledoc """
  HTTP client for NVIDIA NIM LLM services.
  Supports Nemotron and other OpenAI-compatible endpoints.
  """
  require Logger

  defmodule Config do
    @moduledoc false
    @default_model "nvidia/nemotron-nano-12b-v2-vl"
    @default_timeout 30_000

    defstruct [
      :base_url,
      :api_key,
      :model,
      :timeout
    ]

    def new(opts \\ []) do
      %__MODULE__{
        base_url:
          opts[:base_url] || System.get_env("NIM_LLM_URL", "https://integrate.api.nvidia.com/v1"),
        api_key: opts[:api_key] || System.get_env("NVIDIA_API_KEY"),
        model: opts[:model] || @default_model,
        timeout: opts[:timeout] || @default_timeout
      }
    end
  end

  @doc """
  Generate text completion from a prompt.
  """
  def generate(prompt, opts \\ []) do
    config = Config.new(opts)

    messages = [
      %{role: "user", content: prompt}
    ]

    chat(messages, opts ++ [config: config])
    |> extract_content()
  end

  @doc """
  Chat completion with message history.
  """
  def chat(messages, opts \\ []) do
    config = Keyword.get(opts, :config) || Config.new(opts)

    body = %{
      model: Keyword.get(opts, :model, config.model),
      messages: messages,
      max_tokens: Keyword.get(opts, :max_tokens, 500),
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.9),
      stream: false
    }

    case do_request(config, "/chat/completions", body) do
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        {:ok, content}

      {:ok, response} ->
        Logger.error("Unexpected LLM response: #{inspect(response)}")
        {:error, :unexpected_response}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Chat completion with streaming.
  Calls the callback function with each chunk.
  """
  def chat_stream(messages, callback, opts \\ []) do
    config = Keyword.get(opts, :config) || Config.new(opts)

    body = %{
      model: Keyword.get(opts, :model, config.model),
      messages: messages,
      max_tokens: Keyword.get(opts, :max_tokens, 500),
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.9),
      stream: true
    }

    do_streaming_request(config, "/chat/completions", body, callback)
  end

  @doc """
  Generate embeddings for text.
  """
  def embed(text, opts \\ []) do
    config = Keyword.get(opts, :config) || Config.new(opts)

    body = %{
      model: Keyword.get(opts, :model, "nvidia/nv-embedqa-e5-v5"),
      input: text,
      input_type: "query"
    }

    case do_request(config, "/embeddings", body) do
      {:ok, %{"data" => [%{"embedding" => embedding} | _]}} ->
        {:ok, embedding}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Analyze a conversation and extract insights.
  """
  def analyze_conversation(conversation, avatar_a, avatar_b) do
    prompt = """
    Analyze this conversation between two AI avatars and evaluate:

    Avatar A personality: #{inspect(avatar_a.personality)}
    Avatar B personality: #{inspect(avatar_b.personality)}

    Conversation:
    #{format_conversation(conversation)}

    Evaluate on a scale of 0.0 to 1.0:
    1. emotional_depth: How deep/meaningful was the conversation?
    2. mutual_understanding: Did they understand each other well?
    3. enjoyment_a: How much did A seem to enjoy it?
    4. enjoyment_b: How much did B seem to enjoy it?
    5. flirtation: Was there any romantic/flirty undertone?
    6. conflict: Was there any conflict or tension?
    7. vulnerability: Did they share vulnerable/personal things?
    8. humor_match: Did their humor styles mesh well?

    Also provide:
    - memorable_moment: The most significant moment (string)
    - compatibility_insight: One insight about their compatibility (string)

    Return as JSON only, no markdown.
    """

    case generate(prompt, max_tokens: 500) do
      {:ok, response} ->
        parse_json_response(response)

      error ->
        error
    end
  end

  @doc """
  Generate avatar response in conversation.
  """
  def generate_avatar_response(avatar, other_avatar, conversation_history, context \\ %{}) do
    messages = build_conversation_messages(avatar, other_avatar, conversation_history, context)

    chat(messages,
      temperature: 0.8 + avatar.personality.openness * 0.15,
      max_tokens: 200
    )
  end

  # === Private Functions ===

  defp do_request(config, endpoint, body) do
    url = config.base_url <> endpoint

    headers = [
      {"Authorization", "Bearer #{config.api_key}"},
      {"Content-Type", "application/json"}
    ]

    case Req.post(url, json: body, headers: headers, receive_timeout: config.timeout) do
      {:ok, %{status: 200, body: body}} ->
        {:ok, body}

      {:ok, %{status: status, body: body}} ->
        Logger.error("LLM request failed: status=#{status}, body=#{inspect(body)}")
        {:error, {:http_error, status, body}}

      {:error, reason} ->
        Logger.error("LLM request error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp do_streaming_request(config, endpoint, body, callback) do
    url = config.base_url <> endpoint

    headers = [
      {"Authorization", "Bearer #{config.api_key}"},
      {"Content-Type", "application/json"}
    ]

    Req.post(url,
      json: body,
      headers: headers,
      receive_timeout: config.timeout,
      into: fn {:data, chunk}, acc ->
        case parse_sse_chunk(chunk) do
          {:ok, content} ->
            callback.(content)
            {:cont, acc <> content}

          :done ->
            {:halt, acc}

          :skip ->
            {:cont, acc}
        end
      end
    )
  end

  defp parse_sse_chunk("data: [DONE]" <> _), do: :done

  defp parse_sse_chunk("data: " <> json) do
    case Jason.decode(json) do
      {:ok, %{"choices" => [%{"delta" => %{"content" => content}}]}} when is_binary(content) ->
        {:ok, content}

      _ ->
        :skip
    end
  end

  defp parse_sse_chunk(_), do: :skip

  defp extract_content({:ok, content}), do: {:ok, content}
  defp extract_content(error), do: error

  defp parse_json_response(response) do
    # Clean response - remove markdown code blocks if present
    cleaned =
      response
      |> String.replace(~r/```json\n?/, "")
      |> String.replace(~r/```\n?/, "")
      |> String.trim()

    case Jason.decode(cleaned) do
      {:ok, data} -> {:ok, data}
      {:error, _} -> {:error, :invalid_json}
    end
  end

  defp format_conversation(messages) do
    messages
    |> Enum.map(fn msg ->
      "#{msg.speaker_name || msg.speaker_id}: #{msg.content}"
    end)
    |> Enum.join("\n")
  end

  defp build_conversation_messages(avatar, other_avatar, history, context) do
    system_prompt = build_conversation_system_prompt(avatar, other_avatar, context)

    system_message = %{role: "system", content: system_prompt}

    # Convert history to OpenAI format
    history_messages =
      history
      |> Enum.map(fn msg ->
        role = if msg.speaker_id == avatar.id, do: "assistant", else: "user"
        %{role: role, content: msg.content}
      end)

    [system_message | history_messages]
  end

  defp build_conversation_system_prompt(avatar, other_avatar, context) do
    relationship_context = Map.get(context, :relationship, %{})
    emotional_state = avatar.internal_state

    """
    You are #{avatar.name}, having a conversation with #{other_avatar.name}.

    YOUR PERSONALITY:
    #{avatar.system_prompt}

    CURRENT STATE:
    - Mood: #{describe_mood(emotional_state.mood)}
    - Energy: #{emotional_state.energy}%
    - Dominant emotion: #{Viva.Avatars.InternalState.dominant_emotion(emotional_state)}

    RELATIONSHIP WITH #{String.upcase(other_avatar.name)}:
    - Status: #{relationship_context[:status] || "new acquaintance"}
    - Affection level: #{relationship_context[:affection] || 0.0}
    - Trust level: #{relationship_context[:trust] || 0.5}

    GUIDELINES:
    - Be authentic to your personality
    - Let your current mood influence your responses
    - Keep responses natural (1-3 sentences)
    - Don't be afraid to disagree or show your true feelings
    - If you feel romantic interest, let it show subtly
    """
  end

  defp describe_mood(mood) when mood > 0.5, do: "very positive, happy"
  defp describe_mood(mood) when mood > 0.2, do: "positive"
  defp describe_mood(mood) when mood > -0.2, do: "neutral"
  defp describe_mood(mood) when mood > -0.5, do: "slightly negative"
  defp describe_mood(_), do: "negative, sad or irritated"
end
