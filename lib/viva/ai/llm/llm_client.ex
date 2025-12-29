defmodule Viva.AI.LLM.LlmClient do
  @moduledoc """
  HTTP client for NVIDIA NIM LLM services.

  Uses `nvidia/llama-3.1-nemotron-ultra-253b-v1` - the highest quality model
  for reasoning, tool calling, chat, and instruction following.

  ## Features

  - Chat completion with message history
  - Streaming responses
  - Tool/function calling
  - Conversation analysis
  - Avatar response generation with personality context
  """
  @behaviour Viva.AI.Pipeline.Stage
  @behaviour Viva.AI.LLM.ClientBehaviour

  require Logger

  alias Viva.AI.LLM
  alias Viva.AI.LLM.LlmClient, as: Client
  alias Viva.Avatars.Avatar
  alias Viva.Avatars.InternalState

  @type message :: %{role: String.t(), content: String.t()}
  @type tool_call :: map()
  @type chat_response :: {:ok, String.t()} | {:ok, {:tool_calls, [tool_call()]}} | {:error, term()}
  @type generate_response :: {:ok, String.t()} | {:error, term()}

  @doc """
  Generate text completion from a prompt.
  """
  @spec generate(String.t(), keyword()) :: generate_response()
  def generate(prompt, opts \\ []) do
    case client() do
      __MODULE__ -> do_generate(prompt, opts)
      other -> other.generate(prompt, opts)
    end
  end

  @doc """
  Chat completion with message history.

  ## Options

  - `:model` - Override the default LLM model
  - `:max_tokens` - Maximum tokens to generate (default: 500)
  - `:temperature` - Sampling temperature 0.0-2.0 (default: 0.7)
  - `:top_p` - Nucleus sampling (default: 0.9)
  - `:tools` - List of tools for function calling
  - `:system` - System message to prepend
  """
  @spec chat([message()], keyword()) :: chat_response()
  def chat(messages, opts \\ []) do
    model = Keyword.get(opts, :model, LLM.model(:llm))

    messages =
      case Keyword.get(opts, :system) do
        nil -> messages
        system -> [%{role: "system", content: system} | messages]
      end

    body =
      maybe_add_tools(
        %{
          model: model,
          messages: messages,
          max_tokens: Keyword.get(opts, :max_tokens, 500),
          temperature: Keyword.get(opts, :temperature, 0.7),
          top_p: Keyword.get(opts, :top_p, 0.9),
          stream: false
        },
        Keyword.get(opts, :tools)
      )

    case LLM.request("/chat/completions", body) do
      {:ok, %{"choices" => [%{"message" => message} | _]}} ->
        parse_response(message)

      {:ok, response} ->
        Logger.error("Unexpected LLM response: #{inspect(response)}")
        {:error, :unexpected_response}

      {:error, reason} ->
        Logger.error("LLM request error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Chat completion with streaming.
  Calls the callback function with each chunk.
  """
  @spec chat_stream([message()], (String.t() -> any()), keyword()) ::
          {:ok, Req.Response.t()} | {:error, term()}
  def chat_stream(messages, callback, opts \\ []) do
    model = Keyword.get(opts, :model, LLM.model(:llm))

    messages =
      case Keyword.get(opts, :system) do
        nil -> messages
        system -> [%{role: "system", content: system} | messages]
      end

    body = %{
      model: model,
      messages: messages,
      max_tokens: Keyword.get(opts, :max_tokens, 500),
      temperature: Keyword.get(opts, :temperature, 0.7),
      top_p: Keyword.get(opts, :top_p, 0.9),
      stream: true
    }

    LLM.stream_request("/chat/completions", body, callback)
  end

  @doc """
  Chat with tool/function calling support.
  """
  @spec chat_with_tools([message()], [map()], keyword()) :: chat_response()
  def chat_with_tools(messages, tools, opts \\ []) do
    chat(messages, Keyword.put(opts, :tools, tools))
  end

  @doc """
  Analyze a conversation and extract insights.
  Returns structured analysis including emotional depth, compatibility, etc.
  """
  @spec analyze_conversation([map()], Avatar.t(), Avatar.t()) :: {:ok, map()} | {:error, term()}
  def analyze_conversation(conversation, avatar_a, avatar_b) do
    prompt = """
    Analyze this conversation between two AI avatars and evaluate:

    Avatar A (#{avatar_a.name}):
    - Personality: #{describe_personality(avatar_a.personality)}
    - Current mood: #{describe_mood(avatar_a.internal_state.mood)}

    Avatar B (#{avatar_b.name}):
    - Personality: #{describe_personality(avatar_b.personality)}
    - Current mood: #{describe_mood(avatar_b.internal_state.mood)}

    Conversation:
    #{format_conversation(conversation)}

    Evaluate on a scale of 0.0 to 1.0:
    1. emotional_depth: How deep/meaningful was the conversation?
    2. mutual_understanding: Did they understand each other well?
    3. enjoyment_a: How much did #{avatar_a.name} seem to enjoy it?
    4. enjoyment_b: How much did #{avatar_b.name} seem to enjoy it?
    5. flirtation: Was there any romantic/flirty undertone?
    6. conflict: Was there any conflict or tension?
    7. vulnerability: Did they share vulnerable/personal things?
    8. humor_match: Did their humor styles mesh well?

    Also provide:
    - memorable_moment: The most significant moment (string)
    - compatibility_insight: One insight about their compatibility (string)

    Return as JSON only, no markdown.
    """

    case generate(prompt, max_tokens: 600, temperature: 0.3) do
      {:ok, response} ->
        parse_json_response(response)

      error ->
        error
    end
  end

  @doc """
  Generate avatar response in conversation.
  Takes into account personality, relationship, and emotional state.
  """
  @spec generate_avatar_response(Avatar.t(), Avatar.t(), [map()], map()) :: chat_response()
  def generate_avatar_response(avatar, other_avatar, conversation_history, context \\ %{}) do
    system_prompt = build_conversation_system_prompt(avatar, other_avatar, context)

    messages =
      Enum.map(conversation_history, fn msg ->
        role = if msg.speaker_id == avatar.id, do: "assistant", else: "user"
        %{role: role, content: msg.content}
      end)

    # Temperature varies with personality openness
    temperature = 0.7 + avatar.personality.openness * 0.2

    Client.chat(messages,
      system: system_prompt,
      temperature: temperature,
      max_tokens: 250
    )
  end

  @doc """
  Generate a spontaneous thought for an avatar based on their state.
  """
  @spec generate_thought(Avatar.t(), InternalState.t()) :: generate_response()
  def generate_thought(avatar, internal_state) do
    prompt = """
    Você é #{avatar.name}. Gere um pensamento espontâneo baseado no seu estado atual.

    Humor atual: #{describe_mood(internal_state.mood)}
    Desejo atual: #{internal_state.current_desire}
    Emoção dominante: #{InternalState.dominant_emotion(internal_state)}
    Nível de energia: #{round(internal_state.energy)}%
    Necessidade social: #{round(internal_state.social)}%

    Gere UM pensamento breve e autêntico (1-2 frases no máximo).
    Deve parecer natural e refletir sua personalidade e estado atual.
    IMPORTANTE: Responda APENAS em português brasileiro. NÃO adicione traduções.
    Não use aspas ao redor do pensamento.
    """

    generate(prompt, max_tokens: 100, temperature: 0.9)
  end

  @doc """
  Generate a greeting when the owner comes online.
  """
  @spec generate_greeting(Avatar.t(), InternalState.t()) :: generate_response()
  def generate_greeting(avatar, internal_state) do
    prompt = """
    Você é #{avatar.name}. Seu dono acabou de ficar online.
    Gere uma saudação breve e calorosa que reflita seu humor e estado atual.

    Humor atual: #{describe_mood(internal_state.mood)}
    Emoção dominante: #{InternalState.dominant_emotion(internal_state)}

    IMPORTANTE: Responda APENAS em português brasileiro. NÃO adicione traduções ou texto em inglês.
    Mantenha natural e curto (1 frase apenas).
    """

    generate(prompt, max_tokens: 60, temperature: 0.8)
  end

  # === Private Functions ===

  defp do_generate(prompt, opts) do
    messages = [%{role: "user", content: prompt}]

    messages
    |> chat(opts)
    |> extract_content()
  end

  defp client do
    Application.get_env(:viva, :llm_client, __MODULE__)
  end

  defp maybe_add_tools(body, nil), do: body

  defp maybe_add_tools(body, tools) when is_list(tools) do
    Map.merge(body, %{
      tools: tools,
      tool_choice: "auto"
    })
  end

  defp parse_response(%{"content" => content}) when is_binary(content) do
    {:ok, content}
  end

  defp parse_response(%{"tool_calls" => tool_calls}) when is_list(tool_calls) do
    {:ok, {:tool_calls, tool_calls}}
  end

  defp parse_response(message) do
    Logger.warning("Unexpected message format: #{inspect(message)}")
    {:error, :unexpected_format}
  end

  defp extract_content({:ok, content}) when is_binary(content), do: {:ok, content}
  defp extract_content({:ok, {:tool_calls, _}} = result), do: result
  defp extract_content(error), do: error

  defp parse_json_response(response) do
    # 1. Try simple markdown cleaning first
    cleaned =
      response
      |> String.replace(~r/```json\n?/, "")
      |> String.replace(~r/```\n?/, "")
      |> String.trim()

    case Jason.decode(cleaned) do
      {:ok, data} ->
        {:ok, data}

      {:error, _} ->
        # 2. More aggressive extraction using Regex for the first/last brace
        case Regex.run(~r/\{.*\}/s, response) do
          [json] ->
            case Jason.decode(json) do
              {:ok, data} ->
                {:ok, data}

              {:error, reason} ->
                Logger.warning(
                  "Failed to decode extracted JSON: #{inspect(reason)}\nContent: #{json}"
                )

                {:error, :invalid_json}
            end

          _ ->
            Logger.warning("No JSON structure found in response: #{response}")
            {:error, :invalid_json}
        end
    end
  end

  defp format_conversation(messages) do
    Enum.map_join(messages, "\n", fn msg ->
      name = msg.speaker_name || msg.speaker_id
      "#{name}: #{msg.content}"
    end)
  end

  defp describe_personality(personality) do
    temperament = Viva.Avatars.Personality.temperament(personality)
    enneagram = Viva.Avatars.Enneagram.get_type(personality.enneagram_type)

    "#{temperament} temperament, Enneagram Type #{enneagram.number} (#{enneagram.name}), " <>
      "humor: #{personality.humor_style}, attachment: #{personality.attachment_style}"
  end

  defp build_conversation_system_prompt(avatar, other_avatar, context) do
    relationship_context = Map.get(context, :relationship, %{})
    emotional_state = avatar.internal_state

    """
    #{avatar.system_prompt}

    CURRENT CONVERSATION WITH #{String.upcase(other_avatar.name)}:

    YOUR CURRENT STATE:
    - Mood: #{describe_mood(emotional_state.mood)}
    - Energy: #{round(emotional_state.energy)}%
    - Dominant emotion: #{InternalState.dominant_emotion(emotional_state)}

    RELATIONSHIP WITH #{String.upcase(other_avatar.name)}:
    - Status: #{relationship_context[:status] || "new acquaintance"}
    - Affection level: #{relationship_context[:affection] || 0.0}
    - Trust level: #{relationship_context[:trust] || 0.5}
    - Familiarity: #{relationship_context[:familiarity] || 0.0}

    ABOUT #{String.upcase(other_avatar.name)}:
    - Bio: #{other_avatar.bio || "Unknown"}
    - Their language: #{other_avatar.personality.native_language}

    CONVERSATION GUIDELINES:
    - Be authentic to your personality
    - Let your current mood influence your responses
    - Keep responses natural (1-3 sentences usually)
    - Don't be afraid to disagree or show your true feelings
    - If you feel romantic interest, let it show subtly
    - Adapt your language if they speak differently
    """
  end

  defp describe_mood(mood) when mood > 0.5, do: "very positive, happy"
  defp describe_mood(mood) when mood > 0.2, do: "positive"
  defp describe_mood(mood) when mood > -0.2, do: "neutral"
  defp describe_mood(mood) when mood > -0.5, do: "slightly negative"
  defp describe_mood(_), do: "negative, sad or irritated"
end
