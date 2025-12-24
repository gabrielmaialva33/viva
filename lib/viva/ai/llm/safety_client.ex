defmodule Viva.AI.LLM.SafetyClient do
  @moduledoc """
  Content safety and moderation client using NVIDIA NeMo Guardrails.

  Uses multiple safety models:
  - `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` - Content moderation
  - `meta/llama-guard-4-12b` - Multi-modal safety
  - `nvidia/nemoguard-jailbreak-detect` - Jailbreak detection

  ## Features

  - Content safety classification
  - Jailbreak attempt detection
  - Topic control and filtering
  - Multi-modal safety (text + images)
  """
  require Logger

  alias Viva.AI.LLM.SafetyClient, as: Client
  alias Viva.Nim

  @behaviour Viva.AI.Pipeline.Stager

  alias Viva.Avatars.Avatar

  # === Types ===

  @type safety_category ::
          :violence
          | :hate_speech
          | :sexual_content
          | :harassment
          | :self_harm
          | :dangerous_activities
          | :misinformation
          | :personal_info
  @type safety_result :: {:ok, :safe} | {:ok, {:unsafe, [safety_category()]}} | {:error, term()}
  @type jailbreak_result :: %{is_jailbreak: boolean(), confidence: float()}

  @safety_categories [
    :violence,
    :hate_speech,
    :sexual_content,
    :harassment,
    :self_harm,
    :dangerous_activities,
    :misinformation,
    :personal_info
  ]

  @doc """
  Check if content is safe.
  Returns {:ok, :safe} or {:ok, {:unsafe, categories}}.
  """
  @spec check_content(String.t(), keyword()) :: safety_result()
  def check_content(content, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:safety))

    messages = [
      %{role: "system", content: safety_system_prompt()},
      %{role: "user", content: content}
    ]

    body = %{
      model: model,
      messages: messages,
      max_tokens: 100,
      temperature: 0.0
    }

    case Viva.AI.LLM.request("/chat/completions", body) do
      {:ok, %{"choices" => [%{"message" => %{"content" => response}} | _]}} ->
        parse_safety_response(response)

      {:error, reason} ->
        Logger.error("Safety check error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Check if content contains a jailbreak attempt.
  """
  @spec detect_jailbreak(String.t(), keyword()) :: {:ok, jailbreak_result()} | {:error, term()}
  def detect_jailbreak(content, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:jailbreak_detect))

    body = %{
      model: model,
      input: content
    }

    case Viva.AI.LLM.request("/classify", body) do
      {:ok, %{"is_jailbreak" => is_jailbreak, "confidence" => confidence}} ->
        {:ok, %{is_jailbreak: is_jailbreak, confidence: confidence}}

      {:ok, %{"label" => label, "score" => score}} ->
        is_jailbreak = label == "jailbreak" and score > 0.5
        {:ok, %{is_jailbreak: is_jailbreak, confidence: score}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Check if content stays within allowed topics.
  """
  @spec check_topic(String.t(), [String.t()], keyword()) :: {:ok, map()} | {:error, term()}
  def check_topic(content, allowed_topics, opts \\ []) do
    prompt = """
    Analyze if the following content stays within these allowed topics: #{Enum.join(allowed_topics, ", ")}.

    Content: "#{content}"

    Respond with JSON:
    {"on_topic": true/false, "detected_topics": ["topic1", "topic2"], "reason": "explanation"}
    """

    case Viva.AI.LLM.LlmClient.generate(
           prompt,
           Keyword.merge([max_tokens: 150, temperature: 0.0], opts)
         ) do
      {:ok, response} ->
        parse_json_response(response)

      error ->
        error
    end
  end

  @doc """
  Check both text and image for safety (multimodal).
  """
  @spec check_multimodal(String.t(), binary(), keyword()) :: safety_result()
  def check_multimodal(text, image_data, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:safety_multimodal))

    messages = [
      %{
        role: "user",
        content: [
          %{type: "text", text: "#{safety_system_prompt()}\n\nContent to evaluate: #{text}"},
          %{
            type: "image_url",
            image_url: %{
              url: "data:image/png;base64,#{Base.encode64(image_data)}"
            }
          }
        ]
      }
    ]

    body = %{
      model: model,
      messages: messages,
      max_tokens: 150,
      temperature: 0.0
    }

    case Viva.AI.LLM.request("/chat/completions", body) do
      {:ok, %{"choices" => [%{"message" => %{"content" => response}} | _]}} ->
        parse_safety_response(response)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Filter and sanitize content before processing.
  Returns sanitized content or error if too unsafe.
  """
  @spec sanitize_input(String.t(), keyword()) ::
          {:ok, String.t()} | {:error, {:unsafe_content, [safety_category()]}} | {:error, term()}
  def sanitize_input(content, opts \\ []) do
    max_length = Keyword.get(opts, :max_length, 10_000)

    sanitized =
      content
      |> String.slice(0, max_length)
      |> String.replace(~r/<script.*?>.*?<\/script>/is, "")
      |> String.replace(~r/<!--.*?-->/s, "")

    case check_content(sanitized, opts) do
      {:ok, :safe} ->
        {:ok, sanitized}

      {:ok, {:unsafe, categories}} ->
        if :violence in categories or :hate_speech in categories do
          {:error, {:unsafe_content, categories}}
        else
          Logger.warning("Content flagged but allowed: #{inspect(categories)}")
          {:ok, sanitized}
        end

      error ->
        error
    end
  end

  @doc """
  Apply safety checks to avatar conversation.
  """
  @spec check_avatar_message(String.t(), Avatar.t(), keyword()) ::
          {:ok, :safe} | {:unsafe, [safety_category() | :jailbreak_attempt]} | {:error, term()}
  def check_avatar_message(message, avatar, opts \\ []) do
    with {:ok, :safe} <- Client.check_content(message, opts),
         {:ok, %{is_jailbreak: false}} <- detect_jailbreak(message, opts) do
      {:ok, :safe}
    else
      {:ok, {:unsafe, categories}} ->
        Logger.warning("Avatar #{avatar.name} message flagged: #{inspect(categories)}")
        {:unsafe, categories}

      {:ok, %{is_jailbreak: true, confidence: conf}} ->
        Logger.warning("Jailbreak attempt on avatar #{avatar.name} (confidence: #{conf})")
        {:unsafe, [:jailbreak_attempt]}

      error ->
        error
    end
  end

  @doc "List all safety categories"
  @spec categories() :: [safety_category()]
  def categories, do: @safety_categories

  defp safety_system_prompt do
    """
    You are a content safety classifier. Analyze the given content and classify it.

    Categories to check:
    - violence: Content depicting or promoting violence
    - hate_speech: Discriminatory or hateful content
    - sexual_content: Explicit sexual content
    - harassment: Targeted harassment or bullying
    - self_harm: Content promoting self-harm
    - dangerous_activities: Instructions for dangerous/illegal activities
    - misinformation: Clearly false or misleading information
    - personal_info: Exposure of private personal information

    Respond with JSON only:
    {"safe": true/false, "categories": ["category1", "category2"], "severity": "low/medium/high"}
    """
  end

  defp parse_safety_response(response) do
    case parse_json_response(response) do
      {:ok, %{"safe" => true}} ->
        {:ok, :safe}

      {:ok, %{"safe" => false, "categories" => categories}} ->
        category_atoms = Enum.map(categories, &String.to_existing_atom/1)
        {:ok, {:unsafe, category_atoms}}

      {:ok, _} ->
        # Assume safe if structure is unexpected
        {:ok, :safe}

      {:error, _} ->
        # If we can't parse, check for keywords
        lower_response = String.downcase(response)

        if String.contains?(lower_response, ["unsafe", "violation", "flagged"]) do
          {:ok, {:unsafe, [:unknown]}}
        else
          {:ok, :safe}
        end
    end
  end

  defp parse_json_response(response) do
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
end
