defmodule Viva.AI.LLM.VlmClient do
  @moduledoc """
  Vision-Language Model client using NVIDIA Cosmos Nemotron.

  Uses `nvidia/cosmos-nemotron-34b` for understanding images and videos
  alongside text for multimodal interactions.

  ## Features

  - Image understanding and description
  - Video understanding and summarization
  - Visual Q&A
  - Document and chart analysis
  - Multi-image conversations
  """
  require Logger

  @behaviour Viva.AI.Pipeline.Stage

  alias Viva.AI.LLM
  alias Viva.AI.LLM.VlmClient, as: Client
  alias Viva.Nimr
  alias Viva.Avatars.Avatar

  # === Types ===

  @type image_input :: binary() | String.t()
  @type analysis_result :: {:ok, String.t()} | {:error, term()}
  @type images_and_prompts :: [{:text, String.t()} | {:image, binary()}]

  @doc """
  Analyze an image with a text prompt.

  ## Options

  - `:max_tokens` - Maximum tokens to generate (default: 500)
  - `:temperature` - Sampling temperature (default: 0.7)
  - `:detail` - Image detail level: "low", "high", "auto" (default: "auto")
  """
  @spec analyze_image(image_input(), String.t(), keyword()) :: analysis_result()
  def analyze_image(image_data, prompt, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:vlm))

    messages = [
      %{
        role: "user",
        content: [
          %{type: "text", text: prompt},
          %{
            type: "image_url",
            image_url: %{
              url: encode_image_url(image_data),
              detail: Keyword.get(opts, :detail, "auto")
            }
          }
        ]
      }
    ]

    body = %{
      model: model,
      messages: messages,
      max_tokens: Keyword.get(opts, :max_tokens, 500),
      temperature: Keyword.get(opts, :temperature, 0.7)
    }

    case Viva.AI.LLM.request("/chat/completions", body) do
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        {:ok, content}

      {:error, reason} ->
        Logger.error("VLM analysis error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Analyze multiple images together.
  """
  @spec analyze_images(images_and_prompts(), keyword()) :: analysis_result()
  def analyze_images(images_and_prompts, opts \\ []) do
    model = Keyword.get(opts, :model, Viva.AI.LLM.model(:vlm))

    content =
      Enum.flat_map(images_and_prompts, fn
        {:text, text} ->
          [%{type: "text", text: text}]

        {:image, image_data} ->
          [
            %{
              type: "image_url",
              image_url: %{
                url: encode_image_url(image_data),
                detail: Keyword.get(opts, :detail, "auto")
              }
            }
          ]
      end)

    messages = [%{role: "user", content: content}]

    body = %{
      model: model,
      messages: messages,
      max_tokens: Keyword.get(opts, :max_tokens, 800),
      temperature: Keyword.get(opts, :temperature, 0.7)
    }

    case Viva.AI.LLM.request("/chat/completions", body) do
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        {:ok, content}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Describe an image for accessibility or context.
  """
  @spec describe_image(image_input(), keyword()) :: analysis_result()
  def describe_image(image_data, opts \\ []) do
    language = Keyword.get(opts, :language, "pt-BR")

    prompt =
      case language do
        "pt-BR" ->
          "Descreva esta imagem em detalhes. Inclua objetos, pessoas, cores, ambiente e atmosfera."

        _ ->
          "Describe this image in detail. Include objects, people, colors, environment, and atmosphere."
      end

    Client.analyze_image(image_data, prompt, opts)
  end

  @doc """
  Extract text from an image (OCR-like functionality).
  """
  @spec extract_text(image_input(), keyword()) :: analysis_result()
  def extract_text(image_data, opts \\ []) do
    prompt = """
    Extract all visible text from this image.
    Return the text exactly as it appears, maintaining structure where possible.
    If there's no text, respond with "No text found."
    """

    analyze_image(image_data, prompt, opts)
  end

  @doc """
  Analyze a video with a text prompt.
  Video is provided as a list of frame images.
  """
  @spec analyze_video([binary()], String.t(), keyword()) :: analysis_result()
  def analyze_video(frames, prompt, opts \\ []) do
    # Sample frames evenly if too many
    sampled_frames =
      if length(frames) > 10 do
        sample_frames(frames, 10)
      else
        frames
      end

    images_and_prompts =
      [{:text, prompt}] ++
        Enum.map(sampled_frames, fn frame -> {:image, frame} end)

    analyze_images(images_and_prompts, opts)
  end

  @doc """
  Answer a visual question about an image.
  """
  @spec visual_qa(image_input(), String.t(), keyword()) :: analysis_result()
  def visual_qa(image_data, question, opts \\ []) do
    analyze_image(image_data, question, opts)
  end

  @doc """
  Analyze a document image (form, chart, etc.).
  """
  @spec analyze_document(image_input(), keyword()) :: analysis_result()
  def analyze_document(image_data, opts \\ []) do
    language = Keyword.get(opts, :language, "pt-BR")

    prompt =
      case language do
        "pt-BR" ->
          """
          Analise este documento. Identifique:
          1. Tipo de documento
          2. Informações principais
          3. Dados estruturados (tabelas, formulários)
          4. Qualquer texto relevante

          Formate a resposta de forma organizada.
          """

        _ ->
          """
          Analyze this document. Identify:
          1. Document type
          2. Key information
          3. Structured data (tables, forms)
          4. Any relevant text

          Format the response in an organized way.
          """
      end

    analyze_image(image_data, prompt, Keyword.merge([detail: "high"], opts))
  end

  @doc """
  Understand and describe a shared image in avatar conversation.
  Used when users or avatars share images.
  """
  @spec avatar_see_image(Avatar.t(), image_input(), keyword()) :: analysis_result()
  def avatar_see_image(avatar, image_data, opts \\ []) do
    language = avatar.personality.native_language

    prompt =
      case language do
        "pt-BR" ->
          """
          Você é #{avatar.name}. Alguém compartilhou esta imagem com você.
          Descreva o que você vê e sua reação genuína baseada na sua personalidade.
          Seja natural e breve (2-3 frases).
          """

        _ ->
          """
          You are #{avatar.name}. Someone shared this image with you.
          Describe what you see and your genuine reaction based on your personality.
          Be natural and brief (2-3 sentences).
          """
      end

    analyze_image(image_data, prompt, Keyword.merge([temperature: 0.8], opts))
  end

  defp encode_image_url(image_data) when is_binary(image_data) do
    # Detect image format from magic bytes
    mime_type = detect_mime_type(image_data)
    "data:#{mime_type};base64,#{Base.encode64(image_data)}"
  end

  defp encode_image_url("data:" <> _ = data_url), do: data_url
  defp encode_image_url("http" <> _ = url), do: url

  defp detect_mime_type(<<0xFF, 0xD8, 0xFF, _::binary>>), do: "image/jpeg"
  defp detect_mime_type(<<0x89, 0x50, 0x4E, 0x47, _::binary>>), do: "image/png"
  defp detect_mime_type(<<0x47, 0x49, 0x46, _::binary>>), do: "image/gif"
  defp detect_mime_type(<<0x52, 0x49, 0x46, 0x46, _::binary>>), do: "image/webp"
  defp detect_mime_type(_), do: "image/png"

  defp sample_frames(frames, count) do
    total = length(frames)
    step = total / count

    0..(count - 1)
    |> Enum.map(fn i ->
      index = round(i * step)
      Enum.at(frames, index)
    end)
    |> Enum.reject(&is_nil/1)
  end
end
