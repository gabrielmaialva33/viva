defmodule Viva.AI.LLM.ImageClient do
  @moduledoc """
  Image generation and editing client for avatar visuals.

  Uses:
  - `stabilityai/stable-diffusion-3.5-large` - Generate profile pictures
  - `black-forest-labs/FLUX.1-Kontext-dev` - Edit expressions/variations

  ## Features

  - Generate unique profile pictures based on personality
  - Create expression variations (happy, sad, excited)
  - Edit existing images in-context
  - Generate avatar in different styles
  """
  require Logger

  alias Viva.Avatars.Avatar

  # === Types ===

  @type expression :: :happy | :sad | :angry | :surprised | :loving | :neutral
  @type style :: String.t()
  @type image_result :: {:ok, binary()} | {:ok, {:url, String.t()}} | {:error, term()}
  @type expression_pack :: %{optional(expression()) => binary()}

  @doc """
  Generate a profile picture for an avatar.

  ## Options

  - `:style` - Art style: "realistic", "anime", "illustration" (default: "realistic")
  - `:size` - Image size: "512x512", "1024x1024" (default: "1024x1024")
  - `:seed` - Random seed for reproducibility
  """
  @spec generate_profile(Avatar.t(), keyword()) :: image_result()
  def generate_profile(avatar, opts \\ []) do
    style = Keyword.get(opts, :style, "realistic")

    prompt = build_profile_prompt(avatar, style)
    negative_prompt = build_negative_prompt()

    # NVIDIA Image API uses a different endpoint and format
    # Endpoint: ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium
    # Note: NVIDIA NIM API does NOT support 'sampler' parameter
    body = %{
      prompt: prompt,
      negative_prompt: negative_prompt,
      cfg_scale: 7.5,
      seed: Keyword.get(opts, :seed, :rand.uniform(999_999_999)),
      steps: 30,
      aspect_ratio: "1:1"
    }

    case Viva.AI.LLM.image_request("stabilityai/stable-diffusion-3-medium", body, timeout: 120_000) do
      {:ok, %{"artifacts" => [%{"base64" => image_data} | _]}} ->
        {:ok, Base.decode64!(image_data)}

      {:ok, %{"image" => image_data}} when is_binary(image_data) ->
        {:ok, Base.decode64!(image_data)}

      {:ok, response} ->
        Logger.warning("Unexpected image response format: #{inspect(Map.keys(response))}")
        {:error, :unexpected_response_format}

      {:error, reason} ->
        Logger.error("Image generation error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Generate an expression variation of an avatar.

  ## Expressions

  - `:happy` - Smiling, joyful
  - `:sad` - Melancholic, tearful
  - `:angry` - Frustrated, intense
  - `:surprised` - Shocked, wide-eyed
  - `:loving` - Affectionate, warm
  - `:neutral` - Default expression
  """
  @spec generate_expression(Avatar.t(), expression(), keyword()) :: image_result()
  def generate_expression(avatar, expression, opts \\ []) do
    prompt = build_expression_prompt(avatar, expression)

    # Note: NVIDIA NIM API does NOT support 'sampler' parameter
    body = %{
      prompt: prompt,
      negative_prompt: build_negative_prompt(),
      cfg_scale: 7.0,
      seed: Keyword.get(opts, :seed, :rand.uniform(999_999_999)),
      steps: 25,
      aspect_ratio: "1:1"
    }

    case Viva.AI.LLM.image_request("stabilityai/stable-diffusion-3-medium", body, timeout: 90_000) do
      {:ok, %{"artifacts" => [%{"base64" => image_data} | _]}} ->
        {:ok, Base.decode64!(image_data)}

      {:ok, %{"image" => image_data}} when is_binary(image_data) ->
        {:ok, Base.decode64!(image_data)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Edit an existing image with a new prompt (in-context editing).
  Uses FLUX.1-Kontext for intelligent image editing.
  """
  @spec edit_image(binary(), String.t(), keyword()) :: image_result()
  def edit_image(image_data, edit_prompt, opts \\ []) do
    # FLUX.1-Kontext for image editing
    body = %{
      image: Base.encode64(image_data),
      prompt: edit_prompt,
      strength: Keyword.get(opts, :strength, 0.7),
      cfg_scale: 7.0,
      steps: 25
    }

    case Viva.AI.LLM.image_request("black-forest-labs/flux-1-kontext-dev", body, timeout: 90_000) do
      {:ok, %{"artifacts" => [%{"base64" => edited_data} | _]}} ->
        {:ok, Base.decode64!(edited_data)}

      {:ok, %{"image" => edited_data}} when is_binary(edited_data) ->
        {:ok, Base.decode64!(edited_data)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Change avatar expression in existing image.
  """
  @spec change_expression(binary(), expression(), keyword()) :: image_result()
  def change_expression(image_data, new_expression, opts \\ []) do
    edit_prompt =
      case new_expression do
        :happy -> "Change facial expression to happy, smiling warmly"
        :sad -> "Change facial expression to sad, slightly tearful"
        :angry -> "Change facial expression to angry, furrowed brow"
        :surprised -> "Change facial expression to surprised, wide eyes"
        :loving -> "Change facial expression to loving, soft gaze"
        :neutral -> "Change facial expression to neutral, calm"
        _ -> "Keep the same expression"
      end

    edit_image(image_data, edit_prompt, opts)
  end

  @doc """
  Generate avatar in a specific art style.
  """
  @spec stylize(Avatar.t(), style(), keyword()) :: image_result()
  def stylize(avatar, style, opts \\ []) do
    generate_profile(avatar, Keyword.put(opts, :style, style))
  end

  @doc """
  Generate multiple expression variations at once.
  Returns a map of expression => image_data.
  """
  @spec generate_expression_pack(Avatar.t(), [expression()] | nil, keyword()) ::
          {:ok, expression_pack()}
  def generate_expression_pack(avatar, expressions \\ nil, opts \\ []) do
    expressions = expressions || [:happy, :sad, :neutral, :surprised, :loving]

    results =
      expressions
      |> Task.async_stream(
        fn expr ->
          case generate_expression(avatar, expr, opts) do
            {:ok, data} -> {expr, data}
            {:error, _} -> {expr, nil}
          end
        end,
        max_concurrency: 3,
        timeout: 120_000
      )
      |> Enum.map(fn {:ok, result} -> result end)
      |> Enum.reject(fn {_, data} -> is_nil(data) end)
      |> Map.new()

    {:ok, results}
  end

  defp build_profile_prompt(avatar, style) do
    gender_text = gender_description(avatar.gender)
    age_text = age_description(avatar.age)
    personality_text = personality_visual_traits(avatar.personality)

    style_prefix =
      case style do
        "anime" -> "anime style portrait, "
        "illustration" -> "digital illustration portrait, "
        "realistic" -> "professional portrait photo, "
        _ -> "portrait, "
      end

    """
    #{style_prefix}#{gender_text}, #{age_text}, #{personality_text},
    Brazilian #{avatar.name}, warm lighting, high quality,
    looking at camera, professional headshot, detailed face
    """
    |> String.replace("\n", " ")
    |> String.trim()
  end

  defp build_expression_prompt(avatar, expression) do
    gender_text = gender_description(avatar.gender)
    expression_text = expression_description(expression)

    String.replace(
      """
      portrait of #{gender_text}, #{expression_text},
      #{avatar.name}, high quality face, detailed expression
      """,
      "\n",
      " "
    )
  end

  defp build_negative_prompt do
    String.replace(
      """
      deformed, ugly, bad anatomy, bad hands, missing fingers,
      extra fingers, blurry, low quality, watermark, text,
      signature, out of frame, cropped, worst quality
      """,
      "\n",
      " "
    )
  end

  defp gender_description(:male), do: "handsome man"
  defp gender_description(:female), do: "beautiful woman"
  defp gender_description(:non_binary), do: "attractive androgynous person"
  defp gender_description(_), do: "attractive person"

  defp age_description(age) when age < 25, do: "young adult, early twenties"
  defp age_description(age) when age < 35, do: "adult in their late twenties to early thirties"

  defp age_description(age) when age < 45,
    do: "mature adult in their late thirties to early forties"

  defp age_description(_), do: "distinguished mature adult"

  defp personality_visual_traits(personality) do
    base_traits =
      if personality.neuroticism > 0.6,
        do: ["sensitive expression", "deep eyes"],
        else: ["calm demeanor"]

    optional_traits =
      [
        {personality.extraversion > 0.7, ["confident posture", "bright eyes"]},
        {personality.openness > 0.7, ["creative appearance", "artistic vibe"]},
        {personality.agreeableness > 0.7, ["warm smile", "friendly face"]}
      ]
      |> Enum.filter(&elem(&1, 0))
      |> Enum.flat_map(&elem(&1, 1))

    Enum.join(base_traits ++ optional_traits, ", ")
  end

  defp expression_description(:happy), do: "happy expression, genuine smile, joyful eyes"
  defp expression_description(:sad), do: "sad expression, melancholic, slightly tearful"
  defp expression_description(:angry), do: "angry expression, furrowed brow, intense gaze"
  defp expression_description(:surprised), do: "surprised expression, wide eyes, raised eyebrows"
  defp expression_description(:loving), do: "loving expression, soft gaze, gentle smile"
  defp expression_description(:neutral), do: "neutral expression, calm, serene"
  defp expression_description(_), do: "natural expression"
end
