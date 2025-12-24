defmodule Viva.AI.LLM.Avatar3DClient do
  @moduledoc """
  3D Avatar generation and animation client.

  Uses:
  - `microsoft/TRELLIS` - Generate 3D models from text/image
  - `nvidia/audio2face-3d` - Convert audio to facial blendshapes

  ## Features

  - Generate 3D avatar model from description
  - Convert 2D profile to 3D model
  - Real-time lipsync from audio
  - Facial expression animation
  """
  @behaviour Viva.AI.Pipeline.Stage

  require Logger

  alias Viva.AI.LLM, as: Client
  alias Viva.Avatars.Avatar
  alias Viva.Nim

  # === Types ===

  @type model_result :: {:ok, binary()} | {:ok, {:url, String.t()}} | {:error, term()}
  @type animation_data :: %{
          blendshapes: [map()],
          timestamps: [float()] | nil,
          fps: pos_integer(),
          duration: float() | non_neg_integer(),
          expression: atom() | nil
        }
  @type stream_state :: %{
          model: String.t(),
          fps: pos_integer(),
          callback: (term() -> any()),
          buffer: binary()
        }

  @doc """
  Generate a 3D avatar model from text description.

  ## Options

  - `:format` - Output format: "glb", "gltf", "fbx" (default: "glb")
  - `:quality` - Model quality: "low", "medium", "high" (default: "high")
  """
  @spec generate_from_text(String.t(), keyword()) :: model_result()
  def generate_from_text(description, opts \\ []) do
    model = Keyword.get(opts, :model, Client.model(:avatar_3d))
    format = Keyword.get(opts, :format, "glb")
    quality = Keyword.get(opts, :quality, "high")

    body = %{
      model: model,
      prompt: description,
      output_format: format,
      quality: quality,
      num_inference_steps: quality_steps(quality)
    }

    case Client.request("/3d/generate", body, timeout: 180_000) do
      {:ok, %{"model_data" => model_data}} ->
        {:ok, Base.decode64!(model_data)}

      {:ok, %{"url" => url}} ->
        {:ok, {:url, url}}

      {:error, reason} ->
        Logger.error("3D generation error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Generate a 3D avatar from a 2D profile image.
  """
  @spec generate_from_image(binary(), keyword()) :: model_result()
  def generate_from_image(image_data, opts \\ []) do
    model = Keyword.get(opts, :model, Client.model(:avatar_3d))
    format = Keyword.get(opts, :format, "glb")

    body = %{
      model: model,
      image: Base.encode64(image_data),
      output_format: format,
      quality: Keyword.get(opts, :quality, "high")
    }

    case Client.request("/3d/image-to-3d", body, timeout: 180_000) do
      {:ok, %{"model_data" => model_data}} ->
        {:ok, Base.decode64!(model_data)}

      {:ok, %{"url" => url}} ->
        {:ok, {:url, url}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Generate 3D avatar for a VIVA avatar entity.
  Uses personality and appearance to create appropriate model.
  """
  @spec generate_for_avatar(Avatar.t(), keyword()) :: model_result()
  def generate_for_avatar(avatar, opts \\ []) do
    description = build_avatar_description(avatar)
    generate_from_text(description, opts)
  end

  @doc """
  Convert audio to facial blendshapes for lipsync.
  Returns animation data that can be applied to a 3D model.

  ## Options

  - `:fps` - Frames per second (default: 30)
  - `:emotion` - Emotional overlay: "neutral", "happy", "sad" (default: "neutral")
  """
  @spec audio_to_face(binary(), keyword()) :: {:ok, animation_data()} | {:error, term()}
  def audio_to_face(audio_data, opts \\ []) do
    model = Keyword.get(opts, :model, Client.model(:audio2face))
    fps = Keyword.get(opts, :fps, 30)
    emotion = Keyword.get(opts, :emotion, "neutral")

    body = %{
      model: model,
      audio: Base.encode64(audio_data),
      fps: fps,
      emotion_overlay: emotion
    }

    case Client.request("/audio/face", body, timeout: 60_000) do
      {:ok, %{"blendshapes" => blendshapes, "timestamps" => timestamps}} ->
        {:ok,
         %{
           blendshapes: blendshapes,
           timestamps: timestamps,
           fps: fps,
           duration: List.last(timestamps) || 0
         }}

      {:error, reason} ->
        Logger.error("Audio2Face error: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Stream audio to facial animation in real-time.
  Calls callback with blendshape data as it's generated.
  """
  @spec audio_to_face_stream((term() -> any()), keyword()) :: {:ok, stream_state()}
  def audio_to_face_stream(callback, opts \\ []) do
    model = Keyword.get(opts, :model, Client.model(:audio2face))
    fps = Keyword.get(opts, :fps, 30)

    {:ok,
     %{
       model: model,
       fps: fps,
       callback: callback,
       buffer: <<>>
     }}
  end

  @doc """
  Send audio chunk to streaming face animation.
  """
  @spec stream_audio_chunk(stream_state(), binary()) :: {:ok, stream_state()} | {:error, term()}
  def stream_audio_chunk(stream_state, audio_chunk) do
    body = %{
      model: stream_state.model,
      audio: Base.encode64(audio_chunk),
      fps: stream_state.fps,
      stream: true
    }

    case Client.request("/audio/face", body) do
      {:ok, %{"blendshapes" => shapes, "is_final" => is_final}} ->
        stream_state.callback.({:blendshapes, shapes, is_final})
        {:ok, stream_state}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Generate expression-specific facial animation.
  """
  @spec generate_expression_animation(atom(), pos_integer(), keyword()) :: {:ok, animation_data()}
  def generate_expression_animation(expression, duration_ms \\ 1000, opts \\ []) do
    fps = Keyword.get(opts, :fps, 30)
    frames = round(duration_ms / 1000 * fps)

    blendshapes = expression_blendshapes(expression, frames)

    {:ok,
     %{
       blendshapes: blendshapes,
       fps: fps,
       duration: duration_ms,
       expression: expression
     }}
  end

  @doc """
  Get supported blendshape names (ARKit compatible).
  """
  @spec blendshape_names() :: [String.t()]
  def blendshape_names do
    [
      "eyeBlinkLeft",
      "eyeBlinkRight",
      "eyeSquintLeft",
      "eyeSquintRight",
      "eyeWideLeft",
      "eyeWideRight",
      "eyeLookDownLeft",
      "eyeLookDownRight",
      "eyeLookInLeft",
      "eyeLookInRight",
      "eyeLookOutLeft",
      "eyeLookOutRight",
      "eyeLookUpLeft",
      "eyeLookUpRight",
      "browDownLeft",
      "browDownRight",
      "browInnerUp",
      "browOuterUpLeft",
      "browOuterUpRight",
      "cheekPuff",
      "cheekSquintLeft",
      "cheekSquintRight",
      "jawForward",
      "jawLeft",
      "jawRight",
      "jawOpen",
      "mouthClose",
      "mouthFunnel",
      "mouthPucker",
      "mouthLeft",
      "mouthRight",
      "mouthSmileLeft",
      "mouthSmileRight",
      "mouthFrownLeft",
      "mouthFrownRight",
      "mouthDimpleLeft",
      "mouthDimpleRight",
      "mouthStretchLeft",
      "mouthStretchRight",
      "mouthRollLower",
      "mouthRollUpper",
      "mouthShrugLower",
      "mouthShrugUpper",
      "mouthPressLeft",
      "mouthPressRight",
      "mouthLowerDownLeft",
      "mouthLowerDownRight",
      "mouthUpperUpLeft",
      "mouthUpperUpRight",
      "noseSneerLeft",
      "noseSneerRight"
    ]
  end

  defp build_avatar_description(avatar) do
    gender = gender_text(avatar.gender)
    age = age_text(avatar.age)
    personality = personality_style(avatar.personality)

    String.replace(
      """
      3D character model of a #{gender}, #{age} years old, #{personality},
      Brazilian appearance, realistic proportions,
      suitable for real-time rendering, game-ready mesh
      """,
      "\n",
      " "
    )
  end

  defp gender_text(:male), do: "male character"
  defp gender_text(:female), do: "female character"
  defp gender_text(:non_binary), do: "androgynous character"
  defp gender_text(_), do: "character"

  defp age_text(age) when age < 25, do: "young adult around #{age}"
  defp age_text(age) when age < 40, do: "adult around #{age}"
  defp age_text(age), do: "mature adult around #{age}"

  defp personality_style(personality) do
    cond do
      personality.extraversion > 0.7 -> "confident and expressive"
      personality.openness > 0.7 -> "artistic and unique style"
      personality.agreeableness > 0.7 -> "warm and approachable"
      true -> "balanced and calm"
    end
  end

  defp quality_steps("low"), do: 20
  defp quality_steps("medium"), do: 35
  defp quality_steps("high"), do: 50
  defp quality_steps(_), do: 35

  defp expression_blendshapes(:happy, frames) do
    for i <- 0..(frames - 1) do
      progress = i / max(frames - 1, 1)
      intensity = ease_in_out(progress)

      %{
        "mouthSmileLeft" => 0.7 * intensity,
        "mouthSmileRight" => 0.7 * intensity,
        "cheekSquintLeft" => 0.4 * intensity,
        "cheekSquintRight" => 0.4 * intensity
      }
    end
  end

  defp expression_blendshapes(:sad, frames) do
    for i <- 0..(frames - 1) do
      progress = i / max(frames - 1, 1)
      intensity = ease_in_out(progress)

      %{
        "mouthFrownLeft" => 0.5 * intensity,
        "mouthFrownRight" => 0.5 * intensity,
        "browInnerUp" => 0.6 * intensity,
        "eyeSquintLeft" => 0.2 * intensity,
        "eyeSquintRight" => 0.2 * intensity
      }
    end
  end

  defp expression_blendshapes(:surprised, frames) do
    for i <- 0..(frames - 1) do
      progress = i / max(frames - 1, 1)
      intensity = ease_in_out(progress)

      %{
        "eyeWideLeft" => 0.8 * intensity,
        "eyeWideRight" => 0.8 * intensity,
        "browOuterUpLeft" => 0.7 * intensity,
        "browOuterUpRight" => 0.7 * intensity,
        "jawOpen" => 0.3 * intensity
      }
    end
  end

  defp expression_blendshapes(_, frames) do
    for _ <- 0..(frames - 1), do: %{}
  end

  defp ease_in_out(t) do
    if t < 0.5 do
      2 * t * t
    else
      1 - :math.pow(-2 * t + 2, 2) / 2
    end
  end
end
