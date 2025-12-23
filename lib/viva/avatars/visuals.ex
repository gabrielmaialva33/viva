defmodule Viva.Avatars.Visuals do
  @moduledoc """
  Orchestrates visual asset generation for avatars.

  Uses NIM models to generate:
  - Profile images (Stable Diffusion 3.5)
  - Expression variations (FLUX.1-Kontext)
  - 3D models (TRELLIS)
  - Facial animations (Audio2Face)

  ## Usage

      Visuals.generate_complete(avatar)

      Visuals.generate_profile(avatar)

      Visuals.update_expression(avatar, :happy)
  """
  require Logger

  alias Viva.Avatars.Avatar
  alias Viva.Nim.Avatar3DClient
  alias Viva.Nim.ImageClient
  alias Viva.Repo

  # === Types ===

  @type expression :: :neutral | :happy | :sad | :angry | :surprised | :loving
  @type emotion :: atom()

  @storage_base "priv/static/uploads/avatars"
  @public_base "/uploads/avatars"

  @doc """
  Generate complete visual package for an avatar.
  Includes profile image, expression pack, and optionally 3D model.

  ## Options

  - `:include_3d` - Generate 3D model (default: false, expensive)
  - `:style` - Art style: "realistic", "anime", "illustration"
  - `:expressions` - List of expressions to generate
  """
  @spec generate_complete(Avatar.t(), keyword()) :: {:ok, Avatar.t()} | {:error, term()}
  def generate_complete(avatar, opts \\ []) do
    include_3d = Keyword.get(opts, :include_3d, false)
    expressions = Keyword.get(opts, :expressions, [:happy, :sad, :neutral])

    with {:ok, avatar} <- generate_profile(avatar, opts),
         {:ok, avatar} <- generate_expressions(avatar, expressions, opts) do
      if include_3d do
        generate_3d_model(avatar, opts)
      else
        {:ok, avatar}
      end
    end
  end

  @doc """
  Generate profile image for an avatar.
  Saves to disk and updates avatar record.
  """
  @spec generate_profile(Avatar.t(), keyword()) :: {:ok, Avatar.t()} | {:error, term()}
  def generate_profile(avatar, opts \\ []) do
    Logger.info("Generating profile image for avatar #{avatar.id}")

    case ImageClient.generate_profile(avatar, opts) do
      {:ok, image_data} when is_binary(image_data) ->
        save_and_update_profile(avatar, image_data)

      {:error, reason} ->
        Logger.error("Failed to generate profile for #{avatar.id}: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Generate expression variations for an avatar.
  """
  @spec generate_expressions(Avatar.t(), [expression()] | nil, keyword()) ::
          {:ok, Avatar.t()} | {:error, term()}
  def generate_expressions(avatar, expressions \\ nil, opts \\ []) do
    expressions = expressions || [:happy, :sad, :neutral, :surprised, :loving]
    Logger.info("Generating #{length(expressions)} expressions for avatar #{avatar.id}")

    {:ok, expression_map} = ImageClient.generate_expression_pack(avatar, expressions, opts)

    saved_expressions =
      Enum.reduce(expression_map, %{}, fn {expr, data}, acc ->
        case save_expression_image(avatar, expr, data) do
          {:ok, url} -> Map.put(acc, Atom.to_string(expr), url)
          _ -> acc
        end
      end)

    update_avatar(avatar, %{expression_images: saved_expressions})
  end

  @doc """
  Generate 3D model for an avatar.
  """
  @spec generate_3d_model(Avatar.t(), keyword()) :: {:ok, Avatar.t()} | {:error, term()}
  def generate_3d_model(avatar, opts \\ []) do
    Logger.info("Generating 3D model for avatar #{avatar.id}")

    case Avatar3DClient.generate_for_avatar(avatar, opts) do
      {:ok, {:url, url}} ->
        update_avatar(avatar, %{avatar_3d_model_url: url})

      {:ok, model_data} when is_binary(model_data) ->
        save_and_update_3d_model(avatar, model_data)

      {:error, reason} ->
        Logger.error("Failed to generate 3D model: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Update avatar's current expression based on emotion.
  Returns the appropriate image URL for display.
  """
  @spec update_expression(Avatar.t(), emotion()) ::
          {:ok, Avatar.t(), String.t() | nil} | {:error, term()}
  def update_expression(avatar, emotion) do
    expression = emotion_to_expression(emotion)

    case update_avatar(avatar, %{current_expression: expression}) do
      {:ok, updated} ->
        image_url = get_expression_image(updated, expression)
        {:ok, updated, image_url}

      error ->
        error
    end
  end

  @doc """
  Get the appropriate expression image URL.
  Falls back to profile image if expression not available.
  """
  @spec get_expression_image(Avatar.t(), expression()) :: String.t() | nil
  def get_expression_image(avatar, expression) do
    expr_key = Atom.to_string(expression)

    case Map.get(avatar.expression_images || %{}, expr_key) do
      nil -> avatar.profile_image_url || avatar.avatar_url
      url -> url
    end
  end

  @doc """
  Generate lipsync animation from audio.
  Returns blendshape data for 3D avatar animation.
  """
  @spec generate_lipsync(Avatar.t(), binary(), keyword()) :: {:ok, map()} | {:error, term()}
  def generate_lipsync(avatar, audio_data, opts \\ []) do
    emotion = Keyword.get(opts, :emotion, avatar.current_expression)

    Avatar3DClient.audio_to_face(audio_data, Keyword.put(opts, :emotion, emotion))
  end

  @doc """
  Stream lipsync in real-time.
  """
  @spec stream_lipsync(Avatar.t(), (map() -> any()), keyword()) :: {:ok, term()} | {:error, term()}
  def stream_lipsync(avatar, callback, opts \\ []) do
    emotion = Keyword.get(opts, :emotion, avatar.current_expression)

    Avatar3DClient.audio_to_face_stream(callback, Keyword.put(opts, :emotion, emotion))
  end

  @doc """
  Check if avatar needs visual generation.
  """
  @spec needs_generation?(Avatar.t()) :: boolean()
  def needs_generation?(avatar) do
    is_nil(avatar.profile_image_url)
  end

  @doc """
  Check if avatar has 3D model.
  """
  @spec has_3d_model?(Avatar.t()) :: boolean()
  def has_3d_model?(avatar) do
    not is_nil(avatar.avatar_3d_model_url)
  end

  defp save_and_update_profile(avatar, image_data) do
    filename = "#{avatar.id}_profile.png"
    path = Path.join([@storage_base, avatar.id, filename])
    public_url = Path.join([@public_base, avatar.id, filename])

    with :ok <- ensure_directory(Path.dirname(path)),
         :ok <- File.write(path, image_data) do
      update_avatar(avatar, %{profile_image_url: public_url})
    else
      {:error, reason} ->
        Logger.error("Failed to save profile image: #{inspect(reason)}")
        {:error, :storage_failed}
    end
  end

  defp save_expression_image(avatar, expression, image_data) do
    filename = "#{avatar.id}_#{expression}.png"
    path = Path.join([@storage_base, avatar.id, filename])
    public_url = Path.join([@public_base, avatar.id, filename])

    with :ok <- ensure_directory(Path.dirname(path)),
         :ok <- File.write(path, image_data) do
      {:ok, public_url}
    else
      {:error, reason} ->
        Logger.error("Failed to save expression image: #{inspect(reason)}")
        {:error, :storage_failed}
    end
  end

  defp save_and_update_3d_model(avatar, model_data) do
    filename = "#{avatar.id}_model.glb"
    path = Path.join([@storage_base, avatar.id, filename])
    public_url = Path.join([@public_base, avatar.id, filename])

    with :ok <- ensure_directory(Path.dirname(path)),
         :ok <- File.write(path, model_data) do
      update_avatar(avatar, %{avatar_3d_model_url: public_url})
    else
      {:error, reason} ->
        Logger.error("Failed to save 3D model: #{inspect(reason)}")
        {:error, :storage_failed}
    end
  end

  defp update_avatar(avatar, attrs) do
    avatar
    |> Avatar.changeset(attrs)
    |> Repo.update()
  end

  defp ensure_directory(path) do
    case File.mkdir_p(path) do
      :ok -> :ok
      {:error, :eexist} -> :ok
      error -> error
    end
  end

  defp emotion_to_expression(:joy), do: :happy
  defp emotion_to_expression(:happiness), do: :happy
  defp emotion_to_expression(:excitement), do: :happy
  defp emotion_to_expression(:love), do: :loving
  defp emotion_to_expression(:affection), do: :loving
  defp emotion_to_expression(:sadness), do: :sad
  defp emotion_to_expression(:melancholy), do: :sad
  defp emotion_to_expression(:anger), do: :angry
  defp emotion_to_expression(:frustration), do: :angry
  defp emotion_to_expression(:surprise), do: :surprised
  defp emotion_to_expression(:curiosity), do: :surprised
  defp emotion_to_expression(:neutral), do: :neutral
  defp emotion_to_expression(:calm), do: :neutral
  defp emotion_to_expression(_), do: :neutral
end
