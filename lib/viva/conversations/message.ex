defmodule Viva.Conversations.Message do
  @moduledoc """
  Message schema for individual messages in conversations.
  """
  use Ecto.Schema
  import Ecto.Changeset

  alias Viva.Avatars.Avatar
  alias Viva.Conversations.Conversation

  @type t :: %__MODULE__{}

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "messages" do
    belongs_to :conversation, Conversation
    belongs_to :speaker, Avatar

    # Content
    field :content, :string
    # text, audio, image
    field :content_type, :string, default: "text"

    # Emotional context
    field :emotional_tone, :string
    field :emotions, :map, default: %{}

    # Audio (if voice message)
    field :audio_url, :string
    field :duration_seconds, :float

    # Timing
    field :timestamp, :utc_datetime

    # Generation metadata
    # monotonic time for latency tracking
    field :generated_at, :integer

    timestamps(type: :utc_datetime)
  end

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(message, attrs) do
    message
    |> cast(attrs, [
      :conversation_id,
      :speaker_id,
      :content,
      :content_type,
      :emotional_tone,
      :emotions,
      :audio_url,
      :duration_seconds,
      :timestamp,
      :generated_at
    ])
    |> validate_required([:conversation_id, :content, :timestamp])
    |> validate_inclusion(:content_type, ["text", "audio", "image"])
    |> foreign_key_constraint(:conversation_id)
    |> foreign_key_constraint(:speaker_id)
  end
end
