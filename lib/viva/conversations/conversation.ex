defmodule Viva.Conversations.Conversation do
  @moduledoc """
  Conversation schema for tracking conversations between avatars.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Viva.Avatars.Avatar
  alias Viva.Conversations.Message

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "conversations" do
    belongs_to :avatar_a, Avatar
    belongs_to :avatar_b, Avatar

    # Type: interactive (owner-initiated) or autonomous (AI-initiated)
    field :type, :string, default: "interactive"

    # Status: active, ended, paused
    field :status, :string, default: "active"

    # Context and topic
    field :context, :map, default: %{}
    field :topic, :string

    # Timing
    field :started_at, :utc_datetime
    field :ended_at, :utc_datetime
    field :duration_minutes, :integer

    # Stats
    field :message_count, :integer, default: 0

    # Analysis (filled after conversation ends)
    field :analysis, :map

    has_many :messages, Message

    timestamps(type: :utc_datetime)
  end

  def changeset(conversation, attrs) do
    conversation
    |> cast(attrs, [
      :avatar_a_id,
      :avatar_b_id,
      :type,
      :status,
      :context,
      :topic,
      :started_at,
      :ended_at,
      :duration_minutes,
      :message_count,
      :analysis
    ])
    |> validate_required([:avatar_a_id, :avatar_b_id])
    |> validate_inclusion(:type, ["interactive", "autonomous"])
    |> validate_inclusion(:status, ["active", "ended", "paused"])
    |> foreign_key_constraint(:avatar_a_id)
    |> foreign_key_constraint(:avatar_b_id)
  end

  # === Query Helpers ===

  def involving(query, avatar_id) do
    where(query, [c], c.avatar_a_id == ^avatar_id or c.avatar_b_id == ^avatar_id)
  end

  def between(query, avatar_a_id, avatar_b_id) do
    where(
      query,
      [c],
      (c.avatar_a_id == ^avatar_a_id and c.avatar_b_id == ^avatar_b_id) or
        (c.avatar_a_id == ^avatar_b_id and c.avatar_b_id == ^avatar_a_id)
    )
  end

  def active(query) do
    where(query, [c], c.status == "active")
  end

  def autonomous(query) do
    where(query, [c], c.type == "autonomous")
  end
end
