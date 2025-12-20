defmodule Viva.Avatars.Memory do
  @moduledoc """
  Memory schema for avatar experiences and interactions.
  Uses vector embeddings for semantic search and recall.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "memories" do
    belongs_to :avatar, Viva.Avatars.Avatar

    # Memory content
    field :content, :string
    field :summary, :string

    # Type of memory
    field :type, Ecto.Enum,
      values: [:interaction, :experience, :reflection, :dream, :milestone, :emotional_peak],
      default: :interaction

    # Participants in this memory
    field :participant_ids, {:array, :binary_id}, default: []

    # Emotional context when memory was formed
    field :emotions_felt, :map, default: %{}

    # Importance score (affects recall priority)
    field :importance, :float, default: 0.5

    # Vector embedding for semantic search (pgvector)
    field :embedding, Pgvector.Ecto.Vector

    # Memory consolidation
    field :strength, :float, default: 1.0
    field :times_recalled, :integer, default: 0
    field :last_recalled_at, :utc_datetime

    # Metadata
    field :context, :map, default: %{}

    timestamps(type: :utc_datetime)
  end

  def changeset(memory, attrs) do
    memory
    |> cast(attrs, [
      :avatar_id,
      :content,
      :summary,
      :type,
      :participant_ids,
      :emotions_felt,
      :importance,
      :embedding,
      :strength,
      :times_recalled,
      :last_recalled_at,
      :context
    ])
    |> validate_required([:avatar_id, :content])
    |> validate_number(:importance, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:strength, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> foreign_key_constraint(:avatar_id)
  end

  @doc "Apply memory decay based on time passed"
  def decay_strength(memory, hours_passed) do
    # Ebbinghaus forgetting curve approximation
    decay_rate = 0.1
    retention = :math.exp(-decay_rate * hours_passed / 24)
    base_strength = memory.strength

    # Important memories decay slower
    importance_factor = 1 + memory.importance * 0.5

    # Frequently recalled memories are more stable
    recall_factor = 1 + :math.log(memory.times_recalled + 1) * 0.2

    new_strength =
      (base_strength * retention * importance_factor * recall_factor)
      |> min(1.0)
      |> max(0.01)

    %{memory | strength: new_strength}
  end

  @doc "Record that this memory was recalled"
  def record_recall(memory) do
    %{
      memory
      | times_recalled: memory.times_recalled + 1,
        strength: min(memory.strength * 1.1, 1.0),
        last_recalled_at: DateTime.utc_now()
    }
  end

  # Queries
  def for_avatar(avatar_id) do
    from(m in __MODULE__, where: m.avatar_id == ^avatar_id)
  end

  def recent(query \\ __MODULE__, limit \\ 20) do
    from(m in query,
      order_by: [desc: m.inserted_at],
      limit: ^limit
    )
  end

  def by_importance(query \\ __MODULE__) do
    from(m in query,
      order_by: [desc: m.importance, desc: m.strength]
    )
  end

  def with_participant(query \\ __MODULE__, participant_id) do
    from(m in query,
      where: ^participant_id in m.participant_ids
    )
  end

  def of_type(query \\ __MODULE__, type) do
    from(m in query, where: m.type == ^type)
  end

  @doc "Semantic search using vector similarity"
  def similar_to(query \\ __MODULE__, embedding, limit \\ 10) do
    from(m in query,
      order_by: fragment("embedding <-> ?", ^embedding),
      limit: ^limit
    )
  end
end
