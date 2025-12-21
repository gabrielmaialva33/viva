defmodule Viva.Repo.Migrations.AddPerformanceIndexes do
  @moduledoc """
  Adds composite and specialized indexes for query optimization.

  Based on analysis of query patterns in:
  - Viva.Avatars.Memory (semantic search, decay, filtering)
  - Viva.Relationships (matchmaking, status filtering)
  - Viva.Conversations (active conversations, message retrieval)
  - Viva.Avatars (user filtering)
  """
  use Ecto.Migration

  def change do
    # ============================================
    # MEMORIES - Optimized for semantic search and decay
    # ============================================

    # Composite: avatar's memories by type (common filter pattern)
    create index(:memories, [:avatar_id, :type])

    # Composite: avatar's top memories by importance/strength
    create index(:memories, [:avatar_id, :importance, :strength],
             name: :memories_avatar_priority_idx
           )

    # For decay queries (strength-based filtering)
    create index(:memories, [:strength])

    # GIN index for participant_ids array lookups (shared memories)
    execute(
      "CREATE INDEX memories_participant_ids_idx ON memories USING GIN (participant_ids)",
      "DROP INDEX IF EXISTS memories_participant_ids_idx"
    )

    # ============================================
    # RELATIONSHIPS - Optimized for matchmaking
    # ============================================

    # Composite: find avatar's relationships by status
    create index(:relationships, [:avatar_a_id, :status])
    create index(:relationships, [:avatar_b_id, :status])

    # For matchmaking sorting (compatibility-based)
    create index(:relationships, [:compatibility_score])

    # For romantic matching queries
    create index(:relationships, [:affection])
    create index(:relationships, [:attraction])

    # Composite: active relationships with high metrics
    create index(:relationships, [:status, :familiarity],
             name: :relationships_status_familiarity_idx
           )

    # ============================================
    # CONVERSATIONS - Optimized for active lookups
    # ============================================

    # Composite: avatar's conversations by status (find active)
    create index(:conversations, [:avatar_a_id, :status])
    create index(:conversations, [:avatar_b_id, :status])

    # Composite: filter by type and status (autonomous active)
    create index(:conversations, [:status, :type])

    # ============================================
    # MESSAGES - Optimized for ordered retrieval
    # ============================================

    # Composite: messages in conversation ordered by time
    # This is the most common query pattern
    create index(:messages, [:conversation_id, :timestamp])

    # ============================================
    # AVATARS - Optimized for user queries
    # ============================================

    # Composite: user's active avatars
    create index(:avatars, [:user_id, :is_active])

    # For name-based search (if needed later)
    create index(:avatars, [:name])
  end
end
