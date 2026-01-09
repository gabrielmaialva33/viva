defmodule Viva.Social.SocialGraph do
  @moduledoc """
  Interface para o grafo social de avatares.
  Fornece informações sobre vizinhança e relacionamentos para propagação de pensamentos.
  """

  alias Viva.Relationships

  @doc """
  Retorna os IDs dos avatares vizinhos (com relacionamento) de um avatar.
  """
  @spec get_neighbors(String.t()) :: {:ok, [String.t()]} | {:error, term()}
  def get_neighbors(avatar_id) do
    case Relationships.list_relationships(avatar_id) do
      relationships when is_list(relationships) ->
        neighbor_ids = Enum.map(relationships, fn rel ->
          # Relationship uses avatar_a_id and avatar_b_id
          if rel.avatar_a_id == avatar_id, do: rel.avatar_b_id, else: rel.avatar_a_id
        end)
        {:ok, neighbor_ids}

      _ ->
        {:ok, []}
    end
  rescue
    _ -> {:ok, []}
  end

  @doc """
  Retorna informações sobre o relacionamento entre dois avatares.
  """
  @spec get_relationship(String.t(), String.t()) :: {:ok, map()} | {:error, :not_found}
  def get_relationship(avatar_id_1, avatar_id_2) do
    case Relationships.get_relationship_between(avatar_id_1, avatar_id_2) do
      nil ->
        {:error, :not_found}

      relationship ->
        {:ok, %{
          affinity: calculate_affinity(relationship),
          trust: relationship.trust || 0.5,
          familiarity: relationship.familiarity || 0.0
        }}
    end
  rescue
    _ -> {:error, :not_found}
  end

  # Calcula afinidade baseada nos atributos do relacionamento
  defp calculate_affinity(relationship) do
    trust = relationship.trust || 0.5
    affection = relationship.affection || 0.5
    familiarity = relationship.familiarity || 0.0

    # Média ponderada
    (trust * 0.3 + affection * 0.5 + familiarity * 0.2)
    |> min(1.0)
    |> max(0.0)
  end
end
