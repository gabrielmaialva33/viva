defmodule Viva.Collective.ThoughtCell do
  @moduledoc """
  Representa o estado de um pensamento dentro de um avatar específico.

  Cada célula é como uma "instância" do pensamento coletivo,
  mas com conteúdo potencialmente mutado pela individualidade do avatar.

  ## Estados

  - `:alive` - Pensamento ativo, pode propagar
  - `:dormant` - Pensamento adormecido, não propaga mas pode reviver
  - `:dead` - Pensamento morto, será removido

  ## Ciclo de Vida

      :alive ──────► :dormant ──────► :dead
         │              │
         └──────────────┘
           (pode reviver)
  """

  defstruct [
    :avatar_id,      # Avatar que carrega este pensamento
    :thought_id,     # ID do pensamento coletivo
    :content,        # Conteúdo (possivelmente mutado)
    :generation,     # Geração em que recebeu o pensamento
    :state,          # :alive, :dormant, :dead
    :strength,       # 0.0 - 1.0, força do pensamento
    :received_at,    # Quando recebeu
    :last_active_at  # Última atividade
  ]

  @doc """
  Cria uma nova célula de pensamento.
  """
  def new(avatar_id, thought_id, content, generation) do
    now = DateTime.utc_now()

    %__MODULE__{
      avatar_id: avatar_id,
      thought_id: thought_id,
      content: content,
      generation: generation,
      state: :alive,
      strength: 1.0,
      received_at: now,
      last_active_at: now
    }
  end

  @doc """
  Ativa uma célula dormant.
  """
  def activate(%__MODULE__{state: :dormant} = cell) do
    %{cell | state: :alive, strength: min(cell.strength + 0.3, 1.0), last_active_at: DateTime.utc_now()}
  end
  def activate(cell), do: cell

  @doc """
  Adormece uma célula.
  """
  def make_dormant(%__MODULE__{state: :alive} = cell) do
    %{cell | state: :dormant, strength: cell.strength * 0.5}
  end
  def make_dormant(cell), do: cell

  @doc """
  Mata uma célula.
  """
  def kill(cell) do
    %{cell | state: :dead, strength: 0.0}
  end

  @doc """
  Decai a força do pensamento com o tempo.
  """
  def decay(cell, rate \\ 0.1) do
    new_strength = max(cell.strength - rate, 0.0)

    cond do
      new_strength <= 0.0 ->
        kill(cell)

      new_strength <= 0.3 and cell.state == :alive ->
        %{cell | strength: new_strength, state: :dormant}

      true ->
        %{cell | strength: new_strength}
    end
  end

  @doc """
  Fortalece o pensamento (quando reforçado por vizinhos).
  """
  def strengthen(cell, amount \\ 0.2) do
    new_strength = min(cell.strength + amount, 1.0)

    case cell.state do
      :dormant when new_strength > 0.5 ->
        %{cell | strength: new_strength, state: :alive, last_active_at: DateTime.utc_now()}

      _ ->
        %{cell | strength: new_strength, last_active_at: DateTime.utc_now()}
    end
  end

  @doc """
  Verifica se a célula está viva.
  """
  def alive?(%__MODULE__{state: :alive}), do: true
  def alive?(_), do: false

  @doc """
  Verifica se a célula pode propagar.
  """
  def can_propagate?(%__MODULE__{state: :alive, strength: s}) when s > 0.5, do: true
  def can_propagate?(_), do: false
end
