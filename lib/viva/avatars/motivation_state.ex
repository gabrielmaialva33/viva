defmodule Viva.Avatars.MotivationState do
  @moduledoc """
  Tracks avatar's motivational drives hierarchy.

  Based on Maslow's hierarchy + Enneagram triads, each avatar has:
  - Primary drive derived from Enneagram type
  - Current urgencies for each of 6 drives (0.0-1.0)
  - Blocked drive tracking for frustration dynamics

  ## Drive Hierarchy (Maslow-inspired)
  1. :survival - Basic needs (food, rest, safety from immediate harm)
  2. :safety - Security, stability, predictability
  3. :belonging - Connection, community, attachment
  4. :status - Recognition, competence, respect
  5. :autonomy - Freedom, self-determination, choice
  6. :transcendence - Meaning, purpose, beauty, spirituality

  ## Enneagram → Primary Drive Mapping
  - Types 1, 2, 3 (Shame/Heart triad) → :achievement (status-focused)
  - Types 5, 6, 7 (Fear/Head triad) → :safety_seeking
  - Types 4, 8, 9 (Rage/Gut triad) → :autonomy_seeking
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # === Primary Drive (from Enneagram) ===
    field :primary_drive, Ecto.Enum,
      values: [:achievement, :safety_seeking, :autonomy_seeking],
      default: :safety_seeking

    # === Drive Urgencies (0.0-1.0) ===
    # Higher = more urgent need for satisfaction
    field :survival_urgency, :float, default: 0.1
    field :safety_urgency, :float, default: 0.3
    field :belonging_urgency, :float, default: 0.4
    field :status_urgency, :float, default: 0.3
    field :autonomy_urgency, :float, default: 0.3
    field :transcendence_urgency, :float, default: 0.2

    # === Frustration Tracking ===
    # When a drive is repeatedly blocked, avatar becomes frustrated
    field :blocked_drive, Ecto.Enum,
      values: [:survival, :safety, :belonging, :status, :autonomy, :transcendence],
      default: nil

    field :block_duration, :integer, default: 0

    # === Current State ===
    # Last computed urgent drive (cached for efficiency)
    field :current_urgent_drive, Ecto.Enum,
      values: [:survival, :safety, :belonging, :status, :autonomy, :transcendence],
      default: :belonging

    # When urgencies were last updated
    field :last_updated, :utc_datetime
  end

  @type drive ::
          :survival
          | :safety
          | :belonging
          | :status
          | :autonomy
          | :transcendence

  @type primary_drive :: :achievement | :safety_seeking | :autonomy_seeking

  @type t :: %__MODULE__{
          primary_drive: primary_drive(),
          survival_urgency: float(),
          safety_urgency: float(),
          belonging_urgency: float(),
          status_urgency: float(),
          autonomy_urgency: float(),
          transcendence_urgency: float(),
          blocked_drive: drive() | nil,
          block_duration: non_neg_integer(),
          current_urgent_drive: drive(),
          last_updated: DateTime.t() | nil
        }

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :primary_drive,
      :survival_urgency,
      :safety_urgency,
      :belonging_urgency,
      :status_urgency,
      :autonomy_urgency,
      :transcendence_urgency,
      :blocked_drive,
      :block_duration,
      :current_urgent_drive,
      :last_updated
    ])
    |> validate_number(:survival_urgency, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:safety_urgency, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:belonging_urgency,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:status_urgency, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:autonomy_urgency,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:transcendence_urgency,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:block_duration, greater_than_or_equal_to: 0)
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      primary_drive: :safety_seeking,
      survival_urgency: 0.1,
      safety_urgency: 0.3,
      belonging_urgency: 0.4,
      status_urgency: 0.3,
      autonomy_urgency: 0.3,
      transcendence_urgency: 0.2,
      blocked_drive: nil,
      block_duration: 0,
      current_urgent_drive: :belonging,
      last_updated: nil
    }
  end

  @doc """
  Creates motivation state from Enneagram type.
  Maps type to primary drive and sets initial urgencies accordingly.
  """
  @spec from_enneagram(atom() | nil) :: t()
  def from_enneagram(nil), do: new()

  def from_enneagram(enneagram_type) do
    primary_drive = type_to_primary_drive(enneagram_type)
    base_urgencies = base_urgencies_for_drive(primary_drive)

    %__MODULE__{
      primary_drive: primary_drive,
      survival_urgency: base_urgencies.survival,
      safety_urgency: base_urgencies.safety,
      belonging_urgency: base_urgencies.belonging,
      status_urgency: base_urgencies.status,
      autonomy_urgency: base_urgencies.autonomy,
      transcendence_urgency: base_urgencies.transcendence,
      blocked_drive: nil,
      block_duration: 0,
      current_urgent_drive: find_most_urgent(base_urgencies),
      last_updated: DateTime.utc_now(:second)
    }
  end

  @doc """
  Returns the urgencies as a map for easier manipulation.
  """
  @spec urgencies(t()) :: %{drive() => float()}
  def urgencies(%__MODULE__{} = state) do
    %{
      survival: state.survival_urgency,
      safety: state.safety_urgency,
      belonging: state.belonging_urgency,
      status: state.status_urgency,
      autonomy: state.autonomy_urgency,
      transcendence: state.transcendence_urgency
    }
  end

  @doc """
  Returns true if a drive is currently blocked/frustrated.
  """
  @spec frustrated?(t()) :: boolean()
  def frustrated?(%__MODULE__{blocked_drive: nil}), do: false
  def frustrated?(%__MODULE__{block_duration: d}) when d > 3, do: true
  def frustrated?(_), do: false

  @doc """
  Returns true if avatar is in survival mode (high survival urgency).
  """
  @spec survival_mode?(t()) :: boolean()
  def survival_mode?(%__MODULE__{survival_urgency: u}) when u > 0.7, do: true
  def survival_mode?(_), do: false

  # === Private Functions ===

  defp type_to_primary_drive(type) do
    case type do
      # Shame/Heart triad - Achievement focused
      :type_1 -> :achievement
      :type_2 -> :achievement
      :type_3 -> :achievement
      # Fear/Head triad - Safety focused
      :type_5 -> :safety_seeking
      :type_6 -> :safety_seeking
      :type_7 -> :safety_seeking
      # Rage/Gut triad - Autonomy focused
      :type_4 -> :autonomy_seeking
      :type_8 -> :autonomy_seeking
      :type_9 -> :autonomy_seeking
      # Default
      _ -> :safety_seeking
    end
  end

  defp base_urgencies_for_drive(:achievement) do
    %{
      survival: 0.1,
      safety: 0.3,
      belonging: 0.5,
      status: 0.7,
      autonomy: 0.5,
      transcendence: 0.4
    }
  end

  defp base_urgencies_for_drive(:safety_seeking) do
    %{
      survival: 0.2,
      safety: 0.7,
      belonging: 0.6,
      status: 0.3,
      autonomy: 0.3,
      transcendence: 0.3
    }
  end

  defp base_urgencies_for_drive(:autonomy_seeking) do
    %{
      survival: 0.1,
      safety: 0.4,
      belonging: 0.4,
      status: 0.5,
      autonomy: 0.7,
      transcendence: 0.5
    }
  end

  defp find_most_urgent(urgencies) do
    urgencies
    |> Enum.max_by(fn {_, v} -> v end)
    |> elem(0)
  end
end
