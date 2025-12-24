defmodule Viva.Avatars.AllostasisState do
  @moduledoc """
  Tracks allostatic load - the cumulative physiological cost of chronic stress.

  When cortisol stays elevated for extended periods:
  - Receptor downregulation occurs (emotional blunting)
  - Recovery capacity decreases
  - Cognitive function is impaired
  - Risk of "burnout" increases

  This is the long-term consequence of the short-term stress system.
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # Cumulative allostatic load (0.0 = fresh, 1.0 = burnout)
    field :load_level, :float, default: 0.0

    # Receptor sensitivity (1.0 = normal, 0.0 = completely desensitized)
    # Affects how strongly emotions are experienced
    field :receptor_sensitivity, :float, default: 1.0

    # Recovery capacity (1.0 = normal recovery rate, 0.0 = cannot recover)
    field :recovery_capacity, :float, default: 1.0

    # Cognitive impairment level (0.0 = none, 1.0 = severe)
    field :cognitive_impairment, :float, default: 0.0

    # Running history of cortisol levels (for trend detection)
    field :cortisol_history, {:array, :float}, default: []

    # Time spent in high-stress state (in simulated hours)
    field :high_stress_hours, :float, default: 0.0

    # Last recovery period (when load decreased significantly)
    field :last_recovery_at, :utc_datetime
  end

  @type t :: %__MODULE__{
          load_level: float(),
          receptor_sensitivity: float(),
          recovery_capacity: float(),
          cognitive_impairment: float(),
          cortisol_history: [float()],
          high_stress_hours: float(),
          last_recovery_at: DateTime.t() | nil
        }

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(state, attrs) do
    state
    |> cast(attrs, [
      :load_level,
      :receptor_sensitivity,
      :recovery_capacity,
      :cognitive_impairment,
      :cortisol_history,
      :high_stress_hours,
      :last_recovery_at
    ])
    |> validate_number(:load_level, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> validate_number(:receptor_sensitivity,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:recovery_capacity,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:cognitive_impairment,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      load_level: 0.0,
      receptor_sensitivity: 1.0,
      recovery_capacity: 1.0,
      cognitive_impairment: 0.0,
      cortisol_history: [],
      high_stress_hours: 0.0,
      last_recovery_at: nil
    }
  end
end
