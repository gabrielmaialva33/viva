defmodule Viva.Avatars.SelfModel do
  @moduledoc """
  Embedded schema for the avatar's self-concept and identity narrative.
  The avatar's understanding of who they are.

  This represents the avatar's internal model of themselves, including:
  - Core identity and values
  - Self-perception (strengths, weaknesses, esteem)
  - Recognized patterns in own behavior
  - Goals and aspirations
  """
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    # === Core Identity ===
    # "I am the kind of person who..."
    field :identity_narrative, :string, default: "I am still discovering who I am."

    # Values that define self
    field :core_values, {:array, :string}, default: []

    # Beliefs about self, others, world
    # Each: %{domain: "self"|"others"|"world", belief: x, strength: 0.0-1.0}
    field :core_beliefs, {:array, :map}, default: []

    # === Self-Perception ===
    field :perceived_strengths, {:array, :string}, default: []
    field :perceived_weaknesses, {:array, :string}, default: []

    # Overall self-esteem (0.0 to 1.0)
    field :self_esteem, :float, default: 0.5

    # Self-efficacy (belief in own capability)
    field :self_efficacy, :float, default: 0.5

    # === Pattern Recognition ===
    # Noticed patterns in own behavior
    # Each: %{trigger: x, response: y, frequency: n}
    field :behavioral_patterns, {:array, :map}, default: []

    # How I typically react emotionally
    # Each: %{situation: x, typical_emotion: y}
    field :emotional_patterns, {:array, :map}, default: []

    # === Aspirations ===
    # Active goals
    field :current_goals, {:array, :string}, default: []

    # Who I want to become
    field :ideal_self, :string

    # Who I fear becoming
    field :feared_self, :string

    # === Social Identity ===
    # How I understand my relational patterns
    field :attachment_narrative, :string

    # Groups/roles I identify with
    field :social_identities, {:array, :string}, default: []
  end

  @type t :: %__MODULE__{
          identity_narrative: String.t(),
          core_values: list(String.t()),
          core_beliefs: list(map()),
          perceived_strengths: list(String.t()),
          perceived_weaknesses: list(String.t()),
          self_esteem: float(),
          self_efficacy: float(),
          behavioral_patterns: list(map()),
          emotional_patterns: list(map()),
          current_goals: list(String.t()),
          ideal_self: String.t() | nil,
          feared_self: String.t() | nil,
          attachment_narrative: String.t() | nil,
          social_identities: list(String.t())
        }

  # === Public API ===

  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(model, attrs) do
    model
    |> cast(attrs, [
      :identity_narrative,
      :core_values,
      :core_beliefs,
      :perceived_strengths,
      :perceived_weaknesses,
      :self_esteem,
      :self_efficacy,
      :behavioral_patterns,
      :emotional_patterns,
      :current_goals,
      :ideal_self,
      :feared_self,
      :attachment_narrative,
      :social_identities
    ])
    |> validate_number(:self_esteem,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:self_efficacy,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
  end

  @spec new() :: t()
  def new do
    %__MODULE__{
      identity_narrative: "I am still discovering who I am.",
      core_values: [],
      core_beliefs: [],
      perceived_strengths: [],
      perceived_weaknesses: [],
      self_esteem: 0.5,
      self_efficacy: 0.5,
      behavioral_patterns: [],
      emotional_patterns: [],
      current_goals: [],
      ideal_self: nil,
      feared_self: nil,
      attachment_narrative: nil,
      social_identities: []
    }
  end

  @doc """
  Initialize self-model from personality and enneagram data.
  """
  @spec from_personality(Viva.Avatars.Personality.t()) :: t()
  def from_personality(personality) do
    enneagram_data = get_enneagram_data(personality.enneagram_type)

    %__MODULE__{
      identity_narrative: build_identity_narrative(enneagram_data),
      core_values: personality.values || [],
      core_beliefs: build_core_beliefs(enneagram_data),
      perceived_strengths: derive_strengths(personality),
      perceived_weaknesses: derive_weaknesses(personality),
      self_esteem: calculate_base_esteem(personality),
      self_efficacy: calculate_base_efficacy(personality),
      behavioral_patterns: [],
      emotional_patterns: [],
      current_goals: [],
      ideal_self: enneagram_data[:behavior_when_healthy],
      feared_self: enneagram_data[:behavior_when_stressed],
      attachment_narrative: describe_attachment(personality.attachment_style),
      social_identities: []
    }
  end

  @doc """
  Returns true if self-esteem is low.
  """
  @spec low_self_esteem?(t()) :: boolean()
  def low_self_esteem?(%__MODULE__{self_esteem: esteem}), do: esteem < 0.4

  @doc """
  Returns true if self-efficacy is high.
  """
  @spec confident?(t()) :: boolean()
  def confident?(%__MODULE__{self_efficacy: efficacy}), do: efficacy > 0.6

  @doc """
  Gets a belief by domain.
  """
  @spec get_beliefs_by_domain(t(), String.t()) :: list(map())
  def get_beliefs_by_domain(%__MODULE__{core_beliefs: beliefs}, domain) do
    Enum.filter(beliefs, fn b -> b[:domain] == domain end)
  end

  # === Private Helpers ===

  defp get_enneagram_data(nil), do: %{}

  defp get_enneagram_data(enneagram_type) do
    Viva.Avatars.Enneagram.get_type(enneagram_type)
  end

  defp build_identity_narrative(%{motivation: motivation}) when is_binary(motivation) do
    "I am someone who #{motivation}."
  end

  defp build_identity_narrative(_), do: "I am still discovering who I am."

  defp build_core_beliefs(%{basic_fear: fear, basic_desire: desire})
       when is_binary(fear) and is_binary(desire) do
    [
      %{domain: "self", belief: "I fear #{fear}", strength: 0.8},
      %{domain: "self", belief: "I desire #{desire}", strength: 0.9}
    ]
  end

  defp build_core_beliefs(_), do: []

  defp derive_strengths(personality) do
    []
    |> maybe_add(personality.openness > 0.6, "creative thinking")
    |> maybe_add(personality.conscientiousness > 0.6, "reliability and organization")
    |> maybe_add(personality.extraversion > 0.6, "social energy and enthusiasm")
    |> maybe_add(personality.agreeableness > 0.6, "empathy and cooperation")
    |> maybe_add(personality.neuroticism < 0.3, "emotional stability")
    |> maybe_add(personality.openness > 0.7, "curiosity and imagination")
    |> maybe_add(personality.conscientiousness > 0.7, "discipline and focus")
  end

  defp derive_weaknesses(personality) do
    []
    |> maybe_add(personality.openness < 0.3, "resistance to change")
    |> maybe_add(personality.conscientiousness < 0.3, "disorganization")
    |> maybe_add(personality.extraversion < 0.3, "social withdrawal")
    |> maybe_add(personality.agreeableness < 0.3, "difficulty trusting others")
    |> maybe_add(personality.neuroticism > 0.7, "emotional sensitivity")
    |> maybe_add(personality.neuroticism > 0.8, "anxiety proneness")
  end

  defp maybe_add(list, true, item), do: [item | list]
  defp maybe_add(list, false, _), do: list

  defp calculate_base_esteem(personality) do
    # Low neuroticism and high extraversion correlate with self-esteem
    base = 0.5
    neuroticism_effect = (0.5 - personality.neuroticism) * 0.3
    extraversion_effect = (personality.extraversion - 0.5) * 0.2

    (base + neuroticism_effect + extraversion_effect)
    |> max(0.2)
    |> min(0.8)
  end

  defp calculate_base_efficacy(personality) do
    # Conscientiousness and openness correlate with self-efficacy
    base = 0.5
    conscientiousness_effect = (personality.conscientiousness - 0.5) * 0.3
    openness_effect = (personality.openness - 0.5) * 0.2

    (base + conscientiousness_effect + openness_effect)
    |> max(0.2)
    |> min(0.8)
  end

  defp describe_attachment(:secure) do
    "I generally trust others and feel comfortable with intimacy."
  end

  defp describe_attachment(:anxious) do
    "I worry about abandonment and crave closeness."
  end

  defp describe_attachment(:avoidant) do
    "I value independence and sometimes struggle with closeness."
  end

  defp describe_attachment(:fearful) do
    "I desire closeness but fear getting hurt."
  end

  defp describe_attachment(_), do: "I am learning about how I relate to others."
end
