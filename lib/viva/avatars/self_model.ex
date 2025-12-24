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

    # === Dynamic Identity (Coherence Tracking) ===
    # How coherent is my self-image? (0.0 = identity crisis, 1.0 = fully integrated)
    field :coherence_level, :float, default: 0.8

    # Experiences that contradicted my self-image
    # Each: %{experience_type: atom, claimed_aspect: string, occurred_at: datetime}
    field :contradictions, {:array, :map}, default: []

    # History of identity renegotiations
    # Each: %{old_belief: string, new_belief: string, trigger: string, occurred_at: datetime}
    field :identity_negotiations, {:array, :map}, default: []

    # When identity was last updated
    field :last_identity_update, :utc_datetime
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
          social_identities: list(String.t()),
          coherence_level: float(),
          contradictions: list(map()),
          identity_negotiations: list(map()),
          last_identity_update: DateTime.t() | nil
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
      :social_identities,
      :coherence_level,
      :contradictions,
      :identity_negotiations,
      :last_identity_update
    ])
    |> validate_number(:self_esteem,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:self_efficacy,
      greater_than_or_equal_to: 0.0,
      less_than_or_equal_to: 1.0
    )
    |> validate_number(:coherence_level,
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
      social_identities: [],
      coherence_level: 0.8,
      contradictions: [],
      identity_negotiations: [],
      last_identity_update: nil
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

  # === Dynamic Identity Functions ===

  @doc """
  Returns true if the avatar is experiencing an identity crisis.
  Low coherence indicates significant contradictions in self-image.
  """
  @spec identity_crisis?(t()) :: boolean()
  def identity_crisis?(%__MODULE__{coherence_level: level}), do: level < 0.4

  @doc """
  Returns true if identity is somewhat fragmented but not in crisis.
  """
  @spec identity_uncertain?(t()) :: boolean()
  def identity_uncertain?(%__MODULE__{coherence_level: level}), do: level < 0.6

  @doc """
  Integrates an experience and checks for contradictions with self-model.
  Returns updated self-model with potential coherence changes.

  ## Experience Types
  - :social_rejection - Contradicts "I am liked"
  - :failure - Contradicts "I am competent"
  - :success - Reinforces "I am capable"
  - :betrayal - Contradicts "I can trust others"
  - :altruistic_action - Contradicts "I am selfish"
  - :cowardice - Contradicts "I am brave"
  - :kindness_received - Contradicts "Others are against me"
  """
  @spec integrate_experience(t(), map()) :: t()
  def integrate_experience(self_model, experience) do
    # Extract what this experience claims about self
    claims = extract_self_claims(experience)

    # Check each claim against current self-model
    contradictions =
      claims
      |> Enum.map(fn claim -> detect_contradiction(self_model, claim, experience) end)
      |> Enum.filter(& &1)

    if Enum.empty?(contradictions) do
      # No contradictions - possibly reinforce existing beliefs
      maybe_reinforce_identity(self_model, experience)
    else
      # Contradictions detected - update coherence and record
      negotiate_identity(self_model, contradictions, experience)
    end
  end

  @doc """
  Describes the current state of identity coherence.
  """
  @spec describe_coherence(t()) :: String.t()
  def describe_coherence(%__MODULE__{coherence_level: level, contradictions: contradictions}) do
    recent_contradictions = length(contradictions)

    cond do
      level < 0.3 ->
        "I don't recognize myself anymore. Everything I believed about who I am is being questioned."

      level < 0.5 ->
        "I'm going through a change. Not sure who I am anymore."

      level < 0.7 and recent_contradictions > 0 ->
        "Some experiences have challenged what I thought I knew about myself."

      level < 0.7 ->
        "I'm still figuring myself out."

      true ->
        "I have a clear sense of who I am."
    end
  end

  @doc """
  Recovers coherence over time (called during reflection/sleep).
  Integration happens naturally as the avatar processes experiences.
  """
  @spec recover_coherence(t(), float()) :: t()
  def recover_coherence(self_model, recovery_amount \\ 0.05) do
    new_coherence = min(1.0, self_model.coherence_level + recovery_amount)
    %{self_model | coherence_level: new_coherence}
  end

  # === Private Helpers ===

  defp extract_self_claims(experience) do
    case experience do
      %{type: :social_rejection, intensity: i} when i > 0.6 ->
        ["I am liked", "I belong"]

      %{type: :failure, intensity: i} when i > 0.5 ->
        ["I am competent", "I am capable"]

      %{type: :success, intensity: i} when i > 0.5 ->
        ["I struggle", "I often fail"]

      %{type: :betrayal, intensity: i} when i > 0.6 ->
        ["I can trust others", "People are reliable"]

      %{type: :altruistic_action, intensity: i} when i > 0.4 ->
        ["I am selfish"]

      %{type: :cowardice, intensity: i} when i > 0.5 ->
        ["I am brave", "I face my fears"]

      %{type: :kindness_received, intensity: i} when i > 0.4 ->
        ["Others are against me", "I am alone"]

      _ ->
        []
    end
  end

  defp detect_contradiction(self_model, claim, experience) do
    # Check if claim contradicts current beliefs
    contradicting_belief =
      Enum.find(self_model.core_beliefs, fn belief ->
        belief_matches_claim?(belief[:belief], claim)
      end)

    if contradicting_belief do
      %{
        experience_type: experience[:type],
        claimed_aspect: claim,
        contradicted_belief: contradicting_belief[:belief],
        intensity: experience[:intensity] || 0.5,
        occurred_at: DateTime.utc_now(:second)
      }
    else
      nil
    end
  end

  defp belief_matches_claim?(belief, claim) when is_binary(belief) and is_binary(claim) do
    # Simple check - could be enhanced with semantic matching
    belief_lower = String.downcase(belief)
    claim_lower = String.downcase(claim)

    String.contains?(belief_lower, claim_lower) or String.contains?(claim_lower, belief_lower)
  end

  defp belief_matches_claim?(_, _), do: false

  defp negotiate_identity(self_model, contradictions, _) do
    # Calculate coherence decrease based on number and intensity of contradictions
    total_impact =
      Enum.reduce(contradictions, 0.0, fn c, acc ->
        acc + (c.intensity || 0.5) * 0.1
      end)

    coherence_decrease = min(total_impact, 0.3)

    # Record negotiations
    negotiations =
      Enum.map(contradictions, fn c ->
        %{
          old_belief: c.contradicted_belief,
          new_belief: nil,
          trigger: "#{c.experience_type}: #{c.claimed_aspect}",
          occurred_at: DateTime.utc_now(:second)
        }
      end)

    new_coherence = max(0.1, self_model.coherence_level - coherence_decrease)

    self_model
    |> Map.put(:coherence_level, new_coherence)
    |> Map.put(:contradictions, Enum.take(self_model.contradictions ++ contradictions, -10))
    |> Map.put(
      :identity_negotiations,
      Enum.take(self_model.identity_negotiations ++ negotiations, -10)
    )
    |> Map.put(:last_identity_update, DateTime.utc_now(:second))
    |> maybe_update_narrative()
  end

  defp maybe_reinforce_identity(self_model, %{type: :success, intensity: i}) when i > 0.6 do
    # Success reinforces self-efficacy
    new_efficacy = min(1.0, self_model.self_efficacy + 0.02)
    new_coherence = min(1.0, self_model.coherence_level + 0.01)

    %{self_model | self_efficacy: new_efficacy, coherence_level: new_coherence}
  end

  defp maybe_reinforce_identity(self_model, %{type: :kindness_received, intensity: i})
       when i > 0.5 do
    # Kindness reinforces self-esteem
    new_esteem = min(1.0, self_model.self_esteem + 0.02)
    %{self_model | self_esteem: new_esteem}
  end

  defp maybe_reinforce_identity(self_model, _), do: self_model

  defp maybe_update_narrative(self_model) do
    if self_model.coherence_level < 0.4 and length(self_model.contradictions) > 2 do
      # Identity crisis - narrative becomes uncertain
      old_parts = String.split(self_model.identity_narrative, " ")

      uncertain_narrative =
        if length(old_parts) > 3 do
          "I thought I was #{Enum.join(Enum.take(old_parts, 4), " ")}... but I'm not sure anymore."
        else
          "I'm questioning everything I thought I knew about myself."
        end

      %{self_model | identity_narrative: uncertain_narrative}
    else
      self_model
    end
  end

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
