defmodule Viva.Avatars.Systems.AttachmentBias do
  @moduledoc """
  Filters and interprets social stimuli through the lens of attachment style.

  Different attachment styles create different interpretations of the same events:
  - **Secure**: Neutral/positive interpretations, assumes good intent
  - **Anxious**: Hypervigilant to rejection cues, amplifies negative signals
  - **Avoidant**: Dismisses social importance, minimizes connection needs
  - **Fearful**: Wants connection but expects rejection, conflicted responses

  This system runs BEFORE Senses.perceive(), coloring the stimulus before
  it reaches conscious awareness.
  """

  alias Viva.Avatars.Personality

  @type stimulus :: map()
  @type attachment_style :: :secure | :anxious | :avoidant | :fearful
  @type interpretation :: %{
          style: attachment_style(),
          bias_applied: boolean(),
          original_valence: float(),
          interpretation: String.t() | nil
        }

  # Social situations that trigger attachment-based interpretation
  @social_types [:social, :social_ambient]

  # Valence modification by attachment style for ambiguous situations
  @valence_bias %{
    secure: 0.1,
    anxious: -0.15,
    avoidant: -0.05,
    fearful: -0.1
  }

  # Threat level modification by attachment style
  @threat_bias %{
    secure: -0.1,
    anxious: 0.2,
    avoidant: 0.0,
    fearful: 0.15
  }

  # Intensity modification (anxious amplifies, avoidant dampens)
  @intensity_bias %{
    secure: 0.0,
    anxious: 0.15,
    avoidant: -0.1,
    fearful: 0.1
  }

  @doc """
  Interprets a stimulus through the attachment style lens.
  Returns the modified stimulus and interpretation metadata.
  """
  @spec interpret(stimulus(), Personality.t()) :: {stimulus(), interpretation()}
  def interpret(stimulus, personality) do
    style = personality.attachment_style

    if social_stimulus?(stimulus) do
      {apply_bias(stimulus, style), create_interpretation(stimulus, style, true)}
    else
      {stimulus, create_interpretation(stimulus, style, false)}
    end
  end

  @doc """
  Returns a narrative interpretation of a social situation based on attachment style.
  """
  @spec interpret_situation(atom(), attachment_style()) :: String.t()
  def interpret_situation(:no_response, :secure), do: "They're probably busy"

  def interpret_situation(:no_response, :anxious),
    do: "They're ignoring me... did I do something wrong?"

  def interpret_situation(:no_response, :avoidant), do: "I don't need their response anyway"
  def interpret_situation(:no_response, :fearful), do: "I knew they would reject me eventually"

  def interpret_situation(:criticism, :secure), do: "That's useful feedback to consider"
  def interpret_situation(:criticism, :anxious), do: "They must hate me, I'm not good enough"
  def interpret_situation(:criticism, :avoidant), do: "They're being controlling, I don't need this"
  def interpret_situation(:criticism, :fearful), do: "I knew I would disappoint them"

  def interpret_situation(:more_time_together, :secure), do: "That sounds nice"
  def interpret_situation(:more_time_together, :anxious), do: "Finally! I've been waiting for this"
  def interpret_situation(:more_time_together, :avoidant), do: "That's a bit much, I need my space"

  def interpret_situation(:more_time_together, :fearful),
    do: "I want to, but what if it goes wrong?"

  def interpret_situation(:distance, :secure), do: "We both need some space sometimes"
  def interpret_situation(:distance, :anxious), do: "They're pulling away, what did I do?"
  def interpret_situation(:distance, :avoidant), do: "Good, I prefer having my independence"
  def interpret_situation(:distance, :fearful), do: "This is what I expected, people always leave"

  def interpret_situation(:affection, :secure), do: "That's sweet, I appreciate them"
  def interpret_situation(:affection, :anxious), do: "Do they really mean it? I hope this lasts"
  def interpret_situation(:affection, :avoidant), do: "That's a bit intense, slow down"
  def interpret_situation(:affection, :fearful), do: "I want to believe them, but I'm scared"

  def interpret_situation(:conflict, :secure), do: "We can work through this together"
  def interpret_situation(:conflict, :anxious), do: "This is terrible, what if they leave me?"
  def interpret_situation(:conflict, :avoidant), do: "I'm done with this drama, I need out"
  def interpret_situation(:conflict, :fearful), do: "I knew something would go wrong"

  def interpret_situation(_, style), do: default_interpretation(style)

  @doc """
  Returns the attachment style's general orientation toward relationships.
  """
  @spec describe_style(attachment_style()) :: String.t()
  def describe_style(:secure) do
    "Comfortable with intimacy and independence; trusts others"
  end

  def describe_style(:anxious) do
    "Craves closeness but fears abandonment; hypervigilant to rejection"
  end

  def describe_style(:avoidant) do
    "Values independence; uncomfortable with too much closeness"
  end

  def describe_style(:fearful) do
    "Desires connection but expects rejection; conflicted about intimacy"
  end

  @doc """
  Returns how likely this attachment style is to initiate social contact.
  Value from 0.0 (never initiates) to 1.0 (always initiates).
  """
  @spec social_initiative(attachment_style()) :: float()
  def social_initiative(:secure), do: 0.7
  def social_initiative(:anxious), do: 0.8
  def social_initiative(:avoidant), do: 0.3
  def social_initiative(:fearful), do: 0.4

  @doc """
  Returns how this attachment style reacts to relationship uncertainty.
  """
  @spec uncertainty_response(attachment_style()) :: atom()
  def uncertainty_response(:secure), do: :calm_inquiry
  def uncertainty_response(:anxious), do: :protest_behavior
  def uncertainty_response(:avoidant), do: :deactivation
  def uncertainty_response(:fearful), do: :freeze_response

  # === Private Functions ===

  defp social_stimulus?(%{type: type}) when type in @social_types, do: true
  defp social_stimulus?(_), do: false

  defp apply_bias(stimulus, style) do
    original_valence = Map.get(stimulus, :valence, 0.0)
    original_intensity = Map.get(stimulus, :intensity, 0.5)
    original_threat = Map.get(stimulus, :threat_level, 0.0)

    # Apply biases based on attachment style
    new_valence =
      (original_valence + Map.get(@valence_bias, style, 0.0))
      |> clamp(-1.0, 1.0)

    new_intensity =
      (original_intensity + Map.get(@intensity_bias, style, 0.0))
      |> clamp(0.0, 1.0)

    new_threat =
      (original_threat + Map.get(@threat_bias, style, 0.0))
      |> clamp(0.0, 1.0)

    # Add interpretation based on context
    situation = infer_situation(stimulus)
    interpretation_text = interpret_situation(situation, style)

    stimulus
    |> Map.put(:valence, new_valence)
    |> Map.put(:intensity, new_intensity)
    |> Map.put(:threat_level, new_threat)
    |> Map.put(:attachment_interpretation, interpretation_text)
    |> Map.put(:attachment_style, style)
  end

  defp infer_situation(stimulus) do
    cond do
      # Positive social interaction
      Map.get(stimulus, :social_context) == :conversation and
          Map.get(stimulus, :valence, 0.0) > 0.2 ->
        :affection

      # Active conversation
      Map.get(stimulus, :social_context) == :conversation ->
        :more_time_together

      # Just ambient presence (owner online but not interacting)
      Map.get(stimulus, :type) == :social_ambient and
          Map.get(stimulus, :intensity, 0.5) < 0.4 ->
        :distance

      # Owner present but not engaging much
      Map.get(stimulus, :type) == :social_ambient ->
        :no_response

      true ->
        :ambient
    end
  end

  defp create_interpretation(stimulus, style, bias_applied) do
    %{
      style: style,
      bias_applied: bias_applied,
      original_valence: Map.get(stimulus, :valence, 0.0),
      interpretation:
        if(bias_applied,
          do: interpret_situation(infer_situation(stimulus), style),
          else: nil
        )
    }
  end

  defp default_interpretation(:secure), do: "Things seem fine"
  defp default_interpretation(:anxious), do: "I hope everything is okay"
  defp default_interpretation(:avoidant), do: "Whatever happens, happens"
  defp default_interpretation(:fearful), do: "I should be careful"

  defp clamp(value, min_val, max_val), do: value |> max(min_val) |> min(max_val)
end
