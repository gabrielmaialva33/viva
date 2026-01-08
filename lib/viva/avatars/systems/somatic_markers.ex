defmodule Viva.Avatars.Systems.SomaticMarkers do
  @moduledoc """
  Implements Damasio's somatic marker hypothesis for avatars.

  The body "remembers" emotionally intense experiences and creates gut feelings
  that influence future decisions. When a similar stimulus appears:
  - Negative markers create "body warnings" (avoid)
  - Positive markers create "body attractions" (approach)

  These signals happen BEFORE conscious reasoning and bias decisions.

  ## Learning
  Markers are formed when emotional intensity exceeds threshold:
  - High pleasure + high arousal = positive marker
  - Low pleasure + high arousal = negative marker
  - Low arousal experiences don't form markers

  ## Recall
  When similar stimuli appear, matching markers activate and
  create a somatic bias that influences desire/decision-making.

  ## Decay
  Markers lose strength over time if not reactivated.
  Strong repeated experiences strengthen markers.
  """

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.SomaticMarkersState

  # How much markers decay per tick when not activated
  @decay_rate 0.001

  # Minimum strength to keep a marker
  @min_strength 0.1

  # Maximum strength a marker can reach
  @max_strength 1.0

  # Strength increase when reactivated
  @reinforcement_rate 0.1

  @type stimulus :: map()
  @type somatic_bias :: %{
          bias: float(),
          signal: String.t() | nil,
          markers_activated: non_neg_integer()
        }

  @type warning_level ::
          :go_for_it
          | :proceed_cautiously
          | :strong_caution
          | :avoid
          | :neutral

  @type decision_confidence_result :: {float(), warning_level()}

  @doc """
  Recalls somatic markers based on current stimulus.
  Returns updated state and somatic bias for decision-making.
  """
  @spec recall(SomaticMarkersState.t(), stimulus()) :: {SomaticMarkersState.t(), somatic_bias()}
  def recall(somatic, stimulus) do
    # Find matching markers for this stimulus
    {matching_markers, _} = find_matching_markers(somatic, stimulus)

    if Enum.empty?(matching_markers) do
      {somatic, %{bias: 0.0, signal: nil, markers_activated: 0}}
    else
      # Calculate combined bias
      combined_bias = calculate_combined_bias(matching_markers)
      signal = generate_body_signal(combined_bias)

      # Update marker activation times and reinforce
      updated_somatic =
        somatic
        |> reinforce_markers(matching_markers)
        |> Map.put(:current_bias, combined_bias)
        |> Map.put(:body_signal, signal)
        |> Map.put(:last_marker_activation, DateTime.utc_now(:second))

      {updated_somatic,
       %{
         bias: combined_bias,
         signal: signal,
         markers_activated: length(matching_markers)
       }}
    end
  end

  @doc """
  Potentially learns a new marker from an intense experience.
  Called at the end of tick if emotional intensity is high.
  """
  @spec maybe_learn(SomaticMarkersState.t(), stimulus(), BioState.t(), EmotionalState.t()) ::
          SomaticMarkersState.t()
  def maybe_learn(somatic, stimulus, bio, emotional) do
    intensity = calculate_intensity(emotional)
    threshold = somatic.learning_threshold

    if intensity >= threshold do
      learn_marker(somatic, stimulus, bio, emotional, intensity)
    else
      # Decay existing markers slightly
      decay_markers(somatic)
    end
  end

  @doc """
  Describes the current somatic state in human-readable terms.
  """
  @spec describe(SomaticMarkersState.t()) :: String.t()
  def describe(%SomaticMarkersState{current_bias: bias, body_signal: signal})
      when bias > 0.3 do
    signal || "Body feels drawn toward this, a warm gut feeling"
  end

  def describe(%SomaticMarkersState{current_bias: bias, body_signal: signal})
      when bias < -0.3 do
    signal || "Body tenses up, a warning from within"
  end

  def describe(%SomaticMarkersState{markers_formed: 0}) do
    "No strong body memories yet"
  end

  def describe(%SomaticMarkersState{}) do
    "Body is neutral, no strong signals"
  end

  @doc """
  Returns true if body is signaling a warning (avoid).
  """
  @spec body_warning?(SomaticMarkersState.t()) :: boolean()
  def body_warning?(%SomaticMarkersState{current_bias: bias}), do: bias < -0.2

  @doc """
  Returns true if body is signaling attraction (approach).
  """
  @spec body_attraction?(SomaticMarkersState.t()) :: boolean()
  def body_attraction?(%SomaticMarkersState{current_bias: bias}), do: bias > 0.2

  # === Decision Confidence Functions ===

  @doc """
  Returns decision confidence based on somatic markers for a given action type.
  Looks up markers related to the action and returns a confidence score (0.0-1.0)
  and a warning level.

  ## Action Types
  - :social - Social interactions
  - :explore - New/novel activities
  - :rest - Resting/recovery
  - :express - Self-expression
  - :intimate - Intimate/romantic actions

  ## Returns
  - {confidence_score, warning_level}
  - confidence: 0.0 (avoid) to 1.0 (go for it)
  - warning: :go_for_it | :proceed_cautiously | :strong_caution | :avoid | :neutral
  """
  @spec decision_confidence(SomaticMarkersState.t(), atom()) :: {float(), warning_level()}
  def decision_confidence(state, action_type) do
    # Find relevant markers for this action type
    relevant_valence = find_relevant_valence(state, action_type)

    case relevant_valence do
      nil ->
        # No marker = neutral confidence
        {0.5, :neutral}

      v when v > 0.7 ->
        {1.0, :go_for_it}

      v when v > 0.4 ->
        {0.7, :proceed_cautiously}

      v when v > 0.2 ->
        {0.4, :strong_caution}

      _ ->
        {0.1, :avoid}
    end
  end

  @doc """
  Applies decision confidence to modify a desire/action's viability.
  Returns adjusted desire weight based on somatic memory.
  """
  @spec apply_confidence_to_desire(SomaticMarkersState.t(), atom(), float()) :: float()
  def apply_confidence_to_desire(state, action_type, base_weight) do
    {confidence, _} = decision_confidence(state, action_type)

    # Blend base weight with somatic confidence
    # Confidence < 0.5 reduces weight, > 0.5 increases it
    adjustment = (confidence - 0.5) * 0.4
    clamp(base_weight + adjustment, 0.0, 1.0)
  end

  @doc """
  Returns a narrative explanation of why the body feels a certain way about an action.
  """
  @spec explain_confidence(SomaticMarkersState.t(), atom()) :: String.t()
  def explain_confidence(state, action_type) do
    {_, warning} = decision_confidence(state, action_type)

    case warning do
      :go_for_it ->
        "My body remembers good things about this. There's warmth, a pull forward."

      :proceed_cautiously ->
        "Something feels familiar here. Not quite sure, but leaning toward yes."

      :neutral ->
        "No strong feeling either way. This is new territory."

      :strong_caution ->
        "There's tension building. Something about this feels... risky."

      :avoid ->
        "My stomach tightens. Past experiences scream no. Don't go there."
    end
  end

  defp find_relevant_valence(state, action_type) do
    # Map action types to marker categories
    result =
      case action_type do
        :social ->
          # Check social and context markers
          find_avg_valence([
            state.activity_markers[:social],
            state.activity_markers[:social_ambient]
          ])

        :explore ->
          # Check for novelty-related markers
          state.context_markers["new"] || state.context_markers["exploration"]

        :rest ->
          # Rest is usually neutral unless specifically marked
          state.activity_markers[:rest]

        :express ->
          # Expression markers
          state.activity_markers[:express] || state.context_markers["self_expression"]

        :intimate ->
          # Check crush/intimate context
          find_avg_valence([
            state.social_markers["crush"],
            state.context_markers["intimate"]
          ])

        _ ->
          # Default to current bias
          if state.current_bias != 0.0 do
            %{valence: state.current_bias, strength: 0.5}
          else
            nil
          end
      end

    extract_valence(result)
  end

  defp find_avg_valence(markers) do
    valid_markers = Enum.filter(markers, & &1)

    if Enum.empty?(valid_markers) do
      nil
    else
      avg_valence =
        Enum.reduce(valid_markers, 0.0, fn m, acc -> acc + m.valence end) / length(valid_markers)

      avg_strength =
        Enum.reduce(valid_markers, 0.0, fn m, acc -> acc + m.strength end) / length(valid_markers)

      %{valence: avg_valence, strength: avg_strength}
    end
  end

  defp extract_valence(nil), do: nil
  defp extract_valence(%{valence: v, strength: s}), do: v * s
  defp extract_valence(_), do: nil

  # === Private Functions ===

  defp calculate_intensity(%EmotionalState{pleasure: p, arousal: a}) do
    # Intensity based on arousal and absolute pleasure
    arousal_component = abs(a)
    pleasure_component = abs(p)
    min(arousal_component * 0.6 + pleasure_component * 0.4, 1.0)
  end

  defp find_matching_markers(somatic, stimulus) do
    matching = []

    # Check social markers
    source = Map.get(stimulus, :source, "")

    matching_social =
      case Map.get(somatic.social_markers, source) do
        nil -> matching
        marker -> [{:social, source, marker} | matching]
      end

    # Check activity markers
    activity = Map.get(stimulus, :type)

    matching_activity =
      case Map.get(somatic.activity_markers, activity) do
        nil -> matching_social
        marker -> [{:activity, activity, marker} | matching_social]
      end

    # Check context markers (social context)
    context = Map.get(stimulus, :social_context)

    matching_context =
      if context do
        case Map.get(somatic.context_markers, to_string(context)) do
          nil -> matching_activity
          marker -> [{:context, context, marker} | matching_activity]
        end
      else
        matching_activity
      end

    total_bias =
      Enum.reduce(matching_context, 0.0, fn {_, _, marker}, acc ->
        acc + marker.valence * marker.strength
      end)

    {matching_context, total_bias}
  end

  defp calculate_combined_bias(markers) do
    if Enum.empty?(markers) do
      0.0
    else
      total =
        Enum.reduce(markers, 0.0, fn {_, _, marker}, acc ->
          acc + marker.valence * marker.strength
        end)

      # Normalize by number of markers, but not too much
      markers
      |> length()
      |> :math.sqrt()
      |> then(&(total / &1))
      |> clamp(-1.0, 1.0)
    end
  end

  defp generate_body_signal(bias) when bias > 0.5 do
    "A warm feeling spreads through the chest, the body leans forward"
  end

  defp generate_body_signal(bias) when bias > 0.2 do
    "Subtle warmth, a gentle pull toward this"
  end

  defp generate_body_signal(bias) when bias < -0.5 do
    "Stomach tightens, shoulders tense, the body recoils"
  end

  defp generate_body_signal(bias) when bias < -0.2 do
    "A slight unease, something feels off"
  end

  defp generate_body_signal(_), do: nil

  defp reinforce_markers(somatic, matching_markers) do
    Enum.reduce(matching_markers, somatic, fn {category, key, marker}, acc ->
      updated_marker = %{
        marker
        | strength: min(@max_strength, marker.strength + @reinforcement_rate),
          last_activated: DateTime.utc_now(:second)
      }

      case category do
        :social ->
          %{acc | social_markers: Map.put(acc.social_markers, key, updated_marker)}

        :activity ->
          %{acc | activity_markers: Map.put(acc.activity_markers, key, updated_marker)}

        :context ->
          %{acc | context_markers: Map.put(acc.context_markers, to_string(key), updated_marker)}
      end
    end)
  end

  defp learn_marker(somatic, stimulus, _, emotional, intensity) do
    # Determine valence of marker (positive or negative experience)
    valence = calculate_marker_valence(emotional)

    # Determine initial strength based on intensity
    strength = intensity * 0.8

    marker = %{
      valence: valence,
      strength: strength,
      last_activated: DateTime.utc_now(:second),
      context: nil
    }

    # Store marker based on stimulus features
    updated_somatic =
      somatic
      |> store_social_marker(stimulus, marker)
      |> store_activity_marker(stimulus, marker)
      |> store_context_marker(stimulus, marker)

    # Update count if any marker was actually stored
    if updated_somatic != somatic do
      %{updated_somatic | markers_formed: updated_somatic.markers_formed + 1}
    else
      updated_somatic
    end
  end

  defp calculate_marker_valence(%EmotionalState{pleasure: p, arousal: a}) do
    # Valence is primarily based on pleasure, modulated by arousal direction
    base_valence = p
    arousal_direction = if a > 0, do: 1.0, else: -0.5

    clamp(base_valence * (0.7 + abs(a) * 0.3 * arousal_direction), -1.0, 1.0)
  end

  defp store_social_marker(somatic, stimulus, marker) do
    source = Map.get(stimulus, :source)

    if source && social_source?(source) do
      existing = Map.get(somatic.social_markers, source)

      updated_marker =
        if existing do
          merge_markers(existing, marker)
        else
          marker
        end

      %{somatic | social_markers: Map.put(somatic.social_markers, source, updated_marker)}
    else
      somatic
    end
  end

  defp store_activity_marker(somatic, stimulus, marker) do
    activity = Map.get(stimulus, :type)

    if activity && activity in [:social, :social_ambient] do
      existing = Map.get(somatic.activity_markers, activity)

      updated_marker =
        if existing do
          merge_markers(existing, marker)
        else
          marker
        end

      %{somatic | activity_markers: Map.put(somatic.activity_markers, activity, updated_marker)}
    else
      somatic
    end
  end

  defp store_context_marker(somatic, stimulus, marker) do
    context = Map.get(stimulus, :social_context)

    if context do
      key = to_string(context)
      existing = Map.get(somatic.context_markers, key)

      updated_marker =
        if existing do
          merge_markers(existing, marker)
        else
          marker
        end

      %{somatic | context_markers: Map.put(somatic.context_markers, key, updated_marker)}
    else
      somatic
    end
  end

  defp social_source?(source) when is_binary(source) do
    source in ["conversation_partner", "owner_presence", "friend", "crush"]
  end

  defp social_source?(_), do: false

  defp merge_markers(existing, new) do
    # Blend valences weighted by strength
    total_weight = existing.strength + new.strength

    blended_valence =
      (existing.valence * existing.strength + new.valence * new.strength) / total_weight

    %{
      valence: blended_valence,
      strength: min(@max_strength, existing.strength + new.strength * 0.5),
      last_activated: DateTime.utc_now(:second),
      context: new.context || existing.context
    }
  end

  defp decay_markers(somatic) do
    %{
      somatic
      | social_markers: decay_marker_map(somatic.social_markers),
        activity_markers: decay_marker_map(somatic.activity_markers),
        context_markers: decay_marker_map(somatic.context_markers)
    }
  end

  defp decay_marker_map(markers) do
    markers
    |> Enum.map(fn {key, marker} ->
      # Handle both atom and string keys (from DB/JSON)
      current_strength = Map.get(marker, :strength) || Map.get(marker, "strength") || 1.0
      new_strength = current_strength - @decay_rate
      updated_marker = Map.put(marker, :strength, new_strength) |> Map.put("strength", new_strength)
      {key, updated_marker}
    end)
    |> Enum.filter(fn {_, marker} ->
      strength = Map.get(marker, :strength) || Map.get(marker, "strength") || 0.0
      strength >= @min_strength
    end)
    |> Map.new()
  end

  defp clamp(value, min_val, max_val) do
    value
    |> max(min_val)
    |> min(max_val)
  end
end
