defmodule Viva.Avatars.Systems.SocialDynamics do
  @moduledoc """
  Emergent Social Dynamics Engine with Theory of Mind.

  Implements sophisticated social cognition that enables avatars to:
  1. Build mental models of other avatars (beliefs, intentions, emotions)
  2. Predict others' behaviors and reactions
  3. Experience empathy through emotional mirroring
  4. Learn from observing social outcomes
  5. Navigate complex relationship dynamics

  This is the foundation for genuine social intelligence - not just
  responding to others, but truly understanding them as minded beings.

  Theory of Mind Levels:
  - Level 0: No model (treat others as objects)
  - Level 1: First-order beliefs ("I think X feels sad")
  - Level 2: Second-order beliefs ("I think X thinks I'm angry")
  - Level 3: Recursive beliefs ("I think X thinks I think...")

  Most human social interaction operates at Level 2. This system
  implements up to Level 2 with emergent Level 3 in complex situations.
  """

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.Personality

  @type mental_model :: %{
          avatar_id: String.t(),
          name: String.t(),
          # What we believe about their current state
          believed_emotion: atom(),
          believed_arousal: float(),
          believed_intentions: list(atom()),
          # What we believe they think about us
          believed_perception_of_us: atom(),
          # Trust and relationship metrics
          trust_level: float(),
          predictability: float(),
          affection: float(),
          # History
          interaction_count: integer(),
          last_interaction: DateTime.t() | nil,
          memorable_moments: list(map()),
          # Confidence in our model
          model_confidence: float()
        }

  @type social_state :: %{
          mental_models: %{String.t() => mental_model()},
          empathy_resonance: float(),
          social_anxiety: float(),
          belonging_sense: float(),
          social_predictions: list(map()),
          theory_of_mind_level: integer(),
          last_social_update: DateTime.t() | nil
        }

  @type social_event :: %{
          type: atom(),
          other_avatar_id: String.t(),
          content: map(),
          timestamp: DateTime.t()
        }

  @empathy_decay 0.02
  @model_decay 0.01
  @max_memorable_moments 5
  @prediction_horizon 3

  @doc """
  Initialize social dynamics state.
  """
  @spec init() :: social_state()
  def init do
    %{
      mental_models: %{},
      empathy_resonance: 0.0,
      social_anxiety: 0.3,
      belonging_sense: 0.3,
      social_predictions: [],
      theory_of_mind_level: 1,
      last_social_update: nil
    }
  end

  @doc """
  Process a social tick - updates mental models, empathy, and predictions.

  Returns {updated_social_state, social_insights}
  """
  @spec tick(
          social_state(),
          social_events :: list(social_event()),
          EmotionalState.t(),
          ConsciousnessState.t(),
          Personality.t()
        ) :: {social_state(), list(map())}
  def tick(social, events, emotional, consciousness, personality) do
    social
    |> decay_empathy()
    |> decay_model_confidence()
    |> process_social_events(events, emotional, personality)
    |> update_belonging_sense(emotional, consciousness)
    |> update_social_anxiety(emotional, personality)
    |> maybe_advance_tom_level(consciousness)
    |> generate_predictions(personality)
    |> generate_insights(consciousness, personality)
  end

  @doc """
  Get or create a mental model for another avatar.
  """
  @spec get_mental_model(social_state(), String.t(), String.t()) :: mental_model()
  def get_mental_model(social, avatar_id, name \\ "unknown") do
    Map.get(social.mental_models, avatar_id, create_initial_model(avatar_id, name))
  end

  @doc """
  Simulate empathic response to another's emotional state.

  Returns how much our emotions should be influenced by theirs.
  """
  @spec empathic_response(social_state(), mental_model(), Personality.t()) :: float()
  def empathic_response(social, other_model, personality) do
    # Base empathy from agreeableness
    base_empathy = personality.agreeableness * 0.5

    # Relationship depth increases empathy
    relationship_factor = (other_model.trust_level + other_model.affection) / 2

    # Current empathy resonance
    resonance_factor = social.empathy_resonance

    # Calculate total empathic influence
    min(1.0, base_empathy + relationship_factor * 0.3 + resonance_factor * 0.2)
  end

  @doc """
  Predict how another avatar will react to a potential action.

  Uses mental model to simulate their likely response.
  """
  @spec predict_reaction(social_state(), String.t(), atom(), Personality.t()) ::
          {:ok, map()} | {:error, :no_model}
  def predict_reaction(social, other_avatar_id, action_type, personality) do
    case Map.get(social.mental_models, other_avatar_id) do
      nil ->
        {:error, :no_model}

      model ->
        prediction = simulate_reaction(model, action_type, personality)
        {:ok, prediction}
    end
  end

  @doc """
  Generate a narrative about our social understanding.
  """
  @spec social_narrative(social_state(), Personality.t()) :: String.t() | nil
  def social_narrative(%{mental_models: models}, _) when map_size(models) == 0 do
    nil
  end

  def social_narrative(social, personality) do
    model_count = map_size(social.mental_models)

    belonging_text =
      cond do
        social.belonging_sense > 0.7 -> "Me sinto profundamente conectado aos outros"
        social.belonging_sense > 0.4 -> "Tenho algumas conexões significativas"
        true -> "Sinto uma certa distância social"
      end

    anxiety_text =
      if social.social_anxiety > 0.6 and personality.neuroticism > 0.5 do
        ", embora às vezes me preocupe com o que pensam de mim"
      else
        ""
      end

    tom_text =
      cond do
        social.theory_of_mind_level >= 2 ->
          ". Consigo perceber não só o que sentem, mas também o que pensam sobre mim"

        social.theory_of_mind_level == 1 ->
          ". Percebo os sentimentos alheios"

        true ->
          ""
      end

    "#{belonging_text}#{anxiety_text}. Conheço #{model_count} #{if model_count == 1, do: "pessoa", else: "pessoas"}#{tom_text}."
  end

  # === Private Functions ===

  defp decay_empathy(social) do
    new_empathy = max(0.0, social.empathy_resonance - @empathy_decay)
    %{social | empathy_resonance: new_empathy}
  end

  defp decay_model_confidence(social) do
    updated_models =
      Map.new(social.mental_models, fn {id, model} ->
        new_confidence = max(0.2, model.model_confidence - @model_decay)
        {id, %{model | model_confidence: new_confidence}}
      end)

    %{social | mental_models: updated_models}
  end

  defp process_social_events(social, [], _, _), do: social

  defp process_social_events(social, [event | rest], emotional, personality) do
    social
    |> process_single_event(event, emotional, personality)
    |> process_social_events(rest, emotional, personality)
  end

  defp process_single_event(social, event, emotional, personality) do
    other_id = event.other_avatar_id
    other_name = event.content[:name] || "unknown"

    model = get_mental_model(social, other_id, other_name)
    updated_model = update_model_from_event(model, event, emotional, personality)

    # Update empathy resonance based on event
    new_empathy = update_empathy_from_event(social.empathy_resonance, event, personality)

    %{
      social
      | mental_models: Map.put(social.mental_models, other_id, updated_model),
        empathy_resonance: new_empathy,
        last_social_update: DateTime.utc_now()
    }
  end

  defp update_model_from_event(model, event, emotional, personality) do
    case event.type do
      :message_received ->
        update_from_message(model, event, emotional, personality)

      :emotion_observed ->
        update_from_emotion_observation(model, event, personality)

      :action_observed ->
        update_from_action_observation(model, event)

      :trust_event ->
        update_from_trust_event(model, event)

      :conflict ->
        update_from_conflict(model, event, personality)

      _ ->
        model
    end
  end

  defp update_from_message(model, event, _, personality) do
    content = event.content
    sentiment = content[:sentiment] || :neutral
    topic = content[:topic]

    # Infer their emotional state from message
    believed_emotion = infer_emotion_from_sentiment(sentiment)

    # Update our model of what they think about us based on how they treat us
    perception =
      case sentiment do
        :positive -> :liked
        :negative -> :disliked
        :neutral -> :neutral
        _ -> model.believed_perception_of_us
      end

    # Create memorable moment if significant
    memorable =
      if content[:significance] && content[:significance] > 0.5 do
        moment = %{
          type: :conversation,
          topic: topic,
          sentiment: sentiment,
          timestamp: event.timestamp
        }

        Enum.take([moment | model.memorable_moments], @max_memorable_moments)
      else
        model.memorable_moments
      end

    # Adjust trust based on consistency
    trust_delta =
      if personality.agreeableness > 0.6 do
        0.02
      else
        0.01
      end

    %{
      model
      | believed_emotion: believed_emotion,
        believed_perception_of_us: perception,
        interaction_count: model.interaction_count + 1,
        last_interaction: event.timestamp,
        memorable_moments: memorable,
        trust_level: min(1.0, model.trust_level + trust_delta),
        model_confidence: min(1.0, model.model_confidence + 0.05)
    }
  end

  defp update_from_emotion_observation(model, event, personality) do
    observed_emotion = event.content[:emotion] || :neutral
    observed_arousal = event.content[:arousal] || 0.5

    # High agreeableness leads to stronger empathic modeling
    confidence_gain = if personality.agreeableness > 0.6, do: 0.08, else: 0.04

    %{
      model
      | believed_emotion: observed_emotion,
        believed_arousal: observed_arousal,
        model_confidence: min(1.0, model.model_confidence + confidence_gain)
    }
  end

  defp update_from_action_observation(model, event) do
    action = event.content[:action]
    target = event.content[:target]

    intentions = update_believed_intentions(model.believed_intentions, action)
    predictability = update_predictability(model.predictability, action, model.believed_intentions)
    perception = update_perception_of_us(model.believed_perception_of_us, action, target)

    %{
      model
      | believed_intentions: intentions,
        predictability: predictability,
        believed_perception_of_us: perception,
        model_confidence: min(1.0, model.model_confidence + 0.03)
    }
  end

  defp update_believed_intentions(current_intentions, action) do
    case infer_intention_from_action(action) do
      nil ->
        current_intentions

      intention ->
        [intention | current_intentions]
        |> Enum.uniq()
        |> Enum.take(5)
    end
  end

  defp infer_intention_from_action(:helping), do: :prosocial
  defp infer_intention_from_action(:sharing), do: :generous
  defp infer_intention_from_action(:attacking), do: :hostile
  defp infer_intention_from_action(:avoiding), do: :fearful
  defp infer_intention_from_action(:approaching), do: :friendly
  defp infer_intention_from_action(_), do: nil

  defp update_predictability(current, action, believed_intentions) do
    if action in believed_intentions do
      min(1.0, current + 0.05)
    else
      max(0.0, current - 0.03)
    end
  end

  defp update_perception_of_us(current_perception, _, target) when target != :self do
    current_perception
  end

  defp update_perception_of_us(current_perception, action, :self) do
    action_to_perception(action, current_perception)
  end

  defp action_to_perception(:helping, _), do: :valued
  defp action_to_perception(:attacking, _), do: :threatened
  defp action_to_perception(:approaching, _), do: :liked
  defp action_to_perception(:avoiding, _), do: :rejected
  defp action_to_perception(_, current), do: current

  defp update_from_trust_event(model, event) do
    trust_delta = event.content[:delta] || 0.0
    affection_delta = event.content[:affection_delta] || 0.0

    memorable =
      if abs(trust_delta) > 0.1 do
        moment = %{
          type: :trust_change,
          delta: trust_delta,
          reason: event.content[:reason],
          timestamp: event.timestamp
        }

        Enum.take([moment | model.memorable_moments], @max_memorable_moments)
      else
        model.memorable_moments
      end

    %{
      model
      | trust_level: clamp(model.trust_level + trust_delta, 0.0, 1.0),
        affection: clamp(model.affection + affection_delta, -1.0, 1.0),
        memorable_moments: memorable
    }
  end

  defp update_from_conflict(model, event, personality) do
    severity = event.content[:severity] || 0.5

    # Neurotic personalities are more affected by conflict
    impact = if personality.neuroticism > 0.6, do: severity * 1.3, else: severity

    # Record memorable moment
    memorable =
      if severity > 0.3 do
        moment = %{
          type: :conflict,
          severity: severity,
          topic: event.content[:topic],
          timestamp: event.timestamp
        }

        Enum.take([moment | model.memorable_moments], @max_memorable_moments)
      else
        model.memorable_moments
      end

    %{
      model
      | trust_level: max(0.0, model.trust_level - impact * 0.2),
        affection: max(-1.0, model.affection - impact * 0.15),
        believed_perception_of_us: :conflicted,
        memorable_moments: memorable
    }
  end

  defp update_empathy_from_event(current_empathy, event, personality) do
    base_gain = personality.agreeableness * 0.1

    gain =
      case event.type do
        :emotion_observed -> base_gain * 1.5
        :message_received -> base_gain
        :conflict -> base_gain * 0.5
        _ -> base_gain * 0.3
      end

    min(1.0, current_empathy + gain)
  end

  defp update_belonging_sense(social, emotional, consciousness) do
    # Count meaningful relationships
    meaningful_count =
      Enum.count(social.mental_models, fn {_, model} ->
        model.trust_level > 0.5 and model.affection > 0.3
      end)

    # Base belonging from relationships
    base_belonging = min(1.0, meaningful_count * 0.2)

    # Emotional state affects belonging
    emotional_factor =
      if emotional.pleasure > 0 do
        0.1
      else
        -0.05
      end

    # Recent social interaction boosts belonging
    recency_factor =
      if social.last_social_update do
        diff = DateTime.diff(DateTime.utc_now(), social.last_social_update, :minute)

        if diff < 10 do
          0.1
        else
          0.0
        end
      else
        0.0
      end

    # Meta-awareness of loneliness affects belonging
    meta_factor =
      if consciousness.meta_awareness > 0.6 and meaningful_count == 0 do
        -0.1
      else
        0.0
      end

    new_belonging =
      clamp(
        social.belonging_sense * 0.9 + base_belonging * 0.1 + emotional_factor + recency_factor +
          meta_factor,
        0.0,
        1.0
      )

    %{social | belonging_sense: new_belonging}
  end

  defp update_social_anxiety(social, emotional, personality) do
    # Base anxiety from neuroticism
    base_anxiety = personality.neuroticism * 0.4

    # Arousal increases social anxiety
    arousal_factor = emotional.arousal * 0.2

    # Low dominance increases anxiety
    dominance_factor = (1.0 - emotional.dominance) * 0.2

    # Many unpredictable relationships increase anxiety
    unpredictable_count =
      Enum.count(social.mental_models, fn {_, model} ->
        model.predictability < 0.4
      end)

    unpredictability_factor = min(0.2, unpredictable_count * 0.05)

    # High belonging reduces anxiety
    belonging_reduction = social.belonging_sense * -0.2

    new_anxiety =
      clamp(
        base_anxiety + arousal_factor + dominance_factor + unpredictability_factor +
          belonging_reduction,
        0.0,
        1.0
      )

    %{social | social_anxiety: new_anxiety}
  end

  defp maybe_advance_tom_level(social, consciousness) do
    # Theory of Mind level advances with meta-awareness and relationship depth
    current_level = social.theory_of_mind_level

    # Calculate conditions for advancement
    has_deep_relationships =
      Enum.any?(social.mental_models, fn {_, model} ->
        model.trust_level > 0.7 and model.interaction_count > 5
      end)

    high_meta = consciousness.meta_awareness > 0.7
    enough_models = map_size(social.mental_models) >= 2

    new_level =
      cond do
        current_level == 1 and has_deep_relationships and high_meta ->
          2

        current_level == 2 and has_deep_relationships and high_meta and enough_models ->
          # Level 3 is rare and requires exceptional conditions
          if :rand.uniform() < 0.01, do: 3, else: 2

        true ->
          current_level
      end

    %{social | theory_of_mind_level: new_level}
  end

  defp generate_predictions(social, personality) do
    # Generate predictions about likely social outcomes
    predictions =
      social.mental_models
      |> Enum.flat_map(fn {id, model} ->
        generate_predictions_for_model(id, model, personality)
      end)
      |> Enum.take(@prediction_horizon)

    %{social | social_predictions: predictions}
  end

  defp generate_predictions_for_model(avatar_id, model, personality) do
    []
    |> maybe_add_action_prediction(avatar_id, model)
    |> maybe_add_emotion_prediction(avatar_id, model)
    |> maybe_add_relationship_prediction(avatar_id, model, personality)
  end

  defp maybe_add_action_prediction(predictions, avatar_id, model) do
    if model.model_confidence > 0.5 do
      action_prediction = %{
        type: :action_prediction,
        avatar_id: avatar_id,
        predicted_action: predict_action_from_model(model),
        confidence: model.model_confidence * model.predictability
      }

      [action_prediction | predictions]
    else
      predictions
    end
  end

  defp maybe_add_emotion_prediction(predictions, avatar_id, model) do
    if model.believed_arousal > 0.6 do
      emotion_prediction = %{
        type: :emotion_prediction,
        avatar_id: avatar_id,
        predicted_change: if(model.believed_arousal > 0.8, do: :calming, else: :stable),
        confidence: model.model_confidence * 0.7
      }

      [emotion_prediction | predictions]
    else
      predictions
    end
  end

  defp maybe_add_relationship_prediction(predictions, avatar_id, model, personality) do
    if personality.openness > 0.6 and model.trust_level > 0.4 do
      relationship_prediction = %{
        type: :relationship_prediction,
        avatar_id: avatar_id,
        predicted_direction: :deepening,
        confidence: personality.agreeableness * 0.5
      }

      [relationship_prediction | predictions]
    else
      predictions
    end
  end

  defp generate_insights(social, consciousness, personality) do
    insights =
      []
      |> maybe_add_belonging_insight(social, consciousness)
      |> maybe_add_relationship_insight(social, personality)
      |> maybe_add_pattern_insight(social, personality)

    {social, insights}
  end

  defp maybe_add_belonging_insight(insights, social, consciousness) do
    if social.belonging_sense < 0.3 and consciousness.meta_awareness > 0.5 do
      insight = %{
        type: :social_insight,
        content: :loneliness_awareness,
        description: "Percebo que me sinto desconectado dos outros"
      }

      [insight | insights]
    else
      insights
    end
  end

  defp maybe_add_relationship_insight(insights, social, personality) do
    case find_significant_relationship(social.mental_models) do
      nil ->
        insights

      {_, model} ->
        insight = %{
          type: :relationship_insight,
          content: categorize_relationship(model),
          name: model.name,
          description: describe_relationship(model, personality)
        }

        [insight | insights]
    end
  end

  defp maybe_add_pattern_insight(insights, social, personality) do
    if map_size(social.mental_models) >= 3 and personality.openness > 0.5 do
      pattern = detect_social_pattern(social.mental_models)

      if pattern do
        [pattern | insights]
      else
        insights
      end
    else
      insights
    end
  end

  # === Helper Functions ===

  defp create_initial_model(avatar_id, name) do
    %{
      avatar_id: avatar_id,
      name: name,
      believed_emotion: :unknown,
      believed_arousal: 0.5,
      believed_intentions: [],
      believed_perception_of_us: :unknown,
      trust_level: 0.3,
      predictability: 0.3,
      affection: 0.0,
      interaction_count: 0,
      last_interaction: nil,
      memorable_moments: [],
      model_confidence: 0.2
    }
  end

  defp infer_emotion_from_sentiment(:positive), do: :happy
  defp infer_emotion_from_sentiment(:negative), do: :upset
  defp infer_emotion_from_sentiment(:neutral), do: :calm
  defp infer_emotion_from_sentiment(_), do: :unknown

  defp predict_action_from_model(model) do
    cond do
      model.affection > 0.6 and model.trust_level > 0.6 -> :approach
      model.affection < -0.3 -> :avoid
      :prosocial in model.believed_intentions -> :helping
      :hostile in model.believed_intentions -> :conflict
      true -> :neutral
    end
  end

  defp simulate_reaction(model, action_type, personality) do
    # Base reaction on model's perceived state and our personality
    base_reaction =
      case {action_type, model.believed_perception_of_us} do
        {:approach, :liked} -> :welcoming
        {:approach, :disliked} -> :wary
        {:gift, _} -> :grateful
        {:conflict, _} -> :defensive
        {:help, _} -> :appreciative
        _ -> :neutral
      end

    # Adjust confidence based on model quality
    confidence = model.model_confidence * model.predictability

    # Personality affects prediction
    adjusted_confidence =
      if personality.openness > 0.6 do
        confidence * 1.1
      else
        confidence
      end

    %{
      predicted_reaction: base_reaction,
      confidence: min(1.0, adjusted_confidence),
      reasoning: "Based on #{model.interaction_count} interactions"
    }
  end

  defp find_significant_relationship(models) when map_size(models) == 0, do: nil

  defp find_significant_relationship(models) do
    Enum.max_by(
      models,
      fn {_, model} ->
        # Score by significance
        model.trust_level * 0.3 + abs(model.affection) * 0.3 + model.interaction_count * 0.01
      end,
      fn -> nil end
    )
  end

  defp categorize_relationship(model) do
    cond do
      model.trust_level > 0.7 and model.affection > 0.7 -> :close_friend
      model.trust_level > 0.5 and model.affection > 0.3 -> :friend
      model.trust_level > 0.5 and model.affection < 0 -> :complicated
      model.affection < -0.5 -> :adversary
      model.interaction_count > 10 -> :acquaintance
      true -> :stranger
    end
  end

  defp describe_relationship(model, personality) do
    base_description = relationship_base_description(model)
    add_memorable_moment_text(base_description, model, personality)
  end

  defp relationship_base_description(model) do
    category = categorize_relationship(model)
    category_to_description(category, model.name)
  end

  defp category_to_description(:close_friend, name),
    do: "#{name} é alguém em quem confio profundamente"

  defp category_to_description(:friend, name), do: "Tenho uma conexão positiva com #{name}"
  defp category_to_description(:complicated, name), do: "Minha relação com #{name} é complexa"
  defp category_to_description(:adversary, name), do: "Tenho tensão com #{name}"
  defp category_to_description(:acquaintance, name), do: "Conheço #{name} há algum tempo"
  defp category_to_description(:stranger, name), do: "Ainda estou conhecendo #{name}"

  defp add_memorable_moment_text(description, model, personality) do
    if personality.openness > 0.6 and model.memorable_moments != [] do
      moment = hd(model.memorable_moments)
      description <> moment_type_text(moment.type)
    else
      description
    end
  end

  defp moment_type_text(:conversation), do: ", e lembro de conversas significativas"
  defp moment_type_text(:trust_change), do: ", e passamos por momentos importantes"
  defp moment_type_text(:conflict), do: ", apesar de desentendimentos passados"
  defp moment_type_text(_), do: ""

  defp detect_social_pattern(models) do
    trust_levels = Enum.map(models, fn {_, m} -> m.trust_level end)
    avg_trust = Enum.sum(trust_levels) / length(trust_levels)

    affections = Enum.map(models, fn {_, m} -> m.affection end)
    avg_affection = Enum.sum(affections) / length(affections)

    cond do
      avg_trust > 0.6 and avg_affection > 0.5 ->
        %{
          type: :social_pattern,
          pattern: :secure_network,
          description: "Percebo que construí relacionamentos seguros"
        }

      avg_trust < 0.3 ->
        %{
          type: :social_pattern,
          pattern: :trust_issues,
          description: "Noto que tenho dificuldade em confiar nos outros"
        }

      Enum.any?(affections, &(&1 < -0.5)) and Enum.any?(affections, &(&1 > 0.5)) ->
        %{
          type: :social_pattern,
          pattern: :polarized_relationships,
          description: "Minhas relações são intensas - amo ou me afasto"
        }

      true ->
        nil
    end
  end

  defp clamp(value, min_val, max_val) do
    value
    |> max(min_val)
    |> min(max_val)
  end
end
