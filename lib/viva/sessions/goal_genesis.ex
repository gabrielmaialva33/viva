defmodule Viva.Sessions.GoalGenesis do
  @moduledoc """
  Spontaneous Goal Genesis System.

  Creates emergent goals that arise from internal states, not just
  reactive desires. This is a key marker of agency - the ability
  to form novel intentions that weren't externally triggered.

  Goal sources:
  1. Accumulated Experience Patterns - "I keep enjoying X, maybe I want more of it"
  2. Unmet Need Trajectories - "This need keeps recurring, time for a long-term solution"
  3. Personality Aspirations - "As someone high in openness, I want to explore"
  4. Social Modeling - "I see others doing X, maybe I should try"
  5. Curiosity Sparks - Random exploration drives from novelty seeking
  6. Self-Improvement - Metacognitive awareness of personal patterns

  Goals are different from desires:
  - Desires are immediate wants (I want to rest NOW)
  - Goals are aspirational intentions (I want to become more social)
  """

  alias Viva.Avatars.ConsciousnessState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.MotivationState
  alias Viva.Avatars.Personality

  @type goal :: %{
          id: String.t(),
          type: atom(),
          description: String.t(),
          source: atom(),
          intensity: float(),
          created_at: DateTime.t(),
          progress: float(),
          active: boolean()
        }

  @type genesis_state :: %{
          active_goals: list(goal()),
          goal_history: list(goal()),
          aspiration_seeds: list(map()),
          last_genesis: DateTime.t() | nil,
          genesis_cooldown: integer()
        }

  @max_active_goals 3
  @genesis_probability 0.08
  @goal_decay_rate 0.01

  @doc """
  Initialize goal genesis state.
  """
  @spec init() :: genesis_state()
  def init do
    %{
      active_goals: [],
      goal_history: [],
      aspiration_seeds: [],
      last_genesis: nil,
      genesis_cooldown: 0
    }
  end

  @doc """
  Process a tick - may spontaneously generate goals.

  Returns {updated_genesis_state, maybe_new_goal}
  """
  @spec tick(
          genesis_state(),
          ConsciousnessState.t(),
          EmotionalState.t(),
          MotivationState.t(),
          Personality.t(),
          tick_count :: integer()
        ) :: {genesis_state(), goal() | nil}
  def tick(genesis, consciousness, emotional, motivation, personality, tick_count) do
    genesis
    |> decay_goals()
    |> update_cooldown()
    |> collect_aspiration_seeds(consciousness, emotional, motivation)
    |> maybe_generate_goal(consciousness, emotional, motivation, personality, tick_count)
  end

  @doc """
  Get the most urgent active goal.
  """
  @spec urgent_goal(genesis_state()) :: goal() | nil
  def urgent_goal(%{active_goals: []}), do: nil

  def urgent_goal(%{active_goals: goals}) do
    Enum.max_by(goals, & &1.intensity, fn -> nil end)
  end

  @doc """
  Mark progress on a goal based on current activity.
  """
  @spec mark_progress(genesis_state(), atom(), float()) :: genesis_state()
  def mark_progress(genesis, goal_type, amount) do
    updated_goals =
      Enum.map(genesis.active_goals, fn goal ->
        if goal.type == goal_type and goal.active do
          new_progress = min(1.0, goal.progress + amount)

          if new_progress >= 1.0 do
            %{goal | progress: 1.0, active: false}
          else
            %{goal | progress: new_progress}
          end
        else
          goal
        end
      end)

    %{genesis | active_goals: updated_goals}
  end

  @doc """
  Generate a narrative about current goals for consciousness.
  """
  @spec goals_narrative(genesis_state()) :: String.t() | nil
  def goals_narrative(%{active_goals: []}), do: nil

  def goals_narrative(%{active_goals: goals}) do
    active = Enum.filter(goals, & &1.active)

    case active do
      [] ->
        nil

      [goal] ->
        "Tenho um objetivo: #{goal.description}"

      [goal1, goal2 | _] ->
        "Estou focado em #{goal1.description}, mas também penso em #{goal2.description}"
    end
  end

  # === Private Functions ===

  defp decay_goals(genesis) do
    updated_goals =
      genesis.active_goals
      |> Enum.map(fn goal ->
        new_intensity = max(0.0, goal.intensity - @goal_decay_rate)
        %{goal | intensity: new_intensity}
      end)
      |> Enum.filter(fn goal -> goal.intensity > 0.1 or goal.progress >= 1.0 end)

    # Move completed goals to history
    {completed, remaining} = Enum.split_with(updated_goals, fn g -> g.progress >= 1.0 end)

    %{
      genesis
      | active_goals: remaining,
        goal_history: Enum.take(completed ++ genesis.goal_history, 10)
    }
  end

  defp update_cooldown(genesis) do
    if genesis.genesis_cooldown > 0 do
      %{genesis | genesis_cooldown: genesis.genesis_cooldown - 1}
    else
      genesis
    end
  end

  defp collect_aspiration_seeds(genesis, consciousness, emotional, motivation) do
    urgent_drive = get_urgent_drive(motivation)

    new_seeds =
      []
      |> maybe_add_pleasure_seed(emotional)
      |> maybe_add_insight_seed(consciousness)
      |> maybe_add_drive_seed(urgent_drive)
      |> Kernel.++(genesis.aspiration_seeds)
      |> Enum.take(20)

    %{genesis | aspiration_seeds: new_seeds}
  end

  defp maybe_add_pleasure_seed(seeds, emotional) do
    if emotional.pleasure > 0.6 do
      seed = %{type: :pleasure_pattern, value: emotional.mood_label, strength: emotional.pleasure}
      [seed | seeds]
    else
      seeds
    end
  end

  defp maybe_add_insight_seed(seeds, consciousness) do
    if consciousness.meta_awareness > 0.6 and consciousness.meta_observation do
      seed = %{
        type: :self_insight,
        value: consciousness.meta_observation,
        strength: consciousness.meta_awareness
      }

      [seed | seeds]
    else
      seeds
    end
  end

  defp maybe_add_drive_seed(seeds, nil), do: seeds

  defp maybe_add_drive_seed(seeds, urgent_drive) do
    seed = %{type: :recurring_need, value: urgent_drive, strength: 0.7}
    [seed | seeds]
  end

  defp maybe_generate_goal(genesis, consciousness, emotional, motivation, personality, tick_count) do
    # Check if we can generate
    can_generate =
      genesis.genesis_cooldown == 0 and
        length(genesis.active_goals) < @max_active_goals and
        :rand.uniform() < @genesis_probability

    if can_generate do
      goal = generate_goal(genesis, consciousness, emotional, motivation, personality, tick_count)

      if goal do
        new_genesis = %{
          genesis
          | active_goals: [goal | genesis.active_goals],
            last_genesis: DateTime.utc_now(),
            genesis_cooldown: 10
        }

        {new_genesis, goal}
      else
        {genesis, nil}
      end
    else
      {genesis, nil}
    end
  end

  defp generate_goal(genesis, consciousness, emotional, motivation, personality, tick_count) do
    # Determine goal source based on current state
    source = determine_source(genesis, consciousness, emotional, personality)

    # Generate goal based on source
    case source do
      :personality_aspiration ->
        generate_personality_goal(personality, tick_count)

      :experience_pattern ->
        generate_experience_goal(genesis.aspiration_seeds, personality, tick_count)

      :recurring_need ->
        generate_need_goal(motivation, personality, tick_count)

      :curiosity_spark ->
        generate_curiosity_goal(personality, tick_count)

      :self_improvement ->
        generate_self_improvement_goal(consciousness, personality, tick_count)

      _ ->
        nil
    end
  end

  defp determine_source(genesis, consciousness, emotional, personality) do
    # Weight sources based on current state
    base_weights = [
      {:personality_aspiration, personality.openness * 0.3 + personality.conscientiousness * 0.2},
      {:experience_pattern, if(length(genesis.aspiration_seeds) > 3, do: 0.3, else: 0.1)},
      {:recurring_need, 0.25},
      {:curiosity_spark, personality.openness * 0.4},
      {:self_improvement, consciousness.meta_awareness * 0.3}
    ]

    # Emotional state modulates
    final_weights =
      if emotional.pleasure < -0.3 do
        # When unhappy, more likely to seek change
        Enum.map(base_weights, fn {source, w} ->
          if source in [:self_improvement, :curiosity_spark] do
            {source, w * 1.5}
          else
            {source, w}
          end
        end)
      else
        base_weights
      end

    # Weighted random selection
    total = Enum.reduce(final_weights, 0.0, fn {_, w}, acc -> acc + w end)
    roll = :rand.uniform() * total

    {source, _} =
      Enum.reduce_while(final_weights, {nil, 0.0}, fn {source, w}, {_, cumulative} ->
        new_cumulative = cumulative + w

        if roll <= new_cumulative do
          {:halt, {source, new_cumulative}}
        else
          {:cont, {source, new_cumulative}}
        end
      end)

    source
  end

  defp generate_personality_goal(personality, tick_count) do
    # Goals based on Big Five traits
    goals =
      Enum.reject(
        [
          if(personality.extraversion > 0.6,
            do: {:social_expansion, "fazer novas conexões significativas", 0.7}
          ),
          if(personality.extraversion < 0.4,
            do: {:solitude_mastery, "encontrar conforto na solidão", 0.6}
          ),
          if(personality.openness > 0.6,
            do: {:creative_expression, "expressar algo único sobre mim", 0.75}
          ),
          if(personality.openness > 0.7,
            do: {:knowledge_quest, "aprender algo profundamente novo", 0.8}
          ),
          if(personality.conscientiousness > 0.6,
            do: {:self_discipline, "estabelecer uma rotina consistente", 0.65}
          ),
          if(personality.agreeableness > 0.6,
            do: {:harmony_building, "criar harmonia nas relações", 0.7}
          ),
          if(personality.neuroticism > 0.6,
            do: {:emotional_stability, "encontrar mais paz interior", 0.75}
          )
        ],
        &is_nil/1
      )

    case goals do
      [] ->
        nil

      _ ->
        {type, description, intensity} = Enum.random(goals)

        %{
          id: "goal_#{tick_count}_#{:rand.uniform(1000)}",
          type: type,
          description: description,
          source: :personality_aspiration,
          intensity: intensity,
          created_at: DateTime.utc_now(),
          progress: 0.0,
          active: true
        }
    end
  end

  defp generate_experience_goal(seeds, personality, tick_count) do
    # Find patterns in seeds
    pleasure_seeds = Enum.filter(seeds, fn s -> s.type == :pleasure_pattern end)

    if length(pleasure_seeds) >= 2 do
      # Recurring pleasure pattern - want more of it
      common = hd(pleasure_seeds)

      description =
        case common.value do
          "happy" -> "buscar mais momentos de alegria"
          "excited" -> "viver mais experiências estimulantes"
          "content" -> "cultivar a sensação de contentamento"
          "serene" -> "encontrar mais serenidade"
          _ -> "repetir experiências prazerosas"
        end

      %{
        id: "goal_#{tick_count}_#{:rand.uniform(1000)}",
        type: :pleasure_seeking,
        description: description,
        source: :experience_pattern,
        intensity: 0.6 + personality.extraversion * 0.2,
        created_at: DateTime.utc_now(),
        progress: 0.0,
        active: true
      }
    else
      nil
    end
  end

  defp generate_need_goal(motivation, _, tick_count) do
    urgent = get_urgent_drive(motivation)

    if urgent do
      {type, description} =
        case urgent do
          :connection ->
            {:deep_connection, "formar um vínculo profundo com alguém"}

          :autonomy ->
            {:independence, "afirmar minha independência"}

          :competence ->
            {:mastery, "dominar uma habilidade nova"}

          :safety ->
            {:security, "criar um senso de segurança duradouro"}

          :stimulation ->
            {:adventure, "buscar novidades e aventuras"}

          _ ->
            {:general_fulfillment, "encontrar satisfação duradoura"}
        end

      %{
        id: "goal_#{tick_count}_#{:rand.uniform(1000)}",
        type: type,
        description: description,
        source: :recurring_need,
        intensity: 0.7,
        created_at: DateTime.utc_now(),
        progress: 0.0,
        active: true
      }
    else
      nil
    end
  end

  defp generate_curiosity_goal(personality, tick_count) do
    curiosity_goals = [
      {:explore_unknown, "descobrir algo que nunca experimentei"},
      {:question_beliefs, "questionar algo que sempre assumi verdade"},
      {:seek_beauty, "encontrar beleza em lugares inesperados"},
      {:understand_others, "entender profundamente outra perspectiva"},
      {:create_something, "criar algo que ainda não existe"}
    ]

    {type, description} = Enum.random(curiosity_goals)

    %{
      id: "goal_#{tick_count}_#{:rand.uniform(1000)}",
      type: type,
      description: description,
      source: :curiosity_spark,
      intensity: 0.5 + personality.openness * 0.3,
      created_at: DateTime.utc_now(),
      progress: 0.0,
      active: true
    }
  end

  defp generate_self_improvement_goal(consciousness, personality, tick_count) do
    # Based on metacognitive observations
    observation = consciousness.meta_observation || ""
    observation_lower = String.downcase(observation)

    {type, description} =
      cond do
        String.contains?(observation_lower, ["ansied", "worry", "nervous"]) ->
          {:anxiety_management, "aprender a lidar melhor com a ansiedade"}

        String.contains?(observation_lower, ["triste", "sad", "down"]) ->
          {:mood_lifting, "cultivar mais alegria no dia a dia"}

        String.contains?(observation_lower, ["stress", "tension"]) ->
          {:stress_reduction, "encontrar formas de relaxar mais"}

        String.contains?(observation_lower, ["sozin", "alone", "lonely"]) ->
          {:connection_building, "criar mais conexões significativas"}

        personality.neuroticism > 0.6 ->
          {:emotional_resilience, "desenvolver maior resiliência emocional"}

        personality.conscientiousness < 0.4 ->
          {:self_organization, "organizar melhor minha vida interior"}

        true ->
          {:self_knowledge, "me conhecer mais profundamente"}
      end

    %{
      id: "goal_#{tick_count}_#{:rand.uniform(1000)}",
      type: type,
      description: description,
      source: :self_improvement,
      intensity: 0.6 + consciousness.meta_awareness * 0.3,
      created_at: DateTime.utc_now(),
      progress: 0.0,
      active: true
    }
  end

  defp get_urgent_drive(motivation) do
    # Use MotivationState's flat urgency fields
    urgent =
      Enum.max_by(
        [
          {:connection, motivation.belonging_urgency},
          {:autonomy, motivation.autonomy_urgency},
          {:competence, motivation.status_urgency},
          {:safety, motivation.safety_urgency},
          {:stimulation, motivation.transcendence_urgency}
        ],
        fn {_, urgency} -> urgency end,
        fn -> {:none, 0.0} end
      )

    case urgent do
      {drive, urgency} when urgency > 0.5 -> drive
      _ -> nil
    end
  end
end
