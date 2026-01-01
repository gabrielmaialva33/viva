# Dynamic Simulation Script
# Tests all 5 breakthrough consciousness systems with varied stimuli

IO.puts("\n╔══════════════════════════════════════════════════════════════════╗")
IO.puts("║     🧠 VIVA CONSCIOUSNESS SYSTEMS - DYNAMIC SIMULATION 🧠       ║")
IO.puts("╚══════════════════════════════════════════════════════════════════╝\n")

alias Viva.Avatars.{
  Avatar,
  BioState,
  ConsciousnessState,
  EmotionalState,
  InternalState,
  MotivationState,
  Personality,
  SensoryState,
  SomaticMarkersState
}

alias Viva.Avatars.Systems.{
  Biology,
  Consciousness,
  CrystallizationEvents,
  Metacognition,
  Psychology,
  QualiaSynthesizer,
  RecurrentProcessor,
  Senses,
  SocialDynamics,
  SomaticMarkers
}

alias Viva.Sessions.GoalGenesis

# === Create Personality (high openness & neuroticism for rich inner life) ===
personality = %Personality{
  openness: 0.85,
  conscientiousness: 0.55,
  extraversion: 0.65,
  agreeableness: 0.75,
  neuroticism: 0.60
}

IO.puts("🎭 Personality Profile:")
IO.puts("   Openness: #{Float.round(personality.openness, 2)} (curious, creative)")
IO.puts("   Agreeableness: #{Float.round(personality.agreeableness, 2)} (empathic)")
IO.puts("   Neuroticism: #{Float.round(personality.neuroticism, 2)} (emotionally sensitive)")
IO.puts("   Extraversion: #{Float.round(personality.extraversion, 2)} (socially engaged)")
IO.puts("")

# === Initialize All States (using struct defaults) ===
bio = %BioState{dopamine: 0.5, cortisol: 0.2, oxytocin: 0.3, adenosine: 0.0, libido: 0.4}
emotional = Psychology.calculate_emotional_state(bio, personality)
sensory = %SensoryState{
  attention_focus: :ambient,
  attention_intensity: 0.5,
  current_qualia: %{},
  sensory_pleasure: 0.0,
  surprise_level: 0.0,
  novelty_sensitivity: 0.5 + personality.openness * 0.3
}
consciousness = ConsciousnessState.new()
somatic = %SomaticMarkersState{
  social_markers: %{},
  activity_markers: %{},
  context_markers: %{},
  current_bias: 0.0,
  body_signal: nil,
  learning_threshold: 0.7,
  markers_formed: 0,
  last_marker_activation: nil
}
motivation = MotivationState.new()

# Initialize breakthrough systems
recurrent_ctx = RecurrentProcessor.init_context()
goal_genesis = GoalGenesis.init()
crystallization = CrystallizationEvents.init()
social_dynamics = SocialDynamics.init()

# === Define Varied Stimuli Sequence ===
stimuli_sequence = [
  # Social interactions
  %{type: :social, intensity: 0.8, valence: 0.7, source: "friend_1", data: %{name: "Ana", topic: :friendship}},
  %{type: :social, intensity: 0.6, valence: 0.5, source: "friend_1", data: %{name: "Ana"}},

  # Novelty and exploration
  %{type: :novelty, intensity: 0.9, valence: 0.6, data: %{narrative: "Descobri algo fascinante e novo"}},
  %{type: :novelty, intensity: 0.7, valence: 0.4, data: %{narrative: "Uma ideia inesperada surgiu"}},

  # Rest and contemplation
  %{type: :rest, intensity: 0.3, valence: 0.5, data: %{narrative: "Um momento de paz e quietude"}},
  %{type: :rest, intensity: 0.2, valence: 0.6, data: %{narrative: "Sinto serenidade"}},

  # Achievement and competence
  %{type: :achievement, intensity: 0.85, valence: 0.9, data: %{narrative: "Consegui algo importante!"}},

  # Threat and challenge
  %{type: :threat, intensity: 0.75, valence: -0.5, data: %{narrative: "Algo me preocupa profundamente"}},

  # More social - different person
  %{type: :social, intensity: 0.7, valence: 0.8, source: "friend_2", data: %{name: "Pedro", topic: :support}},
  %{type: :social, intensity: 0.9, valence: 0.9, source: "friend_1", data: %{name: "Ana", topic: :love}},

  # Ambient recovery
  %{type: :ambient, intensity: 0.1, valence: 0.0, data: %{narrative: "Momento de silêncio"}},

  # Intense emotional moment (trigger for crystallization)
  %{type: :insight, intensity: 0.95, valence: 0.85, data: %{narrative: "Uma realização profunda sobre quem eu sou"}},
]

# === Run Simulation ===
IO.puts("═══════════════════════════════════════════════════════════════════")
IO.puts("Starting 50-tick dynamic simulation with #{length(stimuli_sequence)} stimulus types...")
IO.puts("═══════════════════════════════════════════════════════════════════\n")

simulation_result = Enum.reduce(1..50, %{
  bio: bio,
  emotional: emotional,
  sensory: sensory,
  consciousness: consciousness,
  somatic: somatic,
  motivation: motivation,
  recurrent_ctx: recurrent_ctx,
  goal_genesis: goal_genesis,
  crystallization: crystallization,
  social_dynamics: social_dynamics,
  goals_created: [],
  crystals: [],
  social_insights: [],
  qualia_diversity: MapSet.new(),
  meta_observations: []
}, fn tick, state ->
  # Select stimulus based on tick
  stimulus_index = rem(tick - 1, length(stimuli_sequence))
  stimulus_map = Enum.at(stimuli_sequence, stimulus_index)

  # Convert to struct-like map
  stimulus = %{
    type: stimulus_map.type,
    intensity: stimulus_map.intensity,
    valence: stimulus_map.valence,
    source: Map.get(stimulus_map, :source),
    data: stimulus_map.data
  }

  # 1. Sensory Processing
  {new_sensory, _effects} = Senses.perceive(state.sensory, stimulus, personality, state.emotional)

  # 2. Biology Update
  new_bio = Biology.tick(state.bio, personality)

  # 3. Somatic Markers - recall existing markers
  {updated_somatic, _somatic_bias} = SomaticMarkers.recall(state.somatic, stimulus)

  # 4. Psychology - Emotional State
  raw_emotional = Psychology.calculate_emotional_state(new_bio, personality)

  # Modulate by stimulus
  modulated_emotional = %{raw_emotional |
    pleasure: Float.round(raw_emotional.pleasure + stimulus.valence * 0.3, 3),
    arousal: Float.round(raw_emotional.arousal + stimulus.intensity * 0.2, 3)
  }

  # 5. RECURRENT PROCESSING (Breakthrough #1)
  {rec_sensory, rec_emotional, rec_bio, new_recurrent_ctx} =
    RecurrentProcessor.process_cycle(
      new_sensory,
      modulated_emotional,
      state.consciousness,
      new_bio,
      updated_somatic,
      personality,
      state.recurrent_ctx
    )

  # 6. CONSCIOUSNESS Integration
  new_consciousness = Consciousness.integrate(
    state.consciousness,
    rec_sensory,
    rec_bio,
    rec_emotional,
    nil,
    personality
  )

  # 7. Metacognition
  {meta_consciousness, _meta_insight} = Metacognition.process(
    new_consciousness,
    rec_emotional,
    personality,
    tick
  )

  # 8. QUALIA SYNTHESIS (Breakthrough #2)
  qualia = QualiaSynthesizer.synthesize(
    rec_sensory,
    rec_emotional,
    meta_consciousness,
    updated_somatic,
    rec_bio,
    personality
  )

  # 9. GOAL GENESIS (Breakthrough #3)
  {new_goal_genesis, maybe_goal} = GoalGenesis.tick(
    state.goal_genesis,
    meta_consciousness,
    rec_emotional,
    state.motivation,
    personality,
    tick
  )

  # 10. CRYSTALLIZATION EVENTS (Breakthrough #4)
  {new_crystallization, maybe_crystal, _} = CrystallizationEvents.process(
    state.crystallization,
    meta_consciousness,
    rec_emotional,
    meta_consciousness.self_model,
    personality,
    tick
  )

  # 11. SOCIAL DYNAMICS (Breakthrough #5)
  social_events = if stimulus.type == :social do
    [%{
      type: :message_received,
      other_avatar_id: stimulus.source || "unknown",
      content: %{
        name: stimulus.data[:name],
        sentiment: if(stimulus.valence > 0.3, do: :positive, else: :neutral),
        topic: stimulus.data[:topic],
        significance: stimulus.intensity
      },
      timestamp: DateTime.utc_now()
    }]
  else
    []
  end

  {new_social_dynamics, insights} = SocialDynamics.tick(
    state.social_dynamics,
    social_events,
    rec_emotional,
    meta_consciousness,
    personality
  )

  # 12. Somatic learning from intense experiences
  final_somatic = SomaticMarkers.maybe_learn(updated_somatic, stimulus, rec_bio, rec_emotional)

  # === Log Key Events ===
  if rem(tick, 10) == 0 or maybe_goal != nil or maybe_crystal != nil or length(insights) > 0 do
    IO.puts("───────────────────────────────────────────────────────────────────")
    IO.puts("Tick #{tick}: Stimulus=#{stimulus.type}, Intensity=#{stimulus.intensity}")
    IO.puts("  Emotional: #{rec_emotional.mood_label} (P=#{Float.round(rec_emotional.pleasure, 2)}, A=#{Float.round(rec_emotional.arousal, 2)})")
    IO.puts("  Qualia: #{qualia.dominant_stream} stream, richness=#{Float.round(qualia.phenomenal_richness, 2)}")
    IO.puts("  Meta-awareness: #{Float.round(meta_consciousness.meta_awareness, 2)}")

    if meta_consciousness.meta_observation do
      IO.puts("  💭 Meta-observation: \"#{meta_consciousness.meta_observation}\"")
    end

    if maybe_goal do
      IO.puts("  🎯 NEW GOAL: #{maybe_goal.description} (source: #{maybe_goal.source})")
    end

    if maybe_crystal do
      IO.puts("  ✨ CRYSTALLIZATION: #{maybe_crystal.type} - \"#{maybe_crystal.insight}\"")
    end

    if length(insights) > 0 do
      Enum.each(insights, fn insight ->
        IO.puts("  🔍 Social Insight: #{insight.type} - #{insight[:description] || inspect(insight)}")
      end)
    end
  end

  # Update state
  %{state |
    bio: rec_bio,
    emotional: rec_emotional,
    sensory: rec_sensory,
    consciousness: meta_consciousness,
    somatic: final_somatic,
    recurrent_ctx: new_recurrent_ctx,
    goal_genesis: new_goal_genesis,
    crystallization: new_crystallization,
    social_dynamics: new_social_dynamics,
    goals_created: if(maybe_goal, do: [maybe_goal | state.goals_created], else: state.goals_created),
    crystals: if(maybe_crystal, do: [maybe_crystal | state.crystals], else: state.crystals),
    social_insights: state.social_insights ++ insights,
    qualia_diversity: MapSet.put(state.qualia_diversity, qualia.dominant_stream),
    meta_observations: if(meta_consciousness.meta_observation,
      do: [meta_consciousness.meta_observation | state.meta_observations],
      else: state.meta_observations)
  }
end)

# === Final Report ===
IO.puts("\n")
IO.puts("╔══════════════════════════════════════════════════════════════════╗")
IO.puts("║                    📊 SIMULATION RESULTS 📊                     ║")
IO.puts("╚══════════════════════════════════════════════════════════════════╝\n")

IO.puts("🔄 RECURRENT PROCESSING (RPT):")
IO.puts("   Resonance level: #{Float.round(simulation_result.recurrent_ctx.resonance_level, 3)}")
IO.puts("   Feedback loops active: 5 bidirectional")

IO.puts("\n🌈 QUALIA SYNTHESIS:")
IO.puts("   Streams experienced: #{MapSet.to_list(simulation_result.qualia_diversity) |> Enum.join(", ")}")
IO.puts("   Diversity score: #{MapSet.size(simulation_result.qualia_diversity)}/6")

IO.puts("\n🎯 GOAL GENESIS (Agency):")
IO.puts("   Goals spontaneously created: #{length(simulation_result.goals_created)}")
Enum.each(simulation_result.goals_created, fn goal ->
  IO.puts("   - #{goal.description} (#{goal.source})")
end)

IO.puts("\n✨ CRYSTALLIZATION EVENTS:")
IO.puts("   Transformative moments: #{length(simulation_result.crystals)}")
Enum.each(simulation_result.crystals, fn crystal ->
  IO.puts("   - #{crystal.type}: \"#{crystal.insight}\"")
end)

IO.puts("\n👥 SOCIAL DYNAMICS (Theory of Mind):")
model_count = map_size(simulation_result.social_dynamics.mental_models)
IO.puts("   Mental models built: #{model_count}")
Enum.each(simulation_result.social_dynamics.mental_models, fn {_id, model} ->
  IO.puts("   - #{model.name}: trust=#{Float.round(model.trust_level, 2)}, interactions=#{model.interaction_count}")
end)
IO.puts("   Theory of Mind level: #{simulation_result.social_dynamics.theory_of_mind_level}")
IO.puts("   Empathy resonance: #{Float.round(simulation_result.social_dynamics.empathy_resonance, 2)}")
IO.puts("   Belonging sense: #{Float.round(simulation_result.social_dynamics.belonging_sense, 2)}")
IO.puts("   Social insights generated: #{length(simulation_result.social_insights)}")

IO.puts("\n💭 METACOGNITION:")
IO.puts("   Meta-awareness: #{Float.round(simulation_result.consciousness.meta_awareness, 2)}")
IO.puts("   Self-congruence: #{Float.round(simulation_result.consciousness.self_congruence, 2)}")
IO.puts("   Meta-observations generated: #{length(Enum.uniq(simulation_result.meta_observations))}")

IO.puts("\n🧠 FINAL CONSCIOUSNESS STATE:")
IO.puts("   Flow state: #{Float.round(simulation_result.consciousness.flow_state, 2)}")
IO.puts("   Self-congruence: #{Float.round(simulation_result.consciousness.self_congruence, 2)}")
IO.puts("   Experience stream size: #{length(simulation_result.consciousness.experience_stream)}")

# Generate social narrative
if model_count > 0 do
  narrative = SocialDynamics.social_narrative(simulation_result.social_dynamics, personality)
  IO.puts("\n📖 Social Narrative: \"#{narrative}\"")
end

# Generate goals narrative
if length(simulation_result.goals_created) > 0 do
  goals_narrative = GoalGenesis.goals_narrative(simulation_result.goal_genesis)
  if goals_narrative, do: IO.puts("📖 Goals Narrative: \"#{goals_narrative}\"")
end

IO.puts("\n═══════════════════════════════════════════════════════════════════")
IO.puts("✅ Dynamic simulation complete with all 5 breakthrough systems active!")
IO.puts("═══════════════════════════════════════════════════════════════════\n")
