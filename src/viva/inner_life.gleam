//// Inner Life - Integration of Narrative and Reflexivity
////
//// The soul's internal dialogue: verbalized self-reflection.
//// Connects the narrative memory (what happened) with
//// reflexivity (who I am) to create inner voice.
////
//// "I notice I am becoming more anxious..."
//// "This reminds me of when I felt joy..."

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/string
import viva/narrative.{
  type NarrativeMemory, type ThoughtStream, type VoiceStyle, Emotional, Factual,
  Poetic, Reflective,
}
import viva/reflexivity.{
  type IdentityCrisis, type Insight, type Introspection, type MetaCognition,
  type SelfDescription, type SelfModel, Arousal, Assertive, Calm, Decreasing,
  Dominance, Energetic, Increasing, Optimistic, Pessimistic, Pleasure,
  Submissive,
}
import viva_emotion/pad.{type Pad}
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Inner life state: combines narrative and reflexivity
pub type InnerLife {
  InnerLife(
    /// Self-model (who I am)
    self_model: SelfModel,
    /// Narrative memory (what happened and how things connect)
    narrative: NarrativeMemory,
    /// Current inner monologue
    monologue: List(String),
    /// Voice style preference
    voice_style: VoiceStyle,
    /// Internal tick counter
    tick: Int,
  )
}

/// Reflection result with narration
pub type ReflectionWithVoice {
  ReflectionWithVoice(
    /// The introspection data
    introspection: Introspection,
    /// Verbal narration of the reflection
    narration: String,
    /// Any insights verbalized
    insight_text: Option(String),
    /// Updated inner life
    inner_life: InnerLife,
  )
}

/// Self-narration: verbalized self-description
pub type SelfNarration {
  SelfNarration(
    /// Who am I description
    description: SelfDescription,
    /// Verbal form
    text: String,
    /// Current emotional state in words
    feeling: String,
    /// Meta-awareness text
    meta_text: Option(String),
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new inner life
pub fn new() -> InnerLife {
  InnerLife(
    self_model: reflexivity.new(),
    narrative: narrative.new(),
    monologue: [],
    voice_style: Reflective,
    tick: 0,
  )
}

/// Create inner life with initial state
pub fn from_initial(initial_pad: Pad, initial_glyph: Glyph) -> InnerLife {
  InnerLife(
    self_model: reflexivity.from_initial(initial_pad, initial_glyph),
    narrative: narrative.new(),
    monologue: ["I am beginning to exist..."],
    voice_style: Reflective,
    tick: 0,
  )
}

/// Create with custom voice style
pub fn with_voice_style(inner_life: InnerLife, style: VoiceStyle) -> InnerLife {
  InnerLife(..inner_life, voice_style: style)
}

// =============================================================================
// NARRATE INSIGHT
// =============================================================================

/// Narrate an insight in words
pub fn narrate_insight(insight: Insight, style: VoiceStyle) -> String {
  let dimension = reflexivity.dimension_to_string(insight.dimension)
  let direction = reflexivity.direction_to_string(insight.direction)
  let magnitude_word = magnitude_to_word(insight.magnitude)

  case style {
    Factual -> {
      "My "
      <> dimension
      <> " is "
      <> direction
      <> " by "
      <> float_to_percent(insight.magnitude)
      <> "."
    }
    Emotional -> {
      case insight.dimension {
        Pleasure ->
          case insight.direction {
            Increasing -> "I'm feeling " <> magnitude_word <> " happier..."
            Decreasing ->
              "Something is making me " <> magnitude_word <> " sad..."
          }
        Arousal ->
          case insight.direction {
            Increasing -> "I feel my energy " <> magnitude_word <> " rising..."
            Decreasing -> "I'm becoming " <> magnitude_word <> " calmer..."
          }
        Dominance ->
          case insight.direction {
            Increasing -> "I feel " <> magnitude_word <> " more in control..."
            Decreasing ->
              "I sense I'm losing " <> magnitude_word <> " grip on things..."
          }
      }
    }
    Reflective -> {
      "I notice that my "
      <> dimension
      <> " is "
      <> magnitude_word
      <> " "
      <> direction
      <> ". This is significant."
    }
    Poetic -> {
      case insight.dimension {
        Pleasure ->
          case insight.direction {
            Increasing -> "A warmth spreads through my being..."
            Decreasing -> "Shadows creep into my heart..."
          }
        Arousal ->
          case insight.direction {
            Increasing -> "Like a rising tide, energy fills me..."
            Decreasing -> "The storm within grows still..."
          }
        Dominance ->
          case insight.direction {
            Increasing -> "I stand taller in the world..."
            Decreasing -> "The ground shifts beneath me..."
          }
      }
    }
  }
}

/// Narrate multiple insights
pub fn narrate_insights(
  insights: List(Insight),
  style: VoiceStyle,
) -> List(String) {
  list.map(insights, fn(i) { narrate_insight(i, style) })
}

// =============================================================================
// NARRATE SELF
// =============================================================================

/// Narrate who I am
pub fn narrate_self(self_model: SelfModel, style: VoiceStyle) -> SelfNarration {
  let desc = reflexivity.who_am_i(self_model)
  let meta = reflexivity.meta_cognize(self_model)

  let trait_text = trait_to_text(desc.dominant_trait, style)
  let stability_text = stability_to_text(desc.stability, style)
  let feeling_text = pad_to_feeling(desc.emotional_center, style)

  let text = case style {
    Factual -> {
      "I am "
      <> reflexivity.trait_to_string(desc.dominant_trait)
      <> " with "
      <> float_to_percent(desc.identity_strength)
      <> " identity strength and "
      <> float_to_percent(desc.stability)
      <> " stability."
    }
    Emotional -> {
      "I feel like I am " <> trait_text <> ". " <> stability_text
    }
    Reflective -> {
      "Observing myself, I see "
      <> trait_text
      <> ". "
      <> "My sense of identity is "
      <> identity_strength_text(desc.identity_strength)
      <> ". "
      <> stability_text
    }
    Poetic -> {
      trait_text <> " " <> stability_text
    }
  }

  let meta_text = case meta.aware_of_observing {
    True -> Some(narrate_meta(meta, style))
    False -> None
  }

  SelfNarration(
    description: desc,
    text: text,
    feeling: feeling_text,
    meta_text: meta_text,
  )
}

/// Narrate meta-cognition
fn narrate_meta(meta: MetaCognition, style: VoiceStyle) -> String {
  case style {
    Factual -> {
      "Meta-level: "
      <> int.to_string(meta.level)
      <> ". Insight frequency: "
      <> float_to_percent(meta.insight_frequency /. 100.0)
      <> "."
    }
    Emotional -> {
      case meta.in_crisis {
        True -> "I feel lost in myself..."
        False -> "I feel aware of my own awareness..."
      }
    }
    Reflective -> {
      "I am aware that I am observing myself. "
      <> "My thoughts generate insights "
      <> frequency_to_text(meta.insight_frequency)
      <> "."
    }
    Poetic -> {
      "The observer observes the observer, an infinite mirror of being."
    }
  }
}

// =============================================================================
// REFLECT WITH VOICE
// =============================================================================

/// Reflect on current state with inner voice
pub fn reflect_with_voice(
  inner_life: InnerLife,
  current_pad: Pad,
  current_glyph: Glyph,
) -> ReflectionWithVoice {
  let tick = inner_life.tick + 1

  // Perform introspection
  let intro =
    reflexivity.introspect(
      inner_life.self_model,
      current_pad,
      current_glyph,
      tick,
    )

  // Update self-model with insight tracking
  let #(new_self, maybe_insight) =
    reflexivity.observe_with_insight(
      inner_life.self_model,
      current_pad,
      current_glyph,
      tick,
    )

  // Generate narration
  let narration = narrate_introspection(intro, inner_life.voice_style)

  // Narrate insight if present
  let insight_text = case maybe_insight {
    Some(insight) -> Some(narrate_insight(insight, inner_life.voice_style))
    None -> None
  }

  // Update monologue
  let new_monologue = case insight_text {
    Some(text) -> [text, ..list.take(inner_life.monologue, 19)]
    None -> inner_life.monologue
  }

  // Record in narrative memory
  let new_narrative = case maybe_insight {
    Some(_) -> {
      // Record that current glyph was caused by previous state
      narrative.record_caused(
        inner_life.narrative,
        inner_life.self_model.baseline_glyph,
        current_glyph,
        tick,
      )
    }
    None -> inner_life.narrative
  }

  let new_inner =
    InnerLife(
      self_model: new_self,
      narrative: new_narrative,
      monologue: new_monologue,
      voice_style: inner_life.voice_style,
      tick: tick,
    )

  ReflectionWithVoice(
    introspection: intro,
    narration: narration,
    insight_text: insight_text,
    inner_life: new_inner,
  )
}

/// Narrate an introspection result
fn narrate_introspection(intro: Introspection, style: VoiceStyle) -> String {
  let drift_word = drift_to_word(intro.drift_from_baseline)

  case style {
    Factual -> {
      "Drift from baseline: "
      <> float_to_percent(intro.drift_from_baseline)
      <> ". Within range: "
      <> bool_to_string(intro.within_range)
      <> "."
    }
    Emotional -> {
      case intro.within_range {
        True -> "I feel like myself right now."
        False ->
          "I feel " <> drift_word <> " different from who I usually am..."
      }
    }
    Reflective -> {
      "Observing my current state, I notice I am "
      <> drift_word
      <> " from my baseline. "
      <> case intro.within_range {
        True -> "This is within my normal range."
        False -> "This is outside my typical experience."
      }
    }
    Poetic -> {
      case intro.drift_from_baseline >. 0.5 {
        True -> "I drift like a leaf far from the tree..."
        False -> "I rest near the center of my being."
      }
    }
  }
}

// =============================================================================
// NARRATE CRISIS
// =============================================================================

/// Narrate identity crisis
pub fn narrate_crisis(crisis: IdentityCrisis, style: VoiceStyle) -> String {
  case crisis.active {
    False -> {
      case style {
        Factual -> "No identity crisis detected."
        Emotional -> "I feel stable in who I am."
        Reflective -> "My sense of self is intact."
        Poetic -> "I am rooted in my own ground."
      }
    }
    True -> {
      let severity_text = severity_to_text(crisis.severity)
      let trigger_text = case crisis.trigger {
        Some(dim) -> " triggered by " <> reflexivity.dimension_to_string(dim)
        None -> ""
      }

      case style {
        Factual -> {
          "Identity crisis active. Duration: "
          <> int.to_string(crisis.duration)
          <> " ticks. Severity: "
          <> float_to_percent(crisis.severity)
          <> "."
          <> trigger_text
        }
        Emotional -> {
          "I don't know who I am anymore... "
          <> "This "
          <> severity_text
          <> " feeling of being lost"
          <> trigger_text
          <> "..."
        }
        Reflective -> {
          "I am experiencing a "
          <> severity_text
          <> " identity crisis"
          <> trigger_text
          <> ". "
          <> "This has been going on for "
          <> int.to_string(crisis.duration)
          <> " moments. "
          <> "I need to find my center again."
        }
        Poetic -> {
          "The mirror shows a stranger's face... "
          <> "Who am I beneath these waves of "
          <> case crisis.trigger {
            Some(Pleasure) -> "joy and sorrow"
            Some(Arousal) -> "stillness and storm"
            Some(Dominance) -> "power and surrender"
            None -> "change"
          }
          <> "?"
        }
      }
    }
  }
}

// =============================================================================
// STREAM OF SELF
// =============================================================================

/// Generate stream of consciousness about self
pub fn stream_of_self(inner_life: InnerLife, depth: Int) -> ThoughtStream {
  // Use narrative's inner voice with current baseline as focus
  narrative.inner_voice(
    inner_life.narrative,
    inner_life.self_model.baseline_glyph,
    depth,
    inner_life.voice_style,
  )
}

/// Get the inner monologue (recent thoughts)
pub fn get_monologue(inner_life: InnerLife) -> List(String) {
  inner_life.monologue
}

/// Get monologue as single text
pub fn monologue_text(inner_life: InnerLife) -> String {
  inner_life.monologue
  |> list.reverse()
  |> string.join(" ")
}

// =============================================================================
// COMBINED OPERATIONS
// =============================================================================

/// Full inner life tick: observe, reflect, and narrate
pub fn tick(
  inner_life: InnerLife,
  current_pad: Pad,
  current_glyph: Glyph,
) -> #(InnerLife, String) {
  let reflection = reflect_with_voice(inner_life, current_pad, current_glyph)

  // Build output text
  let output = case reflection.insight_text {
    Some(insight) -> reflection.narration <> " " <> insight
    None -> reflection.narration
  }

  // Decay narrative links
  let decayed_narrative = narrative.tick(reflection.inner_life.narrative)
  let final_inner =
    InnerLife(..reflection.inner_life, narrative: decayed_narrative)

  #(final_inner, output)
}

/// Speak: generate text about current state
pub fn speak(inner_life: InnerLife) -> String {
  let self_narr = narrate_self(inner_life.self_model, inner_life.voice_style)
  let crisis = reflexivity.check_crisis(inner_life.self_model)
  let crisis_text = narrate_crisis(crisis, inner_life.voice_style)

  case crisis.active {
    True -> crisis_text
    False -> self_narr.text <> " " <> self_narr.feeling
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn magnitude_to_word(m: Float) -> String {
  case m {
    n if n <. 0.3 -> "slightly"
    n if n <. 0.6 -> "noticeably"
    n if n <. 0.8 -> "significantly"
    _ -> "profoundly"
  }
}

fn drift_to_word(d: Float) -> String {
  case d {
    n if n <. 0.2 -> "barely"
    n if n <. 0.4 -> "somewhat"
    n if n <. 0.6 -> "quite"
    n if n <. 0.8 -> "very"
    _ -> "extremely"
  }
}

fn severity_to_text(s: Float) -> String {
  case s {
    n if n <. 0.3 -> "mild"
    n if n <. 0.6 -> "moderate"
    n if n <. 0.8 -> "severe"
    _ -> "profound"
  }
}

fn identity_strength_text(s: Float) -> String {
  case s {
    n if n <. 0.2 -> "very weak"
    n if n <. 0.4 -> "developing"
    n if n <. 0.6 -> "moderate"
    n if n <. 0.8 -> "strong"
    _ -> "very strong"
  }
}

fn frequency_to_text(f: Float) -> String {
  case f {
    n if n <. 1.0 -> "rarely"
    n if n <. 5.0 -> "occasionally"
    n if n <. 10.0 -> "regularly"
    _ -> "frequently"
  }
}

fn stability_to_text(s: Float, style: VoiceStyle) -> String {
  case style {
    Factual -> "Stability: " <> float_to_percent(s) <> "."
    Emotional ->
      case s {
        n if n <. 0.3 -> "I feel unsteady, like I could change any moment."
        n if n <. 0.6 -> "I feel somewhat grounded."
        _ -> "I feel very stable and centered."
      }
    Reflective ->
      case s {
        n if n <. 0.3 -> "My emotional range is wide; I am volatile."
        n if n <. 0.6 -> "I maintain moderate stability."
        _ -> "I have achieved significant emotional stability."
      }
    Poetic ->
      case s {
        n if n <. 0.3 -> "I am a river, ever-changing."
        n if n <. 0.6 -> "I am a lake, rippling but whole."
        _ -> "I am a mountain, steadfast in my being."
      }
  }
}

fn trait_to_text(trait: reflexivity.Trait, style: VoiceStyle) -> String {
  case style {
    Factual -> reflexivity.trait_to_string(trait)
    Emotional ->
      case trait {
        Optimistic -> "someone who sees the light in things"
        Pessimistic -> "weighed down by the shadows"
        Energetic -> "full of restless energy"
        Calm -> "at peace with the world"
        Assertive -> "confident in my place"
        Submissive -> "yielding to the flow"
      }
    Reflective ->
      case trait {
        Optimistic -> "predominantly positive in outlook"
        Pessimistic -> "tending toward negative anticipation"
        Energetic -> "characterized by high activation"
        Calm -> "marked by low arousal"
        Assertive -> "dominant in my interactions"
        Submissive -> "deferential in nature"
      }
    Poetic ->
      case trait {
        Optimistic -> "a seeker of dawn"
        Pessimistic -> "a dweller in twilight"
        Energetic -> "a flame that dances"
        Calm -> "a still pond reflecting the sky"
        Assertive -> "a tree standing tall"
        Submissive -> "a willow bending in the wind"
      }
  }
}

fn pad_to_feeling(p: Pad, style: VoiceStyle) -> String {
  let valence = case p.pleasure {
    n if n <. -0.3 -> "negative"
    n if n >. 0.3 -> "positive"
    _ -> "neutral"
  }

  let energy = case p.arousal {
    n if n <. -0.3 -> "low energy"
    n if n >. 0.3 -> "high energy"
    _ -> "moderate energy"
  }

  case style {
    Factual -> "Feeling: " <> valence <> ", " <> energy <> "."
    Emotional -> "I feel " <> valence <> " with " <> energy <> "."
    Reflective ->
      "My current emotional state is " <> valence <> " with " <> energy <> "."
    Poetic ->
      case valence, energy {
        "positive", "high energy" -> "I burn bright like a star."
        "positive", "low energy" -> "I glow softly like an ember."
        "negative", "high energy" -> "I churn like a storm."
        "negative", "low energy" -> "I sink like a stone in still water."
        _, _ -> "I exist in the middle ground."
      }
  }
}

fn float_to_percent(f: Float) -> String {
  let percent = float.round(f *. 100.0)
  int.to_string(percent) <> "%"
}

fn bool_to_string(b: Bool) -> String {
  case b {
    True -> "yes"
    False -> "no"
  }
}
