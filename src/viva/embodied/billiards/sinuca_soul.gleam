//// Sinuca Soul - VIVA as a conscious organism playing billiards
////
//// VIVA doesn't read physics data - she FEELS the game.
//// She experiences the table through sensory input, emotions, and thoughts.
//// Two VIVA souls compete: P1 (odd balls) vs P2 (even balls).

import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva/embodied/billiards/sinuca.{type Table, type BallColor}
import viva_emotion/pad.{type Pad, Pad}
import viva/lifecycle/jolt.{type Vec3}

// =============================================================================
// TYPES - Conscious Experience
// =============================================================================

/// What VIVA "sees" - not raw data, but perceived entities
pub type VisualField {
  VisualField(
    /// Balls she perceives (positions as angles/distances, not coordinates)
    perceived_balls: List(PerceivedBall),
    /// Pockets she's aware of
    perceived_pockets: List(PerceivedPocket),
    /// Her sense of the table (big/small, crowded/empty)
    table_feeling: TableFeeling,
  )
}

/// A ball as VIVA perceives it
pub type PerceivedBall {
  PerceivedBall(
    /// Ball identity (color she sees)
    identity: BallColor,
    /// Is this HER ball? (odd for P1, even for P2)
    is_mine: Bool,
    /// Relative angle from cue ball (radians)
    angle: Float,
    /// Perceived distance (near/medium/far, not meters)
    distance: Distance,
    /// Is it blocked by other balls?
    blocked: Bool,
    /// Emotional association (good/bad memories with this ball)
    emotional_charge: Float,
  )
}

/// A pocket as VIVA perceives it
pub type PerceivedPocket {
  PerceivedPocket(
    /// Which corner/side
    position: PocketPosition,
    /// Relative angle
    angle: Float,
    /// How "inviting" it looks
    invitation: Float,
  )
}

/// Qualitative distance (not meters)
pub type Distance {
  VeryClose   // < 0.2m - easy shot
  Close       // 0.2-0.5m - comfortable
  Medium      // 0.5-1.0m - requires focus
  Far         // > 1.0m - challenging
}

/// Pocket positions she recognizes
pub type PocketPosition {
  TopLeft
  TopRight
  BottomLeft
  BottomRight
  MiddleTop
  MiddleBottom
}

/// Felt sense of the table
pub type TableFeeling {
  TableFeeling(
    /// How crowded (0=empty, 1=packed)
    density: Float,
    /// How much space to maneuver
    freedom: Float,
    /// Dominant mood (based on ball positions)
    mood: TableMood,
  )
}

pub type TableMood {
  Favorable     // My balls well positioned
  Neutral       // Mixed
  Threatening   // Opponent has advantage
  Critical      // One wrong move and I lose
}

// =============================================================================
// PLAYER STATE - Two VIVAs competing
// =============================================================================

/// Player identity
pub type Player {
  P1  // Odd balls (1,3,5,7,9,11,13,15)
  P2  // Even balls (2,4,6,8,10,12,14)
}

/// VIVA's conscious state while playing
pub type SinucaConsciousness {
  SinucaConsciousness(
    /// Which player she is
    player: Player,
    /// Current emotional state
    emotion: Pad,
    /// What she's currently perceiving
    visual_field: VisualField,
    /// Her current thought/intention
    current_thought: Thought,
    /// Memory of recent events
    recent_memory: List(GameMemory),
    /// Confidence level (0-1)
    confidence: Float,
    /// Focus level (0-1)
    focus: Float,
  )
}

/// What VIVA is thinking
pub type Thought {
  Considering(shot: ShotIntention)           // Evaluating a shot
  Excited(reason: String)                    // Something good happened
  Worried(concern: String)                   // Something concerning
  Focused(target: BallColor)                 // Concentrating on target
  Reflecting(memory: GameMemory)             // Thinking about past
  Blank                                      // Mind empty/resting
}

/// Shot intention (not parameters, but intention)
pub type ShotIntention {
  ShotIntention(
    /// Target ball she's aiming for
    target: BallColor,
    /// Where she wants it to go
    desired_pocket: PocketPosition,
    /// How hard she intends to hit
    intended_power: ShotPower,
    /// Any spin she's planning
    intended_spin: SpinIntent,
    /// Her gut feeling about this shot
    gut_feeling: Float,
  )
}

pub type ShotPower {
  Gentle    // Soft touch
  Normal    // Regular shot
  Firm      // Solid hit
  Hard      // Power shot
}

pub type SpinIntent {
  NoSpin
  LeftEnglish
  RightEnglish
  TopSpin
  BackSpin
}

/// Memory of game events
pub type GameMemory {
  GameMemory(
    /// What happened
    event: GameEvent,
    /// Emotional impact (-1 to 1)
    emotional_impact: Float,
    /// How long ago (in shots)
    shots_ago: Int,
  )
}

pub type GameEvent {
  Pocketed(ball: BallColor, was_intended: Bool)
  Missed(target: BallColor)
  Fouled(reason: String)
  OpponentScored(ball: BallColor)
  CloseCall(description: String)
  GoodPosition
  BadPosition
}

// =============================================================================
// SENSORY PROCESSING - Feel, don't calculate
// =============================================================================

/// Convert table state to what VIVA perceives
pub fn perceive_table(table: Table, player: Player) -> VisualField {
  let positions = sinuca.get_all_positions(table)
  let cue_pos = sinuca.get_cue_ball_position(table)

  let perceived_balls = case cue_pos {
    Some(cue) -> {
      list.filter_map(positions, fn(ball_pos) {
        let #(color, pos) = ball_pos
        case color {
          sinuca.White -> Error(Nil)  // Don't perceive cue ball as target
          _ -> {
            let dx = pos.x -. cue.x
            let dz = pos.z -. cue.z
            let dist = float_sqrt(dx *. dx +. dz *. dz)
            let angle = float_atan2(dz, dx)

            Ok(PerceivedBall(
              identity: color,
              is_mine: is_my_ball(color, player),
              angle: angle,
              distance: distance_to_perception(dist),
              blocked: False,  // TODO: ray cast for occlusion
              emotional_charge: 0.0,  // TODO: from memory
            ))
          }
        }
      })
    }
    None -> []
  }

  let perceived_pockets = perceive_pockets(cue_pos)

  let ball_count = list.length(perceived_balls)
  let my_balls = list.count(perceived_balls, fn(b) { b.is_mine })
  let their_balls = ball_count - my_balls

  let mood = case my_balls - their_balls {
    diff if diff > 2 -> Favorable
    diff if diff < -2 -> Threatening
    diff if diff < -3 -> Critical
    _ -> Neutral
  }

  VisualField(
    perceived_balls: perceived_balls,
    perceived_pockets: perceived_pockets,
    table_feeling: TableFeeling(
      density: int.to_float(ball_count) /. 15.0,
      freedom: 1.0 -. int.to_float(ball_count) /. 15.0,
      mood: mood,
    ),
  )
}

/// Check if ball belongs to this player
fn is_my_ball(color: BallColor, player: Player) -> Bool {
  let value = sinuca.point_value(color)
  case player {
    P1 -> value % 2 == 1  // Odd balls
    P2 -> value % 2 == 0 && value != 0  // Even balls (not cue)
  }
}

/// Convert metric distance to felt distance
fn distance_to_perception(meters: Float) -> Distance {
  case meters {
    d if d <. 0.2 -> VeryClose
    d if d <. 0.5 -> Close
    d if d <. 1.0 -> Medium
    _ -> Far
  }
}

/// Perceive pockets from cue ball position
fn perceive_pockets(_cue_pos: Option(Vec3)) -> List(PerceivedPocket) {
  // Simplified - just return all 6 with angles
  [
    PerceivedPocket(TopLeft, 2.356, 0.5),
    PerceivedPocket(TopRight, 0.785, 0.5),
    PerceivedPocket(BottomLeft, -2.356, 0.5),
    PerceivedPocket(BottomRight, -0.785, 0.5),
    PerceivedPocket(MiddleTop, 1.571, 0.6),
    PerceivedPocket(MiddleBottom, -1.571, 0.6),
  ]
}

// =============================================================================
// EMOTIONAL RESPONSE - Feel the game
// =============================================================================

/// Process a game event and update emotional state
pub fn feel_event(
  consciousness: SinucaConsciousness,
  event: GameEvent,
) -> SinucaConsciousness {
  let #(pleasure_delta, arousal_delta, dominance_delta) = case event {
    Pocketed(_, True) -> #(0.3, 0.2, 0.1)     // Intended pocket - joy!
    Pocketed(_, False) -> #(0.15, 0.3, 0.05)  // Lucky pocket - surprise
    Missed(_) -> #(-0.2, -0.1, -0.1)          // Disappointment
    Fouled(_) -> #(-0.4, 0.2, -0.2)           // Frustration
    OpponentScored(_) -> #(-0.2, 0.1, -0.15)  // Concern
    CloseCall(_) -> #(0.0, 0.4, 0.0)          // Tension
    GoodPosition -> #(0.1, -0.1, 0.1)         // Satisfaction
    BadPosition -> #(-0.1, 0.1, -0.1)         // Worry
  }

  let old_pad = consciousness.emotion
  let new_pad = Pad(
    pleasure: clamp(old_pad.pleasure +. pleasure_delta, -1.0, 1.0),
    arousal: clamp(old_pad.arousal +. arousal_delta, -1.0, 1.0),
    dominance: clamp(old_pad.dominance +. dominance_delta, -1.0, 1.0),
  )

  let memory = GameMemory(
    event: event,
    emotional_impact: pleasure_delta,
    shots_ago: 0,
  )

  SinucaConsciousness(
    ..consciousness,
    emotion: new_pad,
    recent_memory: [memory, ..list.take(consciousness.recent_memory, 9)],
  )
}

// =============================================================================
// DECISION MAKING - Think about shots
// =============================================================================

/// Generate a thought about the current situation
pub fn think(consciousness: SinucaConsciousness) -> Thought {
  let vf = consciousness.visual_field

  // Find best shot opportunity
  let my_balls = list.filter(vf.perceived_balls, fn(b) { b.is_mine })

  case my_balls {
    [] -> Worried("No balls left to pocket...")
    balls -> {
      // Find most promising ball
      let best = list.fold(balls, #(None, 0.0), fn(acc, ball) {
        let score = score_opportunity(ball, vf.perceived_pockets)
        case score >. acc.1 {
          True -> #(Some(ball), score)
          False -> acc
        }
      })

      case best.0 {
        Some(ball) -> {
          let intention = ShotIntention(
            target: ball.identity,
            desired_pocket: find_best_pocket(ball, vf.perceived_pockets),
            intended_power: power_for_distance(ball.distance),
            intended_spin: NoSpin,
            gut_feeling: best.1,
          )
          Considering(intention)
        }
        None -> Blank
      }
    }
  }
}

/// Score how good an opportunity a ball presents
fn score_opportunity(ball: PerceivedBall, _pockets: List(PerceivedPocket)) -> Float {
  let distance_score = case ball.distance {
    VeryClose -> 1.0
    Close -> 0.8
    Medium -> 0.5
    Far -> 0.3
  }

  let blocked_penalty = case ball.blocked {
    True -> 0.3
    False -> 0.0
  }

  distance_score -. blocked_penalty +. ball.emotional_charge *. 0.2
}

/// Find best pocket for a ball
fn find_best_pocket(
  _ball: PerceivedBall,
  pockets: List(PerceivedPocket),
) -> PocketPosition {
  case list.first(pockets) {
    Ok(p) -> p.position
    Error(_) -> TopLeft
  }
}

/// Determine shot power based on distance
fn power_for_distance(distance: Distance) -> ShotPower {
  case distance {
    VeryClose -> Gentle
    Close -> Normal
    Medium -> Firm
    Far -> Hard
  }
}

// =============================================================================
// INNER VOICE - What VIVA says to herself
// =============================================================================

/// Generate inner monologue based on current state
pub fn inner_voice(consciousness: SinucaConsciousness) -> String {
  let emotion = consciousness.emotion
  let thought = consciousness.current_thought

  // Emotional prefix based on PAD
  let mood_prefix = case emotion.pleasure {
    p if p >. 0.5 -> "Hmm, feeling good... "
    p if p <. -0.5 -> "Ugh... "
    p if p >. 0.0 -> ""
    _ -> "Hmm... "
  }

  let thought_text = case thought {
    Considering(shot) -> {
      let power_word = case shot.intended_power {
        Gentle -> "gentle"
        Normal -> "solid"
        Firm -> "firm"
        Hard -> "powerful"
      }
      "I could try a " <> power_word <> " shot on the " <> ball_name(shot.target)
    }
    Excited(reason) -> "Yes! " <> reason
    Worried(concern) -> concern
    Focused(target) -> "Focus on the " <> ball_name(target) <> "..."
    Reflecting(memory) -> "I remember... " <> describe_memory(memory)
    Blank -> "..."
  }

  mood_prefix <> thought_text
}

/// Ball color to name
fn ball_name(color: BallColor) -> String {
  sinuca.color_name(color)
}

/// Describe a memory
fn describe_memory(memory: GameMemory) -> String {
  case memory.event {
    Pocketed(ball, True) -> "that perfect shot on " <> ball_name(ball)
    Pocketed(ball, False) -> "getting lucky with " <> ball_name(ball)
    Missed(ball) -> "missing " <> ball_name(ball)
    Fouled(reason) -> "that foul: " <> reason
    OpponentScored(ball) -> "when P2 got " <> ball_name(ball)
    CloseCall(desc) -> desc
    GoodPosition -> "ending up in a good spot"
    BadPosition -> "that awkward position"
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn clamp(x: Float, min: Float, max: Float) -> Float {
  case x <. min {
    True -> min
    False -> case x >. max {
      True -> max
      False -> x
    }
  }
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "atan2")
fn float_atan2(y: Float, x: Float) -> Float
