//// Math FFI - Centralized math functions via Erlang FFI
////
//// Use these instead of manual implementations (Taylor series, Newton-Raphson).
//// All functions are O(1) and hardware-optimized.
////
//// Created to eliminate duplicated slow implementations across:
//// - neat.gleam (Taylor exp)
//// - holoneat.gleam (Newton sqrt)
//// - holomap.gleam (Newton sqrt)
//// - neat_hybrid.gleam (Newton sqrt)
//// - novelty.gleam (Newton sqrt)
//// - cma_es.gleam (Newton sqrt)

// =============================================================================
// EXPONENTIAL & LOGARITHM
// =============================================================================

/// Exponential function e^x - O(1) via hardware
@external(erlang, "math", "exp")
pub fn exp(x: Float) -> Float

/// Natural logarithm ln(x) - O(1) via hardware
@external(erlang, "math", "log")
pub fn log(x: Float) -> Float

/// Base-10 logarithm log10(x)
@external(erlang, "math", "log10")
pub fn log10(x: Float) -> Float

/// Base-2 logarithm log2(x)
@external(erlang, "math", "log2")
pub fn log2(x: Float) -> Float

/// Power x^y
@external(erlang, "math", "pow")
pub fn pow(x: Float, y: Float) -> Float

// =============================================================================
// ROOTS
// =============================================================================

/// Square root - O(1) via hardware
@external(erlang, "math", "sqrt")
pub fn sqrt(x: Float) -> Float

// =============================================================================
// TRIGONOMETRIC
// =============================================================================

/// Sine
@external(erlang, "math", "sin")
pub fn sin(x: Float) -> Float

/// Cosine
@external(erlang, "math", "cos")
pub fn cos(x: Float) -> Float

/// Tangent
@external(erlang, "math", "tan")
pub fn tan(x: Float) -> Float

/// Hyperbolic tangent - O(1) via hardware
@external(erlang, "math", "tanh")
pub fn tanh(x: Float) -> Float

/// Hyperbolic sine
@external(erlang, "math", "sinh")
pub fn sinh(x: Float) -> Float

/// Hyperbolic cosine
@external(erlang, "math", "cosh")
pub fn cosh(x: Float) -> Float

// =============================================================================
// INVERSE TRIGONOMETRIC
// =============================================================================

/// Arc sine
@external(erlang, "math", "asin")
pub fn asin(x: Float) -> Float

/// Arc cosine
@external(erlang, "math", "acos")
pub fn acos(x: Float) -> Float

/// Arc tangent
@external(erlang, "math", "atan")
pub fn atan(x: Float) -> Float

/// Arc tangent of y/x (handles quadrants)
@external(erlang, "math", "atan2")
pub fn atan2(y: Float, x: Float) -> Float

// =============================================================================
// CONSTANTS
// =============================================================================

/// Pi constant
pub const pi: Float = 3.14159265358979323846

/// Euler's number e
pub const e: Float = 2.71828182845904523536

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Sigmoid function: 1 / (1 + e^(-x))
pub fn sigmoid(x: Float) -> Float {
  1.0 /. { 1.0 +. exp(0.0 -. x) }
}

/// Safe sqrt that returns 0.0 for negative inputs
pub fn safe_sqrt(x: Float) -> Float {
  case x <=. 0.0 {
    True -> 0.0
    False -> sqrt(x)
  }
}

/// Clipped exp to avoid overflow (returns exp(clipped_x))
pub fn safe_exp(x: Float) -> Float {
  let clamped = case x >. 709.0 {
    True -> 709.0
    False -> case x <. -709.0 {
      True -> -709.0
      False -> x
    }
  }
  exp(clamped)
}
