//// Vec4 - 4D Vector for semantic space (PAD + Intensity)

import gleam/float

/// 4D vector: Pleasure, Arousal, Dominance, Intensity
pub type Vec4 {
  Vec4(x: Float, y: Float, z: Float, w: Float)
}

/// Zero vector
pub const zero = Vec4(0.0, 0.0, 0.0, 0.0)

/// One vector
pub const one = Vec4(1.0, 1.0, 1.0, 1.0)

/// Create from PAD + intensity
pub fn from_pad(p: Float, a: Float, d: Float, i: Float) -> Vec4 {
  Vec4(x: p, y: a, z: d, w: i)
}

/// Add two vectors
pub fn add(a: Vec4, b: Vec4) -> Vec4 {
  Vec4(a.x +. b.x, a.y +. b.y, a.z +. b.z, a.w +. b.w)
}

/// Subtract vectors
pub fn sub(a: Vec4, b: Vec4) -> Vec4 {
  Vec4(a.x -. b.x, a.y -. b.y, a.z -. b.z, a.w -. b.w)
}

/// Scale vector
pub fn scale(v: Vec4, s: Float) -> Vec4 {
  Vec4(v.x *. s, v.y *. s, v.z *. s, v.w *. s)
}

/// Dot product
pub fn dot(a: Vec4, b: Vec4) -> Float {
  a.x *. b.x +. a.y *. b.y +. a.z *. b.z +. a.w *. b.w
}

/// Length squared (avoid sqrt when possible)
pub fn length_sq(v: Vec4) -> Float {
  dot(v, v)
}

/// Length
pub fn length(v: Vec4) -> Float {
  float_sqrt(length_sq(v))
}

/// Distance squared between two points
pub fn distance_sq(a: Vec4, b: Vec4) -> Float {
  length_sq(sub(a, b))
}

/// Distance between two points
pub fn distance(a: Vec4, b: Vec4) -> Float {
  float_sqrt(distance_sq(a, b))
}

/// Normalize to unit length
pub fn normalize(v: Vec4) -> Vec4 {
  let len = length(v)
  case len <. 0.0001 {
    True -> zero
    False -> scale(v, 1.0 /. len)
  }
}

/// Linear interpolation
pub fn lerp(a: Vec4, b: Vec4, t: Float) -> Vec4 {
  add(scale(a, 1.0 -. t), scale(b, t))
}

/// Clamp each component
pub fn clamp(v: Vec4, min: Vec4, max: Vec4) -> Vec4 {
  Vec4(
    x: float.min(float.max(v.x, min.x), max.x),
    y: float.min(float.max(v.y, min.y), max.y),
    z: float.min(float.max(v.z, min.z), max.z),
    w: float.min(float.max(v.w, min.w), max.w),
  )
}

/// Clamp to PAD bounds (x,y,z: -1 to 1, w: 0 to 1)
pub fn clamp_pad(v: Vec4) -> Vec4 {
  clamp(v, Vec4(-1.0, -1.0, -1.0, 0.0), Vec4(1.0, 1.0, 1.0, 1.0))
}

/// Component-wise min
pub fn min(a: Vec4, b: Vec4) -> Vec4 {
  Vec4(
    float.min(a.x, b.x),
    float.min(a.y, b.y),
    float.min(a.z, b.z),
    float.min(a.w, b.w),
  )
}

/// Component-wise max
pub fn max(a: Vec4, b: Vec4) -> Vec4 {
  Vec4(
    float.max(a.x, b.x),
    float.max(a.y, b.y),
    float.max(a.z, b.z),
    float.max(a.w, b.w),
  )
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float
