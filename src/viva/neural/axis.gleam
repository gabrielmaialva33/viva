//// Axis - Named axis types and constructors
////
//// Semantic dimension names for tensors.

// =============================================================================
// AXIS TYPES - Semantic dimension names
// =============================================================================

/// Named axis - gives semantic meaning to a dimension
pub type Axis {
  /// Batch dimension (samples in mini-batch)
  Batch
  /// Sequence/time dimension (for RNNs, transformers)
  Seq
  /// Feature/channel dimension
  Feature
  /// Spatial height
  Height
  /// Spatial width
  Width
  /// Input dimension (for weight matrices)
  Input
  /// Output dimension (for weight matrices)
  Output
  /// Head dimension (for multi-head attention)
  Head
  /// Embedding dimension
  Embed
  /// Custom named axis
  Named(String)
  /// Anonymous axis (unnamed, referenced by position)
  Anon
}

/// Axis with its size
pub type AxisSpec {
  AxisSpec(name: Axis, size: Int)
}

// =============================================================================
// AXIS SPEC CONSTRUCTORS
// =============================================================================

/// Create axis spec with shorthand
pub fn axis(name: Axis, size: Int) -> AxisSpec {
  AxisSpec(name: name, size: size)
}

/// Batch dimension
pub fn batch(size: Int) -> AxisSpec {
  AxisSpec(name: Batch, size: size)
}

/// Sequence dimension
pub fn seq(size: Int) -> AxisSpec {
  AxisSpec(name: Seq, size: size)
}

/// Feature dimension
pub fn feature(size: Int) -> AxisSpec {
  AxisSpec(name: Feature, size: size)
}

/// Height dimension
pub fn height(size: Int) -> AxisSpec {
  AxisSpec(name: Height, size: size)
}

/// Width dimension
pub fn width(size: Int) -> AxisSpec {
  AxisSpec(name: Width, size: size)
}

/// Input dimension
pub fn input(size: Int) -> AxisSpec {
  AxisSpec(name: Input, size: size)
}

/// Output dimension
pub fn output(size: Int) -> AxisSpec {
  AxisSpec(name: Output, size: size)
}

/// Head dimension
pub fn head(size: Int) -> AxisSpec {
  AxisSpec(name: Head, size: size)
}

/// Embed dimension
pub fn embed(size: Int) -> AxisSpec {
  AxisSpec(name: Embed, size: size)
}

/// Custom named dimension
pub fn named(name: String, size: Int) -> AxisSpec {
  AxisSpec(name: Named(name), size: size)
}

// =============================================================================
// AXIS UTILITIES
// =============================================================================

/// Check if two axes are equal
pub fn axis_equals(a: Axis, b: Axis) -> Bool {
  case a, b {
    Anon, Anon -> True
    Batch, Batch -> True
    Seq, Seq -> True
    Feature, Feature -> True
    Height, Height -> True
    Width, Width -> True
    Input, Input -> True
    Output, Output -> True
    Head, Head -> True
    Embed, Embed -> True
    Named(s1), Named(s2) -> s1 == s2
    _, _ -> False
  }
}

/// Check if two axis spec lists are equal
pub fn axes_equal(a: List(AxisSpec), b: List(AxisSpec)) -> Bool {
  case a, b {
    [], [] -> True
    [a1, ..a_rest], [b1, ..b_rest] ->
      axis_equals(a1.name, b1.name)
      && a1.size == b1.size
      && axes_equal(a_rest, b_rest)
    _, _ -> False
  }
}

/// Get human-readable axis name
pub fn axis_to_string(a: Axis) -> String {
  case a {
    Batch -> "batch"
    Seq -> "seq"
    Feature -> "feature"
    Height -> "height"
    Width -> "width"
    Input -> "input"
    Output -> "output"
    Head -> "head"
    Embed -> "embed"
    Named(s) -> s
    Anon -> "_"
  }
}
