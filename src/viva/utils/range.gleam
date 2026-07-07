/// Inclusive range helper used across modules.
///
/// Unlike `int.range`, this function includes both bounds.
import gleam/int
import gleam/list

/// Inclusive range [from..to] with support for descending ranges.
pub fn range_inclusive(from: Int, to: Int) -> List(Int) {
  let upper = case from <= to {
    True -> to + 1
    False -> to - 1
  }

  int.range(from, upper, [], fn(acc, i) { [i, ..acc] })
  |> list.reverse
}
