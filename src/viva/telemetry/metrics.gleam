//// Telemetry Metrics
////
//// Aggregated metrics for World state monitoring.
//// Useful for dashboards and performance analysis.

import gleam/dict
import gleam/float
import gleam/json.{type Json}
import gleam/list
import viva/memory/body.{type Body}
import viva/memory/hrr
import viva/memory/world.{type World}
import viva_tensor/tensor.{type Tensor}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// Snapshot of world metrics at a point in time
pub type WorldMetrics {
  WorldMetrics(
    tick: Int,
    total_bodies: Int,
    awake_bodies: Int,
    sleeping_bodies: Int,
    static_bodies: Int,
    dynamic_bodies: Int,
    island_count: Int,
    avg_energy: Float,
    max_energy: Float,
    min_energy: Float,
    avg_velocity_magnitude: Float,
    centroid: List(Float),
    spread: Float,
  )
}

/// Time-series metrics for trends
pub type MetricsHistory {
  MetricsHistory(samples: List(WorldMetrics), max_samples: Int)
}

// =============================================================================
// METRICS COLLECTION
// =============================================================================

/// Collect metrics from World state
pub fn collect(w: World) -> WorldMetrics {
  let bodies_list = dict.to_list(w.bodies) |> list.map(fn(p) { p.1 })
  let total = list.length(bodies_list)

  let awake = list.filter(bodies_list, fn(b) { !b.sleeping }) |> list.length
  let static_count =
    list.filter(bodies_list, fn(b) { body.is_static(b) }) |> list.length
  let dynamic_count = total - static_count

  let energies = list.map(bodies_list, fn(b) { b.energy })
  let avg_e = safe_avg(energies)
  let max_e = safe_max(energies)
  let min_e = safe_min(energies)

  let velocities = list.map(bodies_list, fn(b) { velocity_magnitude(b) })
  let avg_vel = safe_avg(velocities)

  let centroid = compute_centroid(bodies_list)
  let spread = compute_spread(bodies_list, centroid)

  WorldMetrics(
    tick: w.tick,
    total_bodies: total,
    awake_bodies: awake,
    sleeping_bodies: total - awake,
    static_bodies: static_count,
    dynamic_bodies: dynamic_count,
    island_count: list.length(w.islands),
    avg_energy: avg_e,
    max_energy: max_e,
    min_energy: min_e,
    avg_velocity_magnitude: avg_vel,
    centroid: centroid,
    spread: spread,
  )
}

// =============================================================================
// HISTORY MANAGEMENT
// =============================================================================

/// Create new metrics history
pub fn new_history(max_samples: Int) -> MetricsHistory {
  MetricsHistory(samples: [], max_samples: max_samples)
}

/// Add sample to history (FIFO, drops oldest if full)
pub fn add_sample(
  history: MetricsHistory,
  metrics: WorldMetrics,
) -> MetricsHistory {
  let new_samples = [metrics, ..history.samples]
  let trimmed = case list.length(new_samples) > history.max_samples {
    True -> list.take(new_samples, history.max_samples)
    False -> new_samples
  }
  MetricsHistory(..history, samples: trimmed)
}

/// Get trend (last N samples)
pub fn get_trend(history: MetricsHistory, n: Int) -> List(WorldMetrics) {
  list.take(history.samples, n) |> list.reverse
}

// =============================================================================
// JSON SERIALIZATION
// =============================================================================

/// Metrics to JSON
pub fn metrics_to_json(m: WorldMetrics) -> Json {
  json.object([
    #("tick", json.int(m.tick)),
    #(
      "bodies",
      json.object([
        #("total", json.int(m.total_bodies)),
        #("awake", json.int(m.awake_bodies)),
        #("sleeping", json.int(m.sleeping_bodies)),
        #("static", json.int(m.static_bodies)),
        #("dynamic", json.int(m.dynamic_bodies)),
      ]),
    ),
    #("islands", json.int(m.island_count)),
    #(
      "energy",
      json.object([
        #("avg", json.float(m.avg_energy)),
        #("max", json.float(m.max_energy)),
        #("min", json.float(m.min_energy)),
      ]),
    ),
    #("velocity_avg", json.float(m.avg_velocity_magnitude)),
    #("centroid", json.array(m.centroid, json.float)),
    #("spread", json.float(m.spread)),
  ])
}

/// Metrics to JSON string
pub fn metrics_to_string(m: WorldMetrics) -> String {
  metrics_to_json(m) |> json.to_string
}

/// History trend to JSON
pub fn trend_to_json(samples: List(WorldMetrics)) -> Json {
  json.array(samples, metrics_to_json)
}

// =============================================================================
// SIMILARITY METRICS
// =============================================================================

/// Compute similarity matrix between N most active bodies
pub fn similarity_matrix(w: World, n: Int) -> List(List(Float)) {
  let top_bodies =
    dict.to_list(w.bodies)
    |> list.map(fn(p) { p.1 })
    |> list.filter(fn(b) { !b.sleeping })
    |> list.sort(fn(a, b) { float.compare(b.energy, a.energy) })
    |> list.take(n)

  list.map(top_bodies, fn(a) {
    list.map(top_bodies, fn(b) { hrr.similarity(a.shape, b.shape) })
  })
}

/// Similarity matrix to JSON (for D3.js heatmap)
pub fn similarity_matrix_to_json(w: World, n: Int) -> Json {
  let matrix = similarity_matrix(w, n)
  let bodies =
    dict.to_list(w.bodies)
    |> list.map(fn(p) { p.1 })
    |> list.filter(fn(b) { !b.sleeping })
    |> list.sort(fn(a, b) { float.compare(b.energy, a.energy) })
    |> list.take(n)

  json.object([
    #("labels", json.array(bodies, fn(b) { json.string(b.label) })),
    #("matrix", json.array(matrix, fn(row) { json.array(row, json.float) })),
  ])
}

// =============================================================================
// GRAPH DATA (for D3.js force layout)
// =============================================================================

/// Generate nodes and links for D3.js force graph
/// Links are created between bodies with similarity > threshold
pub fn to_force_graph(w: World, similarity_threshold: Float) -> Json {
  let bodies_list = dict.to_list(w.bodies)

  // Nodes
  let nodes =
    list.map(bodies_list, fn(pair) {
      let #(id, b) = pair
      json.object([
        #("id", json.int(id)),
        #("label", json.string(b.label)),
        #("energy", json.float(b.energy)),
        #("sleeping", json.bool(b.sleeping)),
        #("group", json.int(b.island_id)),
        #("x", json.float(list_get(td(b.position), 0, 0.0))),
        #("y", json.float(list_get(td(b.position), 1, 0.0))),
        #("z", json.float(list_get(td(b.position), 2, 0.0))),
      ])
    })

  // Links (similarity-based)
  let links = build_similarity_links(bodies_list, similarity_threshold)

  json.object([
    #("nodes", json.preprocessed_array(nodes)),
    #("links", json.preprocessed_array(links)),
  ])
}

fn build_similarity_links(
  bodies: List(#(Int, Body)),
  threshold: Float,
) -> List(Json) {
  list.flat_map(bodies, fn(pair_a) {
    let #(id_a, body_a) = pair_a
    list.filter_map(bodies, fn(pair_b) {
      let #(id_b, body_b) = pair_b
      case id_a < id_b {
        True -> {
          let sim = hrr.similarity(body_a.shape, body_b.shape)
          case sim >. threshold {
            True ->
              Ok(
                json.object([
                  #("source", json.int(id_a)),
                  #("target", json.int(id_b)),
                  #("value", json.float(sim)),
                ]),
              )
            False -> Error(Nil)
          }
        }
        False -> Error(Nil)
      }
    })
  })
}

// =============================================================================
// HELPERS
// =============================================================================

fn velocity_magnitude(b: Body) -> Float {
  let sum = list.fold(td(b.velocity), 0.0, fn(acc, x) { acc +. x *. x })
  float_sqrt(sum)
}

fn compute_centroid(bodies: List(Body)) -> List(Float) {
  case bodies {
    [] -> [0.0, 0.0, 0.0, 0.0]
    _ -> {
      let n = int_to_float(list.length(bodies))
      let sum =
        list.fold(bodies, [], fn(acc, b) {
          case acc {
            [] -> td(b.position)
            _ -> list.map2(acc, td(b.position), fn(a, x) { a +. x })
          }
        })
      list.map(sum, fn(x) { x /. n })
    }
  }
}

fn compute_spread(bodies: List(Body), centroid: List(Float)) -> Float {
  case bodies {
    [] -> 0.0
    _ -> {
      let distances =
        list.map(bodies, fn(b) {
          let diff = list.map2(td(b.position), centroid, fn(a, c) { a -. c })
          let sq_sum = list.fold(diff, 0.0, fn(acc, x) { acc +. x *. x })
          float_sqrt(sq_sum)
        })
      safe_avg(distances)
    }
  }
}

fn safe_avg(values: List(Float)) -> Float {
  case values {
    [] -> 0.0
    _ -> {
      let sum = list.fold(values, 0.0, fn(acc, x) { acc +. x })
      sum /. int_to_float(list.length(values))
    }
  }
}

fn safe_max(values: List(Float)) -> Float {
  list.fold(values, 0.0, fn(acc, x) {
    case x >. acc {
      True -> x
      False -> acc
    }
  })
}

fn safe_min(values: List(Float)) -> Float {
  case values {
    [] -> 0.0
    [first, ..rest] ->
      list.fold(rest, first, fn(acc, x) {
        case x <. acc {
          True -> x
          False -> acc
        }
      })
  }
}

fn list_get(l: List(Float), idx: Int, default: Float) -> Float {
  case list.drop(l, idx) {
    [x, ..] -> x
    [] -> default
  }
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "erlang", "float")
fn int_to_float(i: Int) -> Float
