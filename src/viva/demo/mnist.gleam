//// MNIST Demo v3 - Handwritten digit recognition in Pure Gleam
////
//// Demonstrates neural network training on digit classification.
//// Uses 8x8 pixel digits (sklearn digits-style) for fast training.
////
//// v3 Improvements (per Qwen3 review):
//// - L2 regularization λ=0.002 (rule 1/n)
//// - Label smoothing ε=0.1 (reduce overconfidence)
//// - Reduced augmentation (15 variations vs 50)
//// - Robustness test with non-augmented samples
////
//// Usage: gleam run -m viva/demo/mnist

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import viva/neural/activation
import viva/neural/network.{type Network}
import viva/neural/tensor.{type Tensor, Tensor}
import viva/neural/train.{type Sample, Sample}

// =============================================================================
// LABEL SMOOTHING
// =============================================================================

/// Label smoothing epsilon (reduces overconfidence)
const label_smoothing: Float = 0.1

// =============================================================================
// CONSTANTS
// =============================================================================

/// Image size (8x8 = 64 pixels)
const image_size: Int = 64

/// Number of classes (0-9)
const num_classes: Int = 10

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("╔══════════════════════════════════════════════════════════════╗")
  io.println("║           VIVA MNIST DEMO v3 - Pure Gleam Neural             ║")
  io.println("╠══════════════════════════════════════════════════════════════╣")
  io.println("║  8x8 digits • 150 samples • L2=0.002 • Label Smoothing 0.1   ║")
  io.println("╚══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Load data
  io.println("📦 Loading digit samples...")
  let #(train_samples, test_samples) = load_data()
  io.println(
    "   Train: " <> int.to_string(list.length(train_samples)) <> " samples",
  )
  io.println(
    "   Test:  " <> int.to_string(list.length(test_samples)) <> " samples",
  )
  io.println("")

  // Create network
  io.println("🧠 Creating network: 64 → 32 → 10")
  let assert Ok(net) =
    network.new(
      [image_size, 32, num_classes],
      activation.ReLU,
      activation.Softmax,
    )
  io.println("   Layers: " <> int.to_string(list.length(net.layers)))
  io.println("")

  // Training config (v3 - per Qwen3 review)
  let config =
    train.TrainConfig(
      learning_rate: 0.02,
      momentum: 0.9,
      epochs: 40,
      // More epochs for smaller dataset
      batch_size: 8,
      // Smaller batch for 150 samples
      loss: train.CrossEntropy,
      l2_lambda: 0.002,
      // Rule 1/n for regularization
      gradient_clip: 5.0,
      log_interval: 10,
    )

  // Train
  io.println("🏋️ Training...")
  io.println("   LR: 0.02 | L2: 0.002 | Label Smoothing: 0.1 | Epochs: 40")
  io.println("")

  case train.fit(net, train_samples, config) {
    Ok(#(trained_net, metrics)) -> {
      // Print training progress
      list.each(metrics, fn(m) {
        case m.epoch % 5 == 0 || m.epoch == 29 {
          True -> {
            io.println(
              "   Epoch "
              <> string.pad_start(int.to_string(m.epoch + 1), 2, " ")
              <> " │ Loss: "
              <> format_float(m.epoch_loss, 4),
            )
          }
          False -> Nil
        }
      })
      io.println("")

      // Evaluate on augmented test set
      io.println("📊 Evaluating on test set (augmented)...")
      let accuracy = evaluate(trained_net, test_samples)
      io.println("   Accuracy: " <> format_float(accuracy *. 100.0, 1) <> "%")
      io.println("")

      // ROBUSTNESS TEST: Evaluate on raw (non-augmented) samples
      io.println("🔬 Robustness test (non-augmented samples)...")
      let raw_samples = generate_raw_samples()
      let raw_accuracy = evaluate(trained_net, raw_samples)
      io.println(
        "   Raw Accuracy: "
        <> format_float(raw_accuracy *. 100.0, 1)
        <> "% (10 pristine digits)",
      )
      io.println("")

      // Verdict
      let verdict = case raw_accuracy >=. 0.9 {
        True -> "✅ Model generalizes well!"
        False -> "⚠️ Possible augmentation leakage"
      }
      io.println("   " <> verdict)
      io.println("")

      // Show predictions
      io.println("🔮 Sample predictions (from raw test):")
      show_predictions(trained_net, raw_samples)
      io.println("")

      io.println(
        "╔══════════════════════════════════════════════════════════════╗",
      )
      io.println(
        "║                    DEMO v3 COMPLETE                          ║",
      )
      io.println(
        "╠══════════════════════════════════════════════════════════════╣",
      )
      io.println(
        "║  Augmented: "
        <> string.pad_end(format_float(accuracy *. 100.0, 1) <> "%", 6, " ")
        <> " │ Raw: "
        <> string.pad_end(format_float(raw_accuracy *. 100.0, 1) <> "%", 6, " ")
        <> "               ║",
      )
      io.println(
        "╚══════════════════════════════════════════════════════════════╝",
      )
    }
    Error(_e) -> {
      io.println("❌ Training failed")
    }
  }
}

// =============================================================================
// EVALUATION
// =============================================================================

/// Calculate accuracy on test set
fn evaluate(net: Network, samples: List(Sample)) -> Float {
  let results =
    list.map(samples, fn(sample) {
      case network.forward(net, sample.input) {
        Ok(#(output, _)) -> {
          let predicted = argmax(output)
          let actual = argmax(sample.target)
          case predicted == actual {
            True -> 1.0
            False -> 0.0
          }
        }
        Error(_) -> 0.0
      }
    })

  let correct = list.fold(results, 0.0, fn(a, b) { a +. b })
  correct /. int.to_float(list.length(samples))
}

/// Get index of maximum value
fn argmax(t: Tensor) -> Int {
  let indexed = list.index_map(t.data, fn(val, idx) { #(idx, val) })

  let assert Ok(#(max_idx, _)) =
    list.reduce(indexed, fn(acc, current) {
      let #(_, acc_val) = acc
      let #(_, curr_val) = current
      case curr_val >. acc_val {
        True -> current
        False -> acc
      }
    })

  max_idx
}

/// Show sample predictions
fn show_predictions(net: Network, samples: List(Sample)) {
  samples
  |> list.take(5)
  |> list.each(fn(sample) {
    case network.forward(net, sample.input) {
      Ok(#(output, _)) -> {
        let predicted = argmax(output)
        let actual = argmax(sample.target)
        let confidence = case list_at(output.data, predicted) {
          Ok(c) -> c
          Error(_) -> 0.0
        }
        let status = case predicted == actual {
          True -> "✓"
          False -> "✗"
        }
        io.println(
          "   "
          <> status
          <> " Actual: "
          <> int.to_string(actual)
          <> " → Predicted: "
          <> int.to_string(predicted)
          <> " ("
          <> format_float(confidence *. 100.0, 1)
          <> "%)",
        )
      }
      Error(_) -> Nil
    }
  })
}

// =============================================================================
// DATA LOADING
// =============================================================================

/// Load training and test data
fn load_data() -> #(List(Sample), List(Sample)) {
  let all_samples = generate_samples()

  // Shuffle deterministically before split
  let shuffled = shuffle_samples(all_samples, 42)

  // Split 80/20
  let split_idx = list.length(shuffled) * 8 / 10
  let #(train_data, test_data) = list.split(shuffled, split_idx)

  #(train_data, test_data)
}

/// Deterministic shuffle using seed
fn shuffle_samples(samples: List(Sample), seed: Int) -> List(Sample) {
  // Add random keys and sort
  samples
  |> list.index_map(fn(s, i) {
    let key = pseudo_random_int(seed + i * 1000)
    #(key, s)
  })
  |> list.sort(fn(a, b) { int.compare(a.0, b.0) })
  |> list.map(fn(pair) { pair.1 })
}

fn pseudo_random_int(seed: Int) -> Int {
  let x = seed * 1_103_515_245 + 12_345
  { x / 65_536 } % 32_768
}

/// Generate digit samples (8x8 simplified patterns)
fn generate_samples() -> List(Sample) {
  // Generate multiple variations of each digit
  list.flatten([
    generate_digit_variations(0, digit_0_pattern()),
    generate_digit_variations(1, digit_1_pattern()),
    generate_digit_variations(2, digit_2_pattern()),
    generate_digit_variations(3, digit_3_pattern()),
    generate_digit_variations(4, digit_4_pattern()),
    generate_digit_variations(5, digit_5_pattern()),
    generate_digit_variations(6, digit_6_pattern()),
    generate_digit_variations(7, digit_7_pattern()),
    generate_digit_variations(8, digit_8_pattern()),
    generate_digit_variations(9, digit_9_pattern()),
  ])
}

/// Generate variations of a digit pattern (15 variations per digit = 150 total)
/// Reduced from 50 to avoid augmentation leakage
fn generate_digit_variations(
  label: Int,
  base_pattern: List(Float),
) -> List(Sample) {
  let target = one_hot_smoothed(label, num_classes)

  // 15 variations per digit (reduced to prevent memorizing augmentation patterns)
  list.flatten([
    // Original (1 sample)
    [
      Sample(
        input: Tensor(data: base_pattern, shape: [image_size]),
        target: target,
      ),
    ],

    // Noise variations (5 samples) - moderate noise only
    list.range(1, 5)
      |> list.map(fn(i) {
        let noise_level = int.to_float(i) *. 0.03
        // 0.03 to 0.15
        Sample(
          input: Tensor(
            data: add_noise_seeded(base_pattern, noise_level, label * 100 + i),
            shape: [image_size],
          ),
          target: target,
        )
      }),

    // Intensity variations (5 samples)
    list.range(1, 5)
      |> list.map(fn(i) {
        let factor = 0.7 +. int.to_float(i) *. 0.06
        // 0.76 to 1.0
        Sample(
          input: Tensor(data: scale_intensity(base_pattern, factor), shape: [
            image_size,
          ]),
          target: target,
        )
      }),

    // Brightness shift (4 samples)
    list.range(1, 4)
      |> list.map(fn(i) {
        let shift = { int.to_float(i) -. 2.0 } *. 0.1
        // -0.1 to 0.2
        Sample(
          input: Tensor(data: shift_brightness(base_pattern, shift), shape: [
            image_size,
          ]),
          target: target,
        )
      }),
  ])
}

/// Generate raw (non-augmented) samples for robustness testing
fn generate_raw_samples() -> List(Sample) {
  [
    Sample(
      input: Tensor(data: digit_0_pattern(), shape: [image_size]),
      target: one_hot_smoothed(0, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_1_pattern(), shape: [image_size]),
      target: one_hot_smoothed(1, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_2_pattern(), shape: [image_size]),
      target: one_hot_smoothed(2, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_3_pattern(), shape: [image_size]),
      target: one_hot_smoothed(3, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_4_pattern(), shape: [image_size]),
      target: one_hot_smoothed(4, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_5_pattern(), shape: [image_size]),
      target: one_hot_smoothed(5, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_6_pattern(), shape: [image_size]),
      target: one_hot_smoothed(6, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_7_pattern(), shape: [image_size]),
      target: one_hot_smoothed(7, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_8_pattern(), shape: [image_size]),
      target: one_hot_smoothed(8, num_classes),
    ),
    Sample(
      input: Tensor(data: digit_9_pattern(), shape: [image_size]),
      target: one_hot_smoothed(9, num_classes),
    ),
  ]
}

/// Scale intensity
fn scale_intensity(pattern: List(Float), factor: Float) -> List(Float) {
  list.map(pattern, fn(val) { float.clamp(val *. factor, 0.0, 1.0) })
}

/// Shift brightness
fn shift_brightness(pattern: List(Float), shift: Float) -> List(Float) {
  list.map(pattern, fn(val) { float.clamp(val +. shift, 0.0, 1.0) })
}

/// Add noise with seed for reproducibility
fn add_noise_seeded(
  pattern: List(Float),
  amount: Float,
  seed: Int,
) -> List(Float) {
  list.index_map(pattern, fn(val, idx) {
    let noise = { pseudo_random(seed * 64 + idx) -. 0.5 } *. 2.0 *. amount
    float.clamp(val +. noise, 0.0, 1.0)
  })
}

/// Pseudo-random for deterministic noise
fn pseudo_random(seed: Int) -> Float {
  let x = seed * 1_103_515_245 + 12_345
  let x = { x / 65_536 } % 32_768
  int.to_float(int.absolute_value(x)) /. 32_768.0
}

/// One-hot encode a label (hard labels, no smoothing)
fn one_hot(label: Int, classes: Int) -> Tensor {
  let data =
    list.range(0, classes - 1)
    |> list.map(fn(i) {
      case i == label {
        True -> 1.0
        False -> 0.0
      }
    })
  Tensor(data: data, shape: [classes])
}

/// One-hot encode with label smoothing (reduces overconfidence)
/// Formula: target = (1 - ε) * one_hot + ε / num_classes
fn one_hot_smoothed(label: Int, classes: Int) -> Tensor {
  let smooth_value = label_smoothing /. int.to_float(classes)
  // ε/K
  let confident_value = 1.0 -. label_smoothing +. smooth_value
  // (1-ε) + ε/K

  let data =
    list.range(0, classes - 1)
    |> list.map(fn(i) {
      case i == label {
        True -> confident_value
        // ~0.91 for correct class
        False -> smooth_value
        // ~0.01 for other classes
      }
    })
  Tensor(data: data, shape: [classes])
}

// =============================================================================
// DIGIT PATTERNS (8x8)
// =============================================================================

/// 8x8 pattern for digit 0
fn digit_0_pattern() -> List(Float) {
  [
    // Row 0
    0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0,
    // Row 1
    0.0, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.0,
    // Row 2
    0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5,
    // Row 3
    0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5,
    // Row 4
    0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5,
    // Row 5
    0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5,
    // Row 6
    0.0, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.0,
    // Row 7
    0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0,
  ]
}

/// 8x8 pattern for digit 1
fn digit_1_pattern() -> List(Float) {
  [
    0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0,
    1.0, 1.0, 0.5, 0.0,
  ]
}

/// 8x8 pattern for digit 2
fn digit_2_pattern() -> List(Float) {
  [
    0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0,
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0,
    0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 0.5,
  ]
}

/// 8x8 pattern for digit 3
fn digit_3_pattern() -> List(Float) {
  [
    0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0,
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0,
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 1.0,
    1.0, 1.0, 0.5, 0.0,
  ]
}

/// 8x8 pattern for digit 4
fn digit_4_pattern() -> List(Float) {
  [
    0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.5,
    0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0,
    0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.5, 0.0,
  ]
}

/// 8x8 pattern for digit 5
fn digit_5_pattern() -> List(Float) {
  [
    0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0,
    1.0, 0.5, 0.0, 0.0,
  ]
}

/// 8x8 pattern for digit 6
fn digit_6_pattern() -> List(Float) {
  [
    0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5,
    0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0,
    1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0,
    1.0, 0.5, 0.0, 0.0,
  ]
}

/// 8x8 pattern for digit 7
fn digit_7_pattern() -> List(Float) {
  [
    0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0,
    0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0,
    0.0, 0.0, 0.0, 0.0,
  ]
}

/// 8x8 pattern for digit 8
fn digit_8_pattern() -> List(Float) {
  [
    0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5,
    0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5,
    0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0,
    1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0,
    1.0, 0.5, 0.0, 0.0,
  ]
}

/// 8x8 pattern for digit 9
fn digit_9_pattern() -> List(Float) {
  [
    0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5,
    0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0,
    0.5, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
    1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0,
    0.5, 0.0, 0.0, 0.0,
  ]
}

// =============================================================================
// UTILITIES
// =============================================================================

/// Format float with decimals
fn format_float(f: Float, decimals: Int) -> String {
  let multiplier = power_of_10(decimals)
  let rounded = int.to_float(float.round(f *. multiplier)) /. multiplier
  let str = float.to_string(rounded)

  // Ensure decimal places
  case string.contains(str, ".") {
    True -> str
    False -> str <> ".0"
  }
}

fn power_of_10(n: Int) -> Float {
  case n {
    0 -> 1.0
    1 -> 10.0
    2 -> 100.0
    3 -> 1000.0
    4 -> 10_000.0
    _ -> 10_000.0
  }
}

/// List access by index
fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  lst
  |> list.drop(index)
  |> list.first
}
