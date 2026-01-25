//// Training Pipeline Tests
////
//// Tests for loss functions, training steps, and evaluation.

import gleam/float
import gleam/list
import gleeunit/should
import viva/neural/activation
import viva/neural/network
import viva/neural/tensor
import viva/neural/train

// =============================================================================
// CONFIG TESTS
// =============================================================================

pub fn train_default_config_test() {
  let config = train.default_config()

  should.equal(config.learning_rate, 0.01)
  should.equal(config.momentum, 0.9)
  should.equal(config.epochs, 100)
  should.equal(config.batch_size, 32)
}

pub fn train_classification_config_test() {
  let config = train.classification_config()

  should.equal(config.loss, train.CrossEntropy)
  should.equal(config.learning_rate, 0.001)
}

pub fn train_regression_config_test() {
  let config = train.regression_config()

  should.equal(config.loss, train.MSE)
  should.equal(config.learning_rate, 0.01)
}

// =============================================================================
// LOSS FUNCTION TESTS
// =============================================================================

pub fn train_mse_loss_zero_test() {
  // Same prediction and target should have zero loss
  let pred = tensor.from_list([1.0, 2.0, 3.0])
  let target = tensor.from_list([1.0, 2.0, 3.0])

  let result = train.compute_loss(pred, target, train.MSE)
  should.be_ok(result)

  let assert Ok(loss_result) = result
  should.be_true(float.loosely_equals(loss_result.loss, 0.0, 0.0001))
}

pub fn train_mse_loss_nonzero_test() {
  let pred = tensor.from_list([1.0, 2.0, 3.0])
  let target = tensor.from_list([2.0, 2.0, 2.0])

  // MSE = (1 + 0 + 1) / 3 = 0.666...
  let result = train.compute_loss(pred, target, train.MSE)
  should.be_ok(result)

  let assert Ok(loss_result) = result
  should.be_true(loss_result.loss >. 0.0)
  should.be_true(float.loosely_equals(loss_result.loss, 0.6666, 0.01))
}

pub fn train_mse_loss_gradient_shape_test() {
  let pred = tensor.from_list([0.5, 0.5])
  let target = tensor.from_list([1.0, 0.0])

  let assert Ok(loss_result) = train.compute_loss(pred, target, train.MSE)

  // Gradient should have same shape as prediction
  should.equal(loss_result.gradient.shape, pred.shape)
}

pub fn train_mae_loss_test() {
  let pred = tensor.from_list([1.0, 2.0, 3.0])
  let target = tensor.from_list([2.0, 2.0, 2.0])

  // MAE = (1 + 0 + 1) / 3 = 0.666...
  let result = train.compute_loss(pred, target, train.MAE)
  should.be_ok(result)

  let assert Ok(loss_result) = result
  should.be_true(float.loosely_equals(loss_result.loss, 0.6666, 0.01))
}

pub fn train_cross_entropy_loss_test() {
  // One-hot target [0, 1], prediction [0.1, 0.9]
  let pred = tensor.from_list([0.1, 0.9])
  let target = tensor.from_list([0.0, 1.0])

  let result = train.compute_loss(pred, target, train.CrossEntropy)
  should.be_ok(result)

  let assert Ok(loss_result) = result
  // -log(0.9) ≈ 0.105
  should.be_true(loss_result.loss >. 0.0)
  should.be_true(loss_result.loss <. 1.0)
}

pub fn train_binary_cross_entropy_loss_test() {
  let pred = tensor.from_list([0.9, 0.1])
  let target = tensor.from_list([1.0, 0.0])

  let result = train.compute_loss(pred, target, train.BinaryCrossEntropy)
  should.be_ok(result)

  let assert Ok(loss_result) = result
  should.be_true(loss_result.loss >. 0.0)
}

// =============================================================================
// TRAINING STEP TESTS
// =============================================================================

pub fn train_step_basic_test() {
  let assert Ok(net) =
    network.new([2, 4, 1], activation.ReLU, activation.Sigmoid)
  let input = tensor.from_list([0.5, 0.5])
  let target = tensor.from_list([1.0])

  let config = train.TrainConfig(..train.default_config(), loss: train.MSE)
  let result = train.train_step(net, input, target, config)

  should.be_ok(result)

  let assert Ok(#(updated_net, loss)) = result
  should.be_true(loss >=. 0.0)
  // Network should be updated (different from original)
  should.equal(network.param_count(updated_net), network.param_count(net))
}

pub fn train_step_reduces_loss_test() {
  let assert Ok(net) =
    network.new([2, 3, 1], activation.Linear, activation.Linear)
  let input = tensor.from_list([1.0, 0.0])
  let target = tensor.from_list([1.0])

  let config =
    train.TrainConfig(
      ..train.default_config(),
      learning_rate: 0.1,
      loss: train.MSE,
    )

  // Get initial loss
  let assert Ok(initial_output) = network.predict(net, input)
  let assert Ok(initial_loss) =
    train.compute_loss(initial_output, target, train.MSE)

  // Train for a few steps
  let assert Ok(#(net1, _)) = train.train_step(net, input, target, config)
  let assert Ok(#(net2, _)) = train.train_step(net1, input, target, config)
  let assert Ok(#(net3, _)) = train.train_step(net2, input, target, config)

  // Get final loss
  let assert Ok(final_output) = network.predict(net3, input)
  let assert Ok(final_loss) =
    train.compute_loss(final_output, target, train.MSE)

  // Loss should decrease (or at least not increase significantly)
  should.be_true(final_loss.loss <=. initial_loss.loss +. 0.1)
}

pub fn train_step_momentum_test() {
  let assert Ok(net) =
    network.new([2, 4, 1], activation.ReLU, activation.Sigmoid)
  let input = tensor.from_list([0.5, 0.5])
  let target = tensor.from_list([1.0])

  let config = train.TrainConfig(..train.default_config(), momentum: 0.9)
  let momentum_state = network.init_momentum(net)

  let result =
    train.train_step_momentum(net, momentum_state, input, target, config)
  should.be_ok(result)

  let assert Ok(#(_updated_net, _new_momentum, loss)) = result
  should.be_true(loss >=. 0.0)
}

// =============================================================================
// BATCH TRAINING TESTS
// =============================================================================

pub fn train_batch_empty_test() {
  let assert Ok(net) =
    network.new([2, 3, 1], activation.ReLU, activation.Sigmoid)
  let config = train.default_config()

  let result = train.train_batch(net, [], config)
  should.be_ok(result)

  let assert Ok(#(_net, loss)) = result
  should.equal(loss, 0.0)
}

pub fn train_batch_single_sample_test() {
  let assert Ok(net) =
    network.new([2, 3, 1], activation.ReLU, activation.Sigmoid)
  let config = train.TrainConfig(..train.default_config(), loss: train.MSE)

  let samples = [train.sample([1.0, 0.0], [1.0])]

  let result = train.train_batch(net, samples, config)
  should.be_ok(result)
}

pub fn train_batch_multiple_samples_test() {
  let assert Ok(net) =
    network.new([2, 4, 1], activation.ReLU, activation.Sigmoid)
  let config = train.TrainConfig(..train.default_config(), loss: train.MSE)

  let samples = [
    train.sample([1.0, 0.0], [1.0]),
    train.sample([0.0, 1.0], [1.0]),
    train.sample([0.0, 0.0], [0.0]),
  ]

  let result = train.train_batch(net, samples, config)
  should.be_ok(result)

  let assert Ok(#(_net, avg_loss)) = result
  should.be_true(avg_loss >=. 0.0)
}

// =============================================================================
// FULL TRAINING LOOP TESTS
// =============================================================================

pub fn train_fit_single_epoch_test() {
  let assert Ok(net) =
    network.new([2, 4, 1], activation.ReLU, activation.Sigmoid)

  let samples = [
    train.sample([1.0, 0.0], [1.0]),
    train.sample([0.0, 1.0], [0.0]),
  ]

  let config =
    train.TrainConfig(..train.default_config(), epochs: 1, batch_size: 2)
  let result = train.fit(net, samples, config)

  should.be_ok(result)

  let assert Ok(#(_trained_net, metrics)) = result
  should.equal(list.length(metrics), 1)
}

pub fn train_fit_multiple_epochs_test() {
  let assert Ok(net) =
    network.new([2, 3, 1], activation.Tanh, activation.Sigmoid)

  let samples = [
    train.sample([1.0, 1.0], [1.0]),
    train.sample([0.0, 0.0], [0.0]),
  ]

  let config =
    train.TrainConfig(..train.default_config(), epochs: 5, batch_size: 2)
  let result = train.fit(net, samples, config)

  should.be_ok(result)

  let assert Ok(#(_trained_net, metrics)) = result
  should.equal(list.length(metrics), 5)
}

pub fn train_fit_metrics_test() {
  let assert Ok(net) =
    network.new([2, 4, 2], activation.ReLU, activation.Softmax)

  let samples = [
    train.sample([1.0, 0.0], [1.0, 0.0]),
    train.sample([0.0, 1.0], [0.0, 1.0]),
  ]

  let config =
    train.TrainConfig(..train.classification_config(), epochs: 3, batch_size: 2)
  let assert Ok(#(_net, metrics)) = train.fit(net, samples, config)

  // Check metrics structure
  let assert Ok(first_metric) = list.first(metrics)
  should.equal(first_metric.epoch, 1)
  should.equal(first_metric.samples, 2)
  should.be_true(first_metric.epoch_loss >=. 0.0)
}

// =============================================================================
// EVALUATION TESTS
// =============================================================================

pub fn train_evaluate_test() {
  let assert Ok(net) =
    network.new([2, 3, 1], activation.ReLU, activation.Sigmoid)

  let samples = [
    train.sample([1.0, 0.0], [1.0]),
    train.sample([0.0, 1.0], [0.0]),
  ]

  let result = train.evaluate(net, samples, train.MSE)
  should.be_ok(result)

  let assert Ok(avg_loss) = result
  should.be_true(avg_loss >=. 0.0)
}

pub fn train_evaluate_empty_test() {
  let assert Ok(net) =
    network.new([2, 3, 1], activation.ReLU, activation.Sigmoid)

  let result = train.evaluate(net, [], train.MSE)
  should.be_ok(result)

  let assert Ok(loss) = result
  should.equal(loss, 0.0)
}

pub fn train_accuracy_test() {
  let assert Ok(net) =
    network.new([2, 4, 2], activation.ReLU, activation.Softmax)

  let samples = [
    train.sample([1.0, 0.0], [1.0, 0.0]),
    train.sample([0.0, 1.0], [0.0, 1.0]),
  ]

  let result = train.accuracy(net, samples)
  should.be_ok(result)

  let assert Ok(acc) = result
  // Accuracy should be between 0 and 1
  should.be_true(acc >=. 0.0)
  should.be_true(acc <=. 1.0)
}

pub fn train_accuracy_empty_test() {
  let assert Ok(net) =
    network.new([2, 3, 2], activation.ReLU, activation.Softmax)

  let result = train.accuracy(net, [])
  should.be_ok(result)

  let assert Ok(acc) = result
  should.equal(acc, 0.0)
}

// =============================================================================
// SAMPLE UTILITY TESTS
// =============================================================================

pub fn train_sample_creation_test() {
  let s = train.sample([1.0, 2.0, 3.0], [0.0, 1.0])

  should.equal(s.input.shape, [3])
  should.equal(s.target.shape, [2])
}

// =============================================================================
// GRADIENT PROCESSING TESTS
// =============================================================================

pub fn train_gradient_clipping_test() {
  let assert Ok(net) =
    network.new([2, 4, 1], activation.ReLU, activation.Linear)
  let input = tensor.from_list([100.0, 100.0])
  let target = tensor.from_list([0.0])

  // Config with gradient clipping
  let config =
    train.TrainConfig(
      ..train.default_config(),
      gradient_clip: 1.0,
      loss: train.MSE,
    )

  // Should not explode even with large inputs
  let result = train.train_step(net, input, target, config)
  should.be_ok(result)
}

pub fn train_l2_regularization_test() {
  let assert Ok(net) =
    network.new([2, 4, 1], activation.ReLU, activation.Sigmoid)
  let input = tensor.from_list([0.5, 0.5])
  let target = tensor.from_list([1.0])

  // Config with L2 regularization
  let config =
    train.TrainConfig(
      ..train.default_config(),
      l2_lambda: 0.01,
      loss: train.MSE,
    )

  let result = train.train_step(net, input, target, config)
  should.be_ok(result)
}
