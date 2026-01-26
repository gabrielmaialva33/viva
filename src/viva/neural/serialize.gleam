//// Serialize - JSON persistence for neural networks
////
//// Allows saving and loading trained networks.
//// Format compatible with the VIVA ecosystem (KarmaBank, etc).

import gleam/dynamic.{type Dynamic}
import gleam/dynamic/decode.{type Decoder}
import gleam/json.{type Json}
import gleam/list
import viva/neural/activation.{type ActivationType}
import viva/neural/layer.{type DenseLayer}
import viva/neural/network.{type Network}
import viva/neural/tensor.{type Tensor, type TensorError}

/// Helper to extract data from tensor
fn td(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// TYPES
// =============================================================================

/// Serialization error
pub type SerializeError {
  InvalidFormat(reason: String)
  MissingField(field: String)
  TensorError(TensorError)
  DecodeError(List(decode.DecodeError))
}

// =============================================================================
// TENSOR SERIALIZATION
// =============================================================================

/// Serialize tensor to JSON
pub fn tensor_to_json(t: Tensor) -> Json {
  json.object([
    #("data", json.array(td(t), json.float)),
    #("shape", json.array(t.shape, json.int)),
  ])
}

/// Tensor decoder
fn tensor_decoder() -> Decoder(Tensor) {
  use data <- decode.field("data", decode.list(decode.float))
  use shape <- decode.field("shape", decode.list(decode.int))
  decode.success(tensor.Tensor(data: data, shape: shape))
}

/// Deserialize tensor from Dynamic
pub fn tensor_from_dynamic(dyn: Dynamic) -> Result(Tensor, SerializeError) {
  case decode.run(dyn, tensor_decoder()) {
    Ok(t) -> Ok(t)
    Error(errors) -> Error(DecodeError(errors))
  }
}

// =============================================================================
// ACTIVATION SERIALIZATION
// =============================================================================

/// Serialize activation type
pub fn activation_to_json(a: ActivationType) -> Json {
  case a {
    activation.Sigmoid -> json.string("sigmoid")
    activation.Tanh -> json.string("tanh")
    activation.ReLU -> json.string("relu")
    activation.LeakyReLU(alpha) ->
      json.object([
        #("type", json.string("leaky_relu")),
        #("alpha", json.float(alpha)),
      ])
    activation.ELU(alpha) ->
      json.object([#("type", json.string("elu")), #("alpha", json.float(alpha))])
    activation.Softmax -> json.string("softmax")
    activation.Linear -> json.string("linear")
    activation.Swish -> json.string("swish")
    activation.GELU -> json.string("gelu")
  }
}

/// Decoder for simple string activations
fn simple_activation_decoder() -> Decoder(ActivationType) {
  use name <- decode.then(decode.string)
  decode.success(activation.from_name(name))
}

/// Decoder for parametrized activations (LeakyReLU, ELU)
fn parametrized_activation_decoder() -> Decoder(ActivationType) {
  use act_type <- decode.field("type", decode.string)
  use alpha <- decode.field("alpha", decode.float)
  case act_type {
    "leaky_relu" -> decode.success(activation.LeakyReLU(alpha))
    "elu" -> decode.success(activation.ELU(alpha))
    _ -> decode.failure(activation.Linear, "Unknown parametrized activation")
  }
}

/// Activation decoder (tries string first, then object)
fn activation_decoder() -> Decoder(ActivationType) {
  decode.one_of(simple_activation_decoder(), [parametrized_activation_decoder()])
}

/// Deserialize activation type
pub fn activation_from_dynamic(
  dyn: Dynamic,
) -> Result(ActivationType, SerializeError) {
  case decode.run(dyn, activation_decoder()) {
    Ok(a) -> Ok(a)
    Error(errors) -> Error(DecodeError(errors))
  }
}

// =============================================================================
// LAYER SERIALIZATION
// =============================================================================

/// Serialize Dense layer
pub fn layer_to_json(l: DenseLayer) -> Json {
  json.object([
    #("type", json.string("dense")),
    #("input_size", json.int(l.input_size)),
    #("output_size", json.int(l.output_size)),
    #("weights", tensor_to_json(l.weights)),
    #("biases", tensor_to_json(l.biases)),
    #("activation", activation_to_json(l.activation)),
  ])
}

/// Dense layer decoder
fn dense_layer_decoder() -> Decoder(DenseLayer) {
  use layer_type <- decode.field("type", decode.string)
  case layer_type {
    "dense" -> {
      use input_size <- decode.field("input_size", decode.int)
      use output_size <- decode.field("output_size", decode.int)
      use weights <- decode.field("weights", tensor_decoder())
      use biases <- decode.field("biases", tensor_decoder())
      use act <- decode.field("activation", activation_decoder())
      decode.success(layer.DenseLayer(
        weights: weights,
        biases: biases,
        activation: act,
        input_size: input_size,
        output_size: output_size,
      ))
    }
    _ ->
      decode.failure(
        layer.DenseLayer(
          weights: tensor.Tensor(data: [], shape: []),
          biases: tensor.Tensor(data: [], shape: []),
          activation: activation.Linear,
          input_size: 0,
          output_size: 0,
        ),
        "Unknown layer type",
      )
  }
}

/// Deserialize Dense layer
pub fn layer_from_dynamic(dyn: Dynamic) -> Result(DenseLayer, SerializeError) {
  case decode.run(dyn, dense_layer_decoder()) {
    Ok(l) -> Ok(l)
    Error(errors) -> Error(DecodeError(errors))
  }
}

// =============================================================================
// NETWORK SERIALIZATION
// =============================================================================

/// Serialize complete network
pub fn network_to_json(net: Network) -> Json {
  json.object([
    #("version", json.string("1.0")),
    #("type", json.string("viva_neural_network")),
    #("input_size", json.int(net.input_size)),
    #("output_size", json.int(net.output_size)),
    #("layers", json.array(net.layers, layer_to_json)),
  ])
}

/// Convert network to JSON string
pub fn network_to_string(net: Network) -> String {
  network_to_json(net)
  |> json.to_string
}

/// Network decoder
fn network_decoder() -> Decoder(Network) {
  use version <- decode.field("version", decode.string)
  use net_type <- decode.field("type", decode.string)
  case version, net_type {
    "1.0", "viva_neural_network" -> {
      use input_size <- decode.field("input_size", decode.int)
      use output_size <- decode.field("output_size", decode.int)
      use layers <- decode.field("layers", decode.list(dense_layer_decoder()))
      decode.success(network.Network(
        layers: layers,
        input_size: input_size,
        output_size: output_size,
      ))
    }
    _, _ ->
      decode.failure(
        network.Network(layers: [], input_size: 0, output_size: 0),
        "Invalid network version or type",
      )
  }
}

/// Deserialize network from Dynamic
pub fn network_from_dynamic(dyn: Dynamic) -> Result(Network, SerializeError) {
  case decode.run(dyn, network_decoder()) {
    Ok(n) -> Ok(n)
    Error(errors) -> Error(DecodeError(errors))
  }
}

/// Load network from JSON string
pub fn network_from_string(json_str: String) -> Result(Network, SerializeError) {
  case json.parse(json_str, network_decoder()) {
    Ok(net) -> Ok(net)
    Error(_) -> Error(InvalidFormat("Invalid JSON"))
  }
}

// =============================================================================
// COMPACT FORMAT (for KarmaBank)
// =============================================================================

/// Compact format: weights only as list of floats
pub fn network_to_weights(net: Network) -> List(Float) {
  list.flat_map(net.layers, fn(l) { list.append(td(l.weights), td(l.biases)) })
}

/// Recreate network from weights (needs architecture)
pub fn network_from_weights(
  architecture: List(Int),
  hidden_activation: ActivationType,
  output_activation: ActivationType,
  weights: List(Float),
) -> Result(Network, SerializeError) {
  case network.new(architecture, hidden_activation, output_activation) {
    Ok(template) -> {
      // Distribute weights across layers
      case distribute_weights(template.layers, weights, []) {
        Ok(new_layers) -> Ok(network.Network(..template, layers: new_layers))
        Error(e) -> Error(e)
      }
    }
    Error(e) -> Error(TensorError(e))
  }
}

fn distribute_weights(
  layers: List(DenseLayer),
  remaining_weights: List(Float),
  built: List(DenseLayer),
) -> Result(List(DenseLayer), SerializeError) {
  case layers {
    [] -> {
      case remaining_weights {
        [] -> Ok(list.reverse(built))
        _ -> Error(InvalidFormat("Too many weights provided"))
      }
    }
    [l, ..rest] -> {
      let weights_size = l.input_size * l.output_size
      let biases_size = l.output_size
      let total = weights_size + biases_size

      case list.length(remaining_weights) >= total {
        True -> {
          let #(layer_weights, after_weights) =
            list.split(remaining_weights, weights_size)
          let #(layer_biases, rest_weights) =
            list.split(after_weights, biases_size)

          let new_layer =
            layer.DenseLayer(
              ..l,
              weights: tensor.Tensor(data: layer_weights, shape: [
                l.input_size,
                l.output_size,
              ]),
              biases: tensor.Tensor(data: layer_biases, shape: [l.output_size]),
            )

          distribute_weights(rest, rest_weights, [new_layer, ..built])
        }
        False -> Error(InvalidFormat("Not enough weights"))
      }
    }
  }
}

// =============================================================================
// METADATA
// =============================================================================

/// Network metadata for storage
pub type NetworkMetadata {
  NetworkMetadata(
    /// Format version
    version: String,
    /// Name/description
    name: String,
    /// Architecture (layer sizes)
    architecture: List(Int),
    /// Number of parameters
    param_count: Int,
    /// Creation timestamp (Unix)
    created_at: Int,
    /// Training metrics
    training_loss: Float,
    /// Epochs trained
    epochs_trained: Int,
  )
}

/// Create metadata for network
pub fn create_metadata(
  net: Network,
  name: String,
  training_loss: Float,
  epochs: Int,
) -> NetworkMetadata {
  NetworkMetadata(
    version: "1.0",
    name: name,
    architecture: network.layer_sizes(net),
    param_count: network.param_count(net),
    created_at: erlang_system_time(),
    training_loss: training_loss,
    epochs_trained: epochs,
  )
}

/// Serialize metadata
pub fn metadata_to_json(m: NetworkMetadata) -> Json {
  json.object([
    #("version", json.string(m.version)),
    #("name", json.string(m.name)),
    #("architecture", json.array(m.architecture, json.int)),
    #("param_count", json.int(m.param_count)),
    #("created_at", json.int(m.created_at)),
    #("training_loss", json.float(m.training_loss)),
    #("epochs_trained", json.int(m.epochs_trained)),
  ])
}

/// Serialize network with metadata
pub fn network_with_metadata_to_json(
  net: Network,
  metadata: NetworkMetadata,
) -> Json {
  json.object([
    #("metadata", metadata_to_json(metadata)),
    #("network", network_to_json(net)),
  ])
}

// =============================================================================
// EXTERNAL
// =============================================================================

@external(erlang, "erlang", "system_time")
fn erlang_system_time() -> Int
