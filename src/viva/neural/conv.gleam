//// Conv - Convolutional Neural Network layers
////
//// Implements Conv2D using the im2col approach for efficiency.
//// Converts convolution into matrix multiplication for better performance.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva/neural/tensor.{type Tensor, type TensorError, Tensor}
import viva/neural/activation.{type ActivationType}

// =============================================================================
// TYPES
// =============================================================================

/// Padding mode for convolution
pub type Padding {
  /// No padding, output is smaller
  Valid
  /// Pad to keep output same size as input (when stride=1)
  Same
  /// Explicit padding values
  Explicit(pad_h: Int, pad_w: Int)
}

/// Conv2D layer
pub type Conv2DLayer {
  Conv2DLayer(
    /// Filter weights [out_channels, in_channels, kernel_h, kernel_w]
    filters: Tensor,
    /// Bias [out_channels]
    biases: Tensor,
    /// Input channels
    in_channels: Int,
    /// Output channels (number of filters)
    out_channels: Int,
    /// Kernel height
    kernel_h: Int,
    /// Kernel width
    kernel_w: Int,
    /// Stride height
    stride_h: Int,
    /// Stride width
    stride_w: Int,
    /// Padding mode
    padding: Padding,
    /// Activation function
    activation: ActivationType,
  )
}

/// Cache for backward pass
pub type Conv2DCache {
  Conv2DCache(
    /// Input to the layer
    input: Tensor,
    /// im2col transformed input
    col: Tensor,
    /// Pre-activation output
    pre_activation: Tensor,
    /// Output height
    out_h: Int,
    /// Output width
    out_w: Int,
    /// Pad height used
    pad_h: Int,
    /// Pad width used
    pad_w: Int,
  )
}

/// Gradients from Conv2D backward
pub type Conv2DGradients {
  Conv2DGradients(
    /// Gradient w.r.t. input
    d_input: Tensor,
    /// Gradient w.r.t. filters
    d_filters: Tensor,
    /// Gradient w.r.t. biases
    d_biases: Tensor,
  )
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create new Conv2D layer
pub fn new(
  in_channels: Int,
  out_channels: Int,
  kernel_size: Int,
  stride: Int,
  padding: Padding,
  activation: ActivationType,
) -> Conv2DLayer {
  new_asymmetric(
    in_channels,
    out_channels,
    kernel_size,
    kernel_size,
    stride,
    stride,
    padding,
    activation,
  )
}

/// Create Conv2D with asymmetric kernel
pub fn new_asymmetric(
  in_channels: Int,
  out_channels: Int,
  kernel_h: Int,
  kernel_w: Int,
  stride_h: Int,
  stride_w: Int,
  padding: Padding,
  activation: ActivationType,
) -> Conv2DLayer {
  // Initialize filters with He initialization
  let fan_in = in_channels * kernel_h * kernel_w
  let filters = he_init_4d(out_channels, in_channels, kernel_h, kernel_w, fan_in)
  let biases = tensor.zeros([out_channels])

  Conv2DLayer(
    filters: filters,
    biases: biases,
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_h: kernel_h,
    kernel_w: kernel_w,
    stride_h: stride_h,
    stride_w: stride_w,
    padding: padding,
    activation: activation,
  )
}

/// Create Conv2D with explicit weights (useful for testing)
pub fn from_weights(
  filters: Tensor,
  biases: Tensor,
  in_channels: Int,
  out_channels: Int,
  kernel_h: Int,
  kernel_w: Int,
  stride_h: Int,
  stride_w: Int,
  padding: Padding,
  activation: ActivationType,
) -> Conv2DLayer {
  Conv2DLayer(
    filters: filters,
    biases: biases,
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_h: kernel_h,
    kernel_w: kernel_w,
    stride_h: stride_h,
    stride_w: stride_w,
    padding: padding,
    activation: activation,
  )
}

// =============================================================================
// FORWARD PASS
// =============================================================================

/// Forward pass through Conv2D
/// Input shape: [batch, in_channels, height, width]
/// Output shape: [batch, out_channels, out_height, out_width]
pub fn forward(
  layer: Conv2DLayer,
  input: Tensor,
) -> Result(#(Tensor, Conv2DCache), TensorError) {
  case input.shape {
    [batch, in_ch, in_h, in_w] if in_ch == layer.in_channels -> {
      // Calculate padding
      let #(pad_h, pad_w) = calculate_padding(
        layer.padding,
        in_h,
        in_w,
        layer.kernel_h,
        layer.kernel_w,
        layer.stride_h,
        layer.stride_w,
      )

      // Calculate output dimensions
      let out_h = { in_h + 2 * pad_h - layer.kernel_h } / layer.stride_h + 1
      let out_w = { in_w + 2 * pad_w - layer.kernel_w } / layer.stride_w + 1

      // Process each batch sample
      let #(output_data, col_data) =
        list.range(0, batch - 1)
        |> list.fold(#([], []), fn(acc, b) {
          let #(out_acc, col_acc) = acc

          // Extract single sample [in_ch, in_h, in_w]
          let sample_start = b * in_ch * in_h * in_w
          let sample_data =
            input.data
            |> list.drop(sample_start)
            |> list.take(in_ch * in_h * in_w)
          let sample = Tensor(data: sample_data, shape: [in_ch, in_h, in_w])

          // Apply im2col
          let col = im2col(
            sample,
            layer.kernel_h,
            layer.kernel_w,
            layer.stride_h,
            layer.stride_w,
            pad_h,
            pad_w,
          )

          // Reshape filters to 2D: [out_ch, in_ch * kH * kW]
          let filters_2d = tensor.reshape(
            layer.filters,
            [layer.out_channels, layer.in_channels * layer.kernel_h * layer.kernel_w],
          )

          // Convolution via matmul: filters_2d @ col
          let conv_result = case filters_2d {
            Ok(f2d) -> {
              case tensor.matmul(f2d, col) {
                Ok(result) -> result
                Error(_) -> tensor.zeros([layer.out_channels, out_h * out_w])
              }
            }
            Error(_) -> tensor.zeros([layer.out_channels, out_h * out_w])
          }

          // Add bias (broadcast to each output position)
          let biased_data =
            list.range(0, layer.out_channels - 1)
            |> list.flat_map(fn(c) {
              let bias_val = case list_at(layer.biases.data, c) {
                Ok(b) -> b
                Error(_) -> 0.0
              }
              let row_start = c * out_h * out_w
              conv_result.data
              |> list.drop(row_start)
              |> list.take(out_h * out_w)
              |> list.map(fn(x) { x +. bias_val })
            })

          #(list.append(out_acc, biased_data), list.append(col_acc, col.data))
        })

      // Apply activation
      let pre_act = Tensor(
        data: output_data,
        shape: [batch, layer.out_channels, out_h, out_w],
      )
      let output = apply_activation(pre_act, layer.activation)

      // Store col for backward pass
      let col_tensor = Tensor(
        data: col_data,
        shape: [batch, layer.in_channels * layer.kernel_h * layer.kernel_w, out_h * out_w],
      )

      let cache = Conv2DCache(
        input: input,
        col: col_tensor,
        pre_activation: pre_act,
        out_h: out_h,
        out_w: out_w,
        pad_h: pad_h,
        pad_w: pad_w,
      )

      Ok(#(output, cache))
    }
    _ ->
      Error(tensor.DimensionError(
        "Conv2D expects [batch, in_channels, height, width] input",
      ))
  }
}

// =============================================================================
// BACKWARD PASS
// =============================================================================

/// Backward pass through Conv2D
pub fn backward(
  layer: Conv2DLayer,
  cache: Conv2DCache,
  upstream: Tensor,
) -> Result(Conv2DGradients, TensorError) {
  case upstream.shape, cache.input.shape {
    [batch, out_ch, out_h, out_w], [_b, in_ch, in_h, in_w]
      if out_ch == layer.out_channels
    -> {
      // Apply activation derivative
      let d_pre_act = apply_activation_backward(
        upstream,
        cache.pre_activation,
        layer.activation,
      )

      // Compute d_biases: sum over batch and spatial dims
      let d_biases_data =
        list.range(0, layer.out_channels - 1)
        |> list.map(fn(c) {
          list.range(0, batch - 1)
          |> list.fold(0.0, fn(acc, b) {
            let start = { b * out_ch + c } * out_h * out_w
            d_pre_act.data
            |> list.drop(start)
            |> list.take(out_h * out_w)
            |> list.fold(acc, fn(a, x) { a +. x })
          })
        })
      let d_biases = Tensor(data: d_biases_data, shape: [layer.out_channels])

      // Compute d_filters
      // d_filters = d_pre_act @ col^T for each batch, then sum
      let d_filters_data =
        list.range(0, batch - 1)
        |> list.fold(
          list.repeat(0.0, layer.out_channels * layer.in_channels * layer.kernel_h * layer.kernel_w),
          fn(acc_data, b) {
            // Get d_pre_act for this batch [out_ch, out_h * out_w]
            let d_start = b * out_ch * out_h * out_w
            let d_batch =
              d_pre_act.data
              |> list.drop(d_start)
              |> list.take(out_ch * out_h * out_w)
            let d_batch_tensor = Tensor(
              data: d_batch,
              shape: [out_ch, out_h * out_w],
            )

            // Get col for this batch [in_ch * kH * kW, out_h * out_w]
            let col_size =
              layer.in_channels
              * layer.kernel_h
              * layer.kernel_w
              * out_h
              * out_w
            let col_start = b * col_size
            let col_batch =
              cache.col.data
              |> list.drop(col_start)
              |> list.take(col_size)
            let col_tensor = Tensor(
              data: col_batch,
              shape: [layer.in_channels * layer.kernel_h * layer.kernel_w, out_h * out_w],
            )

            // d_filters_batch = d_batch @ col^T
            case tensor.transpose(col_tensor) {
              Ok(col_t) -> {
                case tensor.matmul(d_batch_tensor, col_t) {
                  Ok(d_f) -> {
                    list.map2(acc_data, d_f.data, fn(a, b) { a +. b })
                  }
                  Error(_) -> acc_data
                }
              }
              Error(_) -> acc_data
            }
          },
        )

      let d_filters = Tensor(
        data: d_filters_data,
        shape: [layer.out_channels, layer.in_channels, layer.kernel_h, layer.kernel_w],
      )

      // Compute d_input via col2im
      // d_col = filters^T @ d_pre_act
      let d_input_data =
        list.range(0, batch - 1)
        |> list.flat_map(fn(b) {
          let d_start = b * out_ch * out_h * out_w
          let d_batch =
            d_pre_act.data
            |> list.drop(d_start)
            |> list.take(out_ch * out_h * out_w)
          let d_batch_tensor = Tensor(
            data: d_batch,
            shape: [out_ch, out_h * out_w],
          )

          // Reshape filters to 2D and transpose
          case
            tensor.reshape(
              layer.filters,
              [layer.out_channels, layer.in_channels * layer.kernel_h * layer.kernel_w],
            )
          {
            Ok(f2d) -> {
              case tensor.transpose(f2d) {
                Ok(f2d_t) -> {
                  case tensor.matmul(f2d_t, d_batch_tensor) {
                    Ok(d_col) -> {
                      // col2im to convert back to input shape
                      let d_input_sample = col2im(
                        d_col,
                        in_ch,
                        in_h,
                        in_w,
                        layer.kernel_h,
                        layer.kernel_w,
                        layer.stride_h,
                        layer.stride_w,
                        cache.pad_h,
                        cache.pad_w,
                      )
                      d_input_sample.data
                    }
                    Error(_) -> list.repeat(0.0, in_ch * in_h * in_w)
                  }
                }
                Error(_) -> list.repeat(0.0, in_ch * in_h * in_w)
              }
            }
            Error(_) -> list.repeat(0.0, in_ch * in_h * in_w)
          }
        })

      let d_input = Tensor(
        data: d_input_data,
        shape: [batch, in_ch, in_h, in_w],
      )

      Ok(Conv2DGradients(d_input: d_input, d_filters: d_filters, d_biases: d_biases))
    }
    _, _ -> Error(tensor.DimensionError("Shape mismatch in Conv2D backward"))
  }
}

// =============================================================================
// IM2COL / COL2IM
// =============================================================================

/// Convert image to column matrix for efficient convolution
/// Input: [in_channels, height, width]
/// Output: [in_channels * kernel_h * kernel_w, out_h * out_w]
fn im2col(
  input: Tensor,
  kernel_h: Int,
  kernel_w: Int,
  stride_h: Int,
  stride_w: Int,
  pad_h: Int,
  pad_w: Int,
) -> Tensor {
  case input.shape {
    [in_ch, in_h, in_w] -> {
      let padded_h = in_h + 2 * pad_h
      let padded_w = in_w + 2 * pad_w
      let out_h = { padded_h - kernel_h } / stride_h + 1
      let out_w = { padded_w - kernel_w } / stride_w + 1

      // Pad the input
      let padded = pad_image(input, in_ch, in_h, in_w, pad_h, pad_w)

      // Extract patches column by column
      let col_data =
        list.range(0, in_ch * kernel_h * kernel_w - 1)
        |> list.flat_map(fn(col_idx) {
          let c = col_idx / { kernel_h * kernel_w }
          let kh = { col_idx % { kernel_h * kernel_w } } / kernel_w
          let kw = col_idx % kernel_w

          list.range(0, out_h * out_w - 1)
          |> list.map(fn(out_idx) {
            let oh = out_idx / out_w
            let ow = out_idx % out_w
            let h = oh * stride_h + kh
            let w = ow * stride_w + kw

            // Get value from padded input
            let flat_idx = c * padded_h * padded_w + h * padded_w + w
            case list_at(padded.data, flat_idx) {
              Ok(v) -> v
              Error(_) -> 0.0
            }
          })
        })

      Tensor(
        data: col_data,
        shape: [in_ch * kernel_h * kernel_w, out_h * out_w],
      )
    }
    _ -> tensor.zeros([1, 1])
  }
}

/// Convert column matrix back to image (for backprop)
fn col2im(
  col: Tensor,
  in_ch: Int,
  in_h: Int,
  in_w: Int,
  kernel_h: Int,
  kernel_w: Int,
  stride_h: Int,
  stride_w: Int,
  pad_h: Int,
  pad_w: Int,
) -> Tensor {
  let padded_h = in_h + 2 * pad_h
  let padded_w = in_w + 2 * pad_w
  let out_h = { padded_h - kernel_h } / stride_h + 1
  let out_w = { padded_w - kernel_w } / stride_w + 1

  // Initialize output with zeros
  let init_size = in_ch * padded_h * padded_w
  let init_data = list.repeat(0.0, init_size)

  // Accumulate values from col back to image positions
  let accumulated =
    list.range(0, in_ch * kernel_h * kernel_w - 1)
    |> list.fold(init_data, fn(acc, col_idx) {
      let c = col_idx / { kernel_h * kernel_w }
      let kh = { col_idx % { kernel_h * kernel_w } } / kernel_w
      let kw = col_idx % kernel_w

      list.range(0, out_h * out_w - 1)
      |> list.fold(acc, fn(inner_acc, out_idx) {
        let oh = out_idx / out_w
        let ow = out_idx % out_w
        let h = oh * stride_h + kh
        let w = ow * stride_w + kw

        // Get col value
        let col_flat_idx = col_idx * out_h * out_w + out_idx
        let col_val = case list_at(col.data, col_flat_idx) {
          Ok(v) -> v
          Error(_) -> 0.0
        }

        // Add to output position
        let out_flat_idx = c * padded_h * padded_w + h * padded_w + w
        list.index_map(inner_acc, fn(x, i) {
          case i == out_flat_idx {
            True -> x +. col_val
            False -> x
          }
        })
      })
    })

  // Remove padding to get original size
  unpad_image(
    Tensor(data: accumulated, shape: [in_ch, padded_h, padded_w]),
    in_ch,
    in_h,
    in_w,
    pad_h,
    pad_w,
  )
}

/// Pad image with zeros
fn pad_image(
  input: Tensor,
  in_ch: Int,
  in_h: Int,
  in_w: Int,
  pad_h: Int,
  pad_w: Int,
) -> Tensor {
  let padded_h = in_h + 2 * pad_h
  let padded_w = in_w + 2 * pad_w

  let data =
    list.range(0, in_ch - 1)
    |> list.flat_map(fn(c) {
      list.range(0, padded_h - 1)
      |> list.flat_map(fn(h) {
        list.range(0, padded_w - 1)
        |> list.map(fn(w) {
          // Check if in valid (non-padded) region
          let orig_h = h - pad_h
          let orig_w = w - pad_w
          case
            orig_h >= 0 && orig_h < in_h && orig_w >= 0 && orig_w < in_w
          {
            True -> {
              let flat_idx = c * in_h * in_w + orig_h * in_w + orig_w
              case list_at(input.data, flat_idx) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
            }
            False -> 0.0
          }
        })
      })
    })

  Tensor(data: data, shape: [in_ch, padded_h, padded_w])
}

/// Remove padding from image
fn unpad_image(
  input: Tensor,
  in_ch: Int,
  orig_h: Int,
  orig_w: Int,
  pad_h: Int,
  pad_w: Int,
) -> Tensor {
  case input.shape {
    [_ch, padded_h, padded_w] -> {
      let data =
        list.range(0, in_ch - 1)
        |> list.flat_map(fn(c) {
          list.range(0, orig_h - 1)
          |> list.flat_map(fn(h) {
            list.range(0, orig_w - 1)
            |> list.map(fn(w) {
              let padded_row = h + pad_h
              let padded_col = w + pad_w
              let flat_idx =
                c * padded_h * padded_w + padded_row * padded_w + padded_col
              case list_at(input.data, flat_idx) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
            })
          })
        })

      Tensor(data: data, shape: [in_ch, orig_h, orig_w])
    }
    _ -> input
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Calculate padding based on mode
fn calculate_padding(
  mode: Padding,
  in_h: Int,
  in_w: Int,
  kernel_h: Int,
  kernel_w: Int,
  stride_h: Int,
  stride_w: Int,
) -> #(Int, Int) {
  case mode {
    Valid -> #(0, 0)
    Same -> {
      // Calculate padding to make output same size as input (when stride=1)
      let pad_h = { { in_h - 1 } * stride_h + kernel_h - in_h } / 2
      let pad_w = { { in_w - 1 } * stride_w + kernel_w - in_w } / 2
      #(int.max(0, pad_h), int.max(0, pad_w))
    }
    Explicit(pad_h, pad_w) -> #(pad_h, pad_w)
  }
}

/// He initialization for 4D filters
fn he_init_4d(
  out_ch: Int,
  in_ch: Int,
  kernel_h: Int,
  kernel_w: Int,
  fan_in: Int,
) -> Tensor {
  let std = float_sqrt(2.0 /. int.to_float(fan_in))
  let size = out_ch * in_ch * kernel_h * kernel_w

  let data =
    list.range(1, size)
    |> list.map(fn(_) {
      // Box-Muller approximation
      let u1 = float.max(random_float(), 0.0001)
      let u2 = random_float()
      let z =
        float_sqrt(-2.0 *. float_log(u1))
        *. float_cos(2.0 *. 3.14159265359 *. u2)
      z *. std
    })

  Tensor(data: data, shape: [out_ch, in_ch, kernel_h, kernel_w])
}

/// Apply activation function to tensor
fn apply_activation(t: Tensor, act: ActivationType) -> Tensor {
  case act {
    activation.Linear -> t
    activation.ReLU -> tensor.map(t, fn(x) { float.max(0.0, x) })
    activation.Sigmoid ->
      tensor.map(t, fn(x) { 1.0 /. { 1.0 +. float_exp(0.0 -. x) } })
    activation.Tanh -> tensor.map(t, fn(x) {
      let e2x = float_exp(2.0 *. x)
      { e2x -. 1.0 } /. { e2x +. 1.0 }
    })
    _ -> t
  }
}

/// Apply activation derivative for backprop
fn apply_activation_backward(
  upstream: Tensor,
  pre_act: Tensor,
  act: ActivationType,
) -> Tensor {
  case act {
    activation.Linear -> upstream
    activation.ReLU -> {
      let data =
        list.map2(upstream.data, pre_act.data, fn(u, p) {
          case p >. 0.0 {
            True -> u
            False -> 0.0
          }
        })
      Tensor(data: data, shape: upstream.shape)
    }
    activation.Sigmoid -> {
      let data =
        list.map2(upstream.data, pre_act.data, fn(u, p) {
          let s = 1.0 /. { 1.0 +. float_exp(0.0 -. p) }
          u *. s *. { 1.0 -. s }
        })
      Tensor(data: data, shape: upstream.shape)
    }
    activation.Tanh -> {
      let data =
        list.map2(upstream.data, pre_act.data, fn(u, p) {
          let e2x = float_exp(2.0 *. p)
          let t = { e2x -. 1.0 } /. { e2x +. 1.0 }
          u *. { 1.0 -. t *. t }
        })
      Tensor(data: data, shape: upstream.shape)
    }
    _ -> upstream
  }
}

fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  lst
  |> list.drop(index)
  |> list.first
}

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "exp")
fn float_exp(x: Float) -> Float

@external(erlang, "math", "log")
fn float_log(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_float() -> Float
