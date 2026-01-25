# VivaNx - Elixir helper module for Gleam/Nx interop
#
# Add this to your Elixir project that has Nx/EXLA deps.
# Or compile standalone: elixirc viva_nx.ex
#
# Required deps in mix.exs:
#   {:nx, "~> 0.7"},
#   {:exla, "~> 0.7"}
#
# Config (config/config.exs):
#   config :nx, default_backend: EXLA.Backend

defmodule VivaNx do
  @moduledoc """
  Bridge module for Gleam → Nx/EXLA interop.
  Provides functions that Gleam can call via @external FFI.
  """

  # ============================================================================
  # TENSOR CREATION/CONVERSION
  # ============================================================================

  @doc "Create Nx tensor from flat list and shape"
  def tensor_create(data, shape) when is_list(data) and is_list(shape) do
    data
    |> Nx.tensor()
    |> Nx.reshape(List.to_tuple(shape))
  end

  @doc "Convert Nx tensor to flat list"
  def tensor_to_list(tensor) do
    Nx.to_flat_list(tensor)
  end

  @doc "Get tensor shape as list"
  def tensor_shape(tensor) do
    tensor
    |> Nx.shape()
    |> Tuple.to_list()
  end

  # ============================================================================
  # SCALAR OPERATIONS
  # ============================================================================

  @doc "Multiply tensor by scalar"
  def multiply_scalar(tensor, scalar) do
    Nx.multiply(tensor, scalar)
  end

  @doc "Sum tensor to scalar"
  def sum_scalar(tensor) do
    tensor
    |> Nx.sum()
    |> Nx.to_number()
  end

  @doc "Mean tensor to scalar"
  def mean_scalar(tensor) do
    tensor
    |> Nx.mean()
    |> Nx.to_number()
  end

  # ============================================================================
  # ACTIVATIONS
  # ============================================================================

  @doc "Softmax activation"
  def softmax(tensor) do
    Nx.exp(tensor)
    |> then(fn exp_t ->
      Nx.divide(exp_t, Nx.sum(exp_t))
    end)
  end

  @doc "Softmax with axis (for batched)"
  def softmax(tensor, axis) do
    Axon.Activations.softmax(tensor, axis: axis)
  end

  @doc "ReLU activation"
  def relu(tensor) do
    Nx.max(tensor, 0)
  end

  @doc "Sigmoid activation"
  def sigmoid(tensor) do
    Nx.sigmoid(tensor)
  end

  @doc "Tanh activation"
  def tanh(tensor) do
    Nx.tanh(tensor)
  end

  # ============================================================================
  # AUTOGRAD
  # ============================================================================

  @doc "Compute value and gradient of function"
  def value_and_grad(fun, input) do
    Nx.Defn.value_and_grad(input, fun)
  end

  @doc "Compute gradient only"
  def grad(fun, input) do
    Nx.Defn.grad(input, fun)
  end

  # ============================================================================
  # JIT COMPILATION
  # ============================================================================

  @doc "JIT compile unary function"
  def jit(fun) do
    Nx.Defn.jit(fun)
  end

  @doc "JIT compile binary function"
  def jit2(fun) do
    Nx.Defn.jit(fun)
  end

  # ============================================================================
  # DEVICE MANAGEMENT
  # ============================================================================

  @doc "Copy tensor to specific backend/device"
  def backend_copy(tensor, backend, device) do
    backend_module = String.to_existing_atom(backend)
    Nx.backend_copy(tensor, {backend_module, device: device})
  end

  @doc "Check if CUDA is available"
  def cuda_available do
    case System.get_env("EXLA_TARGET") do
      "cuda" -> true
      _ ->
        # Try to detect CUDA
        case System.cmd("which", ["nvidia-smi"], stderr_to_stdout: true) do
          {_, 0} -> true
          _ -> false
        end
    end
  end

  @doc "Get default device string"
  def default_device do
    case Application.get_env(:exla, :default_client) do
      :cuda -> "cuda:0"
      :rocm -> "rocm:0"
      _ -> "host"
    end
  end

  # ============================================================================
  # NEURAL NETWORK HELPERS
  # ============================================================================

  @doc "Dense layer forward pass"
  def dense_forward(input, weights, biases, activation \\ :linear) do
    output =
      input
      |> Nx.dot(weights)
      |> Nx.add(biases)

    apply_activation(output, activation)
  end

  @doc "Apply activation function"
  def apply_activation(tensor, :linear), do: tensor
  def apply_activation(tensor, :relu), do: Nx.max(tensor, 0)
  def apply_activation(tensor, :sigmoid), do: Nx.sigmoid(tensor)
  def apply_activation(tensor, :tanh), do: Nx.tanh(tensor)
  def apply_activation(tensor, :softmax), do: softmax(tensor)

  @doc "Batch dense forward"
  def batch_dense_forward(inputs, weights, biases, activation \\ :linear) do
    # inputs: [batch_size, input_dim]
    # weights: [input_dim, output_dim]
    # biases: [output_dim]

    inputs
    |> Nx.dot(weights)
    |> Nx.add(biases)
    |> apply_activation(activation)
  end

  # ============================================================================
  # LOSS FUNCTIONS
  # ============================================================================

  @doc "Mean Squared Error"
  def mse_loss(predictions, targets) do
    predictions
    |> Nx.subtract(targets)
    |> Nx.power(2)
    |> Nx.mean()
    |> Nx.to_number()
  end

  @doc "Cross Entropy Loss"
  def cross_entropy_loss(predictions, targets) do
    epsilon = 1.0e-7

    predictions
    |> Nx.clip(epsilon, 1.0 - epsilon)
    |> Nx.log()
    |> Nx.multiply(targets)
    |> Nx.sum()
    |> Nx.negate()
    |> Nx.to_number()
  end

  # ============================================================================
  # INITIALIZATION
  # ============================================================================

  @doc "Xavier initialization"
  def xavier_init(rows, cols, key \\ Nx.Random.key(42)) do
    std = :math.sqrt(2.0 / (rows + cols))
    {tensor, _} = Nx.Random.normal(key, 0.0, std, shape: {rows, cols})
    tensor
  end

  @doc "He initialization (for ReLU)"
  def he_init(rows, cols, key \\ Nx.Random.key(42)) do
    std = :math.sqrt(2.0 / rows)
    {tensor, _} = Nx.Random.normal(key, 0.0, std, shape: {rows, cols})
    tensor
  end

  @doc "Zeros tensor"
  def zeros(shape) when is_list(shape) do
    Nx.broadcast(0.0, List.to_tuple(shape))
  end
end
