defmodule VivaNx do
  @moduledoc """
  Bridge module between Gleam and Nx for GPU-accelerated tensor operations.
  Called from viva/neural/nx_backend.gleam via FFI.

  Features:
  - CUDA/EXLA backend support for RTX 4090
  - Async operations to avoid blocking BEAM scheduler
  - Batch processing optimizations
  - JIT compilation for repeated operations
  """

  require Logger

  # ============================================================================
  # CONFIGURATION
  # ============================================================================

  @default_backend EXLA.Backend
  @default_device {:cuda, 0}
  @batch_size 256  # Multiple of 32 for GPU warp alignment (Qwen3-235B recommendation)

  # ============================================================================
  # INITIALIZATION
  # ============================================================================

  @doc "Initialize Nx with CUDA backend"
  def init do
    if cuda_available?() do
      Nx.default_backend({EXLA.Backend, device_id: 0})
      Logger.info("[VivaNx] Initialized with CUDA backend")
      :ok
    else
      Nx.default_backend(Nx.BinaryBackend)
      Logger.warning("[VivaNx] CUDA not available, using CPU backend")
      :cpu_only
    end
  end

  @doc "Set default backend"
  def set_backend(:cuda), do: Nx.default_backend({EXLA.Backend, device_id: 0})
  def set_backend(:cpu), do: Nx.default_backend(Nx.BinaryBackend)
  def set_backend(:exla), do: Nx.default_backend(EXLA.Backend)

  # ============================================================================
  # TENSOR CREATION/CONVERSION
  # ============================================================================

  @doc "Create Nx tensor from flat list with shape"
  def tensor_create(data, shape) when is_list(data) and is_list(shape) do
    shape_tuple = List.to_tuple(shape)
    Nx.tensor(data, type: :f32)
    |> Nx.reshape(shape_tuple)
  end

  @doc "Create Nx tensor with specific type"
  def tensor_create(data, shape, type) do
    shape_tuple = List.to_tuple(shape)
    Nx.tensor(data, type: type)
    |> Nx.reshape(shape_tuple)
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

  @doc "Get tensor size (total elements)"
  def tensor_size(tensor) do
    Nx.size(tensor)
  end

  # ============================================================================
  # DEVICE MANAGEMENT
  # ============================================================================

  @doc "Check if CUDA is available"
  def cuda_available do
    cuda_available?()
  end

  defp cuda_available? do
    case System.cmd("which", ["nvidia-smi"], stderr_to_stdout: true) do
      {_, 0} ->
        # Double check EXLA is loaded with CUDA
        Code.ensure_loaded?(EXLA.Backend)
      _ ->
        false
    end
  end

  @doc "Get default device string"
  def default_device do
    if cuda_available?() do
      "cuda:0"
    else
      "cpu"
    end
  end

  @doc "Copy tensor to specific backend/device"
  def backend_copy(tensor, backend_str, device_str) do
    backend = case backend_str do
      "EXLA.Backend" -> EXLA.Backend
      "Nx.BinaryBackend" -> Nx.BinaryBackend
      _ -> EXLA.Backend
    end

    opts = case device_str do
      "host" -> [client: :host]
      "cuda:0" -> [device_id: 0]
      "cuda:" <> id -> [device_id: String.to_integer(id)]
      _ -> []
    end

    Nx.backend_copy(tensor, {backend, opts})
  end

  @doc "Transfer tensor between backends (async-friendly)"
  def backend_transfer(tensor) do
    Nx.backend_transfer(tensor)
  end

  # ============================================================================
  # CORE TENSOR OPERATIONS
  # ============================================================================

  @doc "Matrix multiplication"
  def dot(a, b) do
    Nx.dot(a, b)
  end

  @doc "Element-wise addition"
  def add(a, b) do
    Nx.add(a, b)
  end

  @doc "Element-wise subtraction"
  def subtract(a, b) do
    Nx.subtract(a, b)
  end

  @doc "Element-wise multiplication"
  def multiply(a, b) do
    Nx.multiply(a, b)
  end

  @doc "Scalar multiplication"
  def multiply_scalar(tensor, scalar) do
    Nx.multiply(tensor, scalar)
  end

  @doc "Element-wise division"
  def divide(a, b) do
    Nx.divide(a, b)
  end

  @doc "Transpose"
  def transpose(tensor) do
    Nx.transpose(tensor)
  end

  @doc "Transpose with axes permutation"
  def transpose(tensor, axes) when is_list(axes) do
    Nx.transpose(tensor, axes: axes)
  end

  @doc "Reshape tensor"
  def reshape(tensor, shape) when is_list(shape) do
    Nx.reshape(tensor, List.to_tuple(shape))
  end

  @doc "Concatenate tensors along axis"
  def concatenate(tensors, axis) when is_list(tensors) do
    Nx.concatenate(tensors, axis: axis)
  end

  @doc "Stack tensors along new axis"
  def stack(tensors, axis) when is_list(tensors) do
    Nx.stack(tensors, axis: axis)
  end

  # ============================================================================
  # REDUCTION OPERATIONS
  # ============================================================================

  @doc "Sum (returns tensor)"
  def sum(tensor) do
    Nx.sum(tensor)
  end

  @doc "Sum along axis"
  def sum(tensor, axis) do
    Nx.sum(tensor, axes: [axis])
  end

  @doc "Sum to scalar"
  def sum_scalar(tensor) do
    tensor |> Nx.sum() |> Nx.to_number()
  end

  @doc "Mean (returns tensor)"
  def mean(tensor) do
    Nx.mean(tensor)
  end

  @doc "Mean along axis"
  def mean(tensor, axis) do
    Nx.mean(tensor, axes: [axis])
  end

  @doc "Mean to scalar"
  def mean_scalar(tensor) do
    tensor |> Nx.mean() |> Nx.to_number()
  end

  @doc "Max value"
  def max(tensor) do
    Nx.reduce_max(tensor)
  end

  @doc "Min value"
  def min(tensor) do
    Nx.reduce_min(tensor)
  end

  @doc "Variance"
  def variance(tensor) do
    Nx.variance(tensor)
  end

  @doc "Variance along axis"
  def variance(tensor, axis) do
    Nx.variance(tensor, axes: [axis])
  end

  @doc "Standard deviation"
  def standard_deviation(tensor) do
    Nx.standard_deviation(tensor)
  end

  # ============================================================================
  # ACTIVATION FUNCTIONS
  # ============================================================================

  @doc "Softmax"
  def softmax(tensor) do
    Nx.exp(tensor)
    |> then(fn exp_t ->
      sum = Nx.sum(exp_t)
      Nx.divide(exp_t, sum)
    end)
  end

  @doc "Softmax along axis (proper implementation)"
  def softmax_axis(tensor, axis) do
    # Numerical stability: subtract max
    max_val = Nx.reduce_max(tensor, axes: [axis], keep_axes: true)
    shifted = Nx.subtract(tensor, max_val)
    exp_t = Nx.exp(shifted)
    sum_exp = Nx.sum(exp_t, axes: [axis], keep_axes: true)
    Nx.divide(exp_t, sum_exp)
  end

  @doc "ReLU"
  def relu(tensor) do
    Nx.max(tensor, 0)
  end

  @doc "Leaky ReLU"
  def leaky_relu(tensor, alpha \\ 0.01) do
    Nx.select(Nx.greater(tensor, 0), tensor, Nx.multiply(tensor, alpha))
  end

  @doc "Sigmoid"
  def sigmoid(tensor) do
    Nx.sigmoid(tensor)
  end

  @doc "Tanh"
  def tanh(tensor) do
    Nx.tanh(tensor)
  end

  @doc "GELU (Gaussian Error Linear Unit)"
  def gelu(tensor) do
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Nx.multiply(tensor, 0.5)
    |> Nx.multiply(
      Nx.add(1, Nx.tanh(
        Nx.multiply(
          0.7978845608,  # sqrt(2/pi)
          Nx.add(tensor, Nx.multiply(0.044715, Nx.power(tensor, 3)))
        )
      ))
    )
  end

  # ============================================================================
  # BATCH OPERATIONS (optimized for GPU)
  # ============================================================================

  @doc "Batch matrix-vector multiplication"
  def batch_matmul(batch_tensor, weights) do
    # batch_tensor: [batch, in_features]
    # weights: [out_features, in_features]
    # result: [batch, out_features]
    Nx.dot(batch_tensor, Nx.transpose(weights))
  end

  @doc "Batch forward pass (linear layer)"
  def batch_linear(batch_input, weights, biases) do
    batch_input
    |> Nx.dot(Nx.transpose(weights))
    |> Nx.add(biases)
  end

  @doc "Evaluate batch of inputs through network layer (with explicit memory management)"
  def batch_forward(inputs_list, weights, biases, activation) do
    # Stack inputs into batch tensor
    batch = inputs_list
    |> Enum.map(&Nx.tensor(&1, type: :f32))
    |> Nx.stack()

    # Forward pass
    result = batch
    |> Nx.dot(Nx.transpose(weights))
    |> Nx.add(biases)
    |> apply_activation(activation)

    # CRITICAL: Transfer back from GPU before unstack to prevent memory leak
    transferred = Nx.backend_transfer(result)

    # Unstack results
    transferred
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.to_flat_list/1)
  end

  defp apply_activation(tensor, :relu), do: relu(tensor)
  defp apply_activation(tensor, :sigmoid), do: sigmoid(tensor)
  defp apply_activation(tensor, :tanh), do: tanh(tensor)
  defp apply_activation(tensor, :softmax), do: softmax_axis(tensor, 1)
  defp apply_activation(tensor, :gelu), do: gelu(tensor)
  defp apply_activation(tensor, :linear), do: tensor
  defp apply_activation(tensor, _), do: tensor

  # ============================================================================
  # NEAT GPU OPERATIONS
  # ============================================================================

  @doc """
  Evaluate multiple NEAT genomes in parallel on GPU.
  Each genome is a list of {weight_matrix, bias_vector, activation}.
  Returns list of outputs for each genome.
  """
  def batch_evaluate_genomes(genomes, inputs) do
    input_tensor = Nx.tensor(inputs, type: :f32)

    genomes
    |> Enum.map(fn genome_layers ->
      Enum.reduce(genome_layers, input_tensor, fn {weights, biases, activation}, acc ->
        weights_t = Nx.tensor(weights, type: :f32)
        biases_t = Nx.tensor(biases, type: :f32)

        acc
        |> Nx.dot(Nx.transpose(weights_t))
        |> Nx.add(biases_t)
        |> apply_activation(activation)
      end)
      |> Nx.to_flat_list()
    end)
  end

  @doc "Parallel fitness evaluation for NEAT population (GPU optimized, warp-aligned)"
  def parallel_fitness_eval(population_weights, inputs_batch, expected_outputs) do
    # population_weights: list of weight matrices
    # inputs_batch: [batch_size, input_features]
    # expected_outputs: [batch_size, output_features]

    inputs = Nx.tensor(inputs_batch, type: :f32)
    expected = Nx.tensor(expected_outputs, type: :f32)

    # Process in chunks of 256 genomes for warp alignment
    population_weights
    |> Enum.chunk_every(@batch_size)
    |> Enum.flat_map(fn chunk ->
      results = Enum.map(chunk, fn weights ->
        weights_t = Nx.tensor(weights, type: :f32)

        # Forward pass
        outputs = Nx.dot(inputs, Nx.transpose(weights_t))
        |> Nx.sigmoid()  # Default activation for NEAT

        # MSE fitness (higher is better, so we invert)
        error = Nx.subtract(outputs, expected)
        mse = error |> Nx.power(2) |> Nx.mean() |> Nx.to_number()
        1.0 / (1.0 + mse)  # Convert to fitness
      end)

      # Yield to scheduler between chunks
      Process.sleep(0)
      results
    end)
  end

  # ============================================================================
  # AUTOGRAD
  # ============================================================================

  @doc "Compute value and gradient"
  def value_and_grad(fun, input) do
    {value, grad_fn} = Nx.Defn.jit(fn x ->
      Nx.Defn.grad(x, fun)
    end).(input)

    {fun.(input), grad_fn}
  end

  @doc "Simple gradient computation"
  def grad(fun, input) do
    Nx.Defn.grad(input, fun)
  end

  # ============================================================================
  # JIT COMPILATION
  # ============================================================================

  @doc "JIT compile a unary function"
  def jit(fun) when is_function(fun, 1) do
    Nx.Defn.jit(fun)
  end

  @doc "JIT compile a binary function"
  def jit2(fun) when is_function(fun, 2) do
    Nx.Defn.jit(fun)
  end

  @doc "JIT compile with specific options"
  def jit_with_opts(fun, opts) do
    Nx.Defn.jit(fun, opts)
  end

  # ============================================================================
  # ASYNC OPERATIONS (dirty schedulers for BEAM safety)
  # ============================================================================

  @doc """
  Run tensor operation on dirty CPU scheduler to avoid blocking BEAM.
  Uses Task.Supervisor with :cpu_bound for proper dirty scheduler usage.
  Explicit backend_transfer prevents GPU memory leaks.
  """
  def async_op(fun) do
    # Use dirty scheduler for CPU-bound Nx operations
    Task.async(fn ->
      # Run on dirty scheduler
      :erlang.process_flag(:priority, :low)
      result = fun.()

      # CRITICAL: Explicit backend_transfer to prevent GPU memory leaks
      if is_struct(result, Nx.Tensor) do
        Nx.backend_transfer(result)
      else
        result
      end
    end)
  end

  @doc "Run operation on dirty CPU scheduler (for heavy Nx operations)"
  def dirty_cpu(fun) do
    # Spawn on dirty CPU scheduler
    ref = make_ref()
    parent = self()

    spawn(fn ->
      :erlang.process_flag(:scheduler, :dirty_cpu)
      result = fun.()
      # Explicit memory transfer
      transferred = if is_struct(result, Nx.Tensor) do
        Nx.backend_transfer(result)
      else
        result
      end
      send(parent, {ref, transferred})
    end)

    receive do
      {^ref, result} -> result
    after
      30_000 -> {:error, :timeout}
    end
  end

  @doc "Await async operation result"
  def await_op(task, timeout \\ 30_000) do
    Task.await(task, timeout)
  end

  @doc "Run batch operation with chunking for scheduler safety (256 chunks)"
  def chunked_batch_op(items, chunk_size \\ @batch_size, op_fun) do
    items
    |> Enum.chunk_every(chunk_size)
    |> Enum.flat_map(fn chunk ->
      # Process chunk and explicitly transfer memory back
      result = op_fun.(chunk)
      # Small yield between chunks
      Process.sleep(0)

      # Ensure GPU memory is released
      Enum.map(result, fn item ->
        if is_struct(item, Nx.Tensor) do
          Nx.backend_transfer(item)
        else
          item
        end
      end)
    end)
  end

  @doc "Stream-based processing for very large batches (Nx.Defn.stream)"
  def stream_batch(items, batch_size \\ @batch_size, op_fun) do
    items
    |> Stream.chunk_every(batch_size)
    |> Stream.map(fn chunk ->
      result = op_fun.(chunk)
      # Force evaluation and memory transfer
      Enum.map(result, &Nx.backend_transfer/1)
    end)
    |> Enum.to_list()
    |> List.flatten()
  end

  # ============================================================================
  # HRR (Holographic Reduced Representation) OPERATIONS
  # ============================================================================

  @doc "Circular convolution for HRR binding"
  def hrr_bind(a, b) do
    # FFT-based circular convolution
    fft_a = Nx.fft(a)
    fft_b = Nx.fft(b)
    Nx.ifft(Nx.multiply(fft_a, fft_b))
    |> Nx.real()
  end

  @doc "Circular correlation for HRR unbinding"
  def hrr_unbind(bound, key) do
    # Unbind: convolve with inverse (conjugate in frequency domain)
    fft_bound = Nx.fft(bound)
    fft_key = Nx.fft(key)
    fft_key_conj = Nx.conjugate(fft_key)
    Nx.ifft(Nx.multiply(fft_bound, fft_key_conj))
    |> Nx.real()
  end

  @doc "Cosine similarity for HRR"
  def hrr_similarity(a, b) do
    dot_product = Nx.dot(a, b) |> Nx.to_number()
    norm_a = a |> Nx.power(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number()
    norm_b = b |> Nx.power(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number()
    dot_product / (norm_a * norm_b + 1.0e-10)
  end

  # ============================================================================
  # CONV2D OPERATIONS
  # ============================================================================

  @doc "2D convolution"
  def conv2d(input, kernel, opts \\ []) do
    strides = Keyword.get(opts, :strides, [1, 1])
    padding = Keyword.get(opts, :padding, :valid)

    Nx.conv(input, kernel, strides: strides, padding: padding)
  end

  @doc "Im2col transformation for efficient conv2d via matmul"
  def im2col(input, kernel_h, kernel_w, stride_h, stride_w) do
    {batch, channels, height, width} = Nx.shape(input)

    out_h = div(height - kernel_h, stride_h) + 1
    out_w = div(width - kernel_w, stride_w) + 1

    # Extract patches
    patches = for i <- 0..(out_h - 1), j <- 0..(out_w - 1) do
      Nx.slice(input, [0, 0, i * stride_h, j * stride_w], [batch, channels, kernel_h, kernel_w])
      |> Nx.reshape({batch, channels * kernel_h * kernel_w})
    end

    Nx.stack(patches, axis: 1)
    |> Nx.reshape({batch, out_h * out_w, channels * kernel_h * kernel_w})
  end

  # ============================================================================
  # ATTENTION OPERATIONS
  # ============================================================================

  @doc "Scaled dot-product attention"
  def scaled_dot_product_attention(query, key, value, mask \\ nil) do
    # query: [seq_q, d_k]
    # key: [seq_k, d_k]
    # value: [seq_k, d_v]

    d_k = elem(Nx.shape(query), -1)
    scale = :math.sqrt(d_k)

    # Scores: Q @ K^T / sqrt(d_k)
    scores = Nx.dot(query, Nx.transpose(key))
    |> Nx.divide(scale)

    # Apply mask if provided
    scores = if mask do
      Nx.add(scores, Nx.multiply(mask, -1.0e9))
    else
      scores
    end

    # Attention weights
    weights = softmax_axis(scores, -1)

    # Output: weights @ V
    output = Nx.dot(weights, value)

    {output, weights}
  end

  @doc "Create causal mask for autoregressive attention"
  def causal_mask(seq_len) do
    Nx.iota({seq_len, seq_len})
    |> then(fn indices ->
      rows = Nx.quotient(indices, seq_len)
      cols = Nx.remainder(indices, seq_len)
      Nx.select(Nx.greater(cols, rows), 1.0, 0.0)
    end)
  end

  # ============================================================================
  # RANDOM OPERATIONS
  # ============================================================================

  @doc "Create tensor of random uniform values"
  def random_uniform(shape, min_val \\ 0.0, max_val \\ 1.0) when is_list(shape) do
    shape_tuple = List.to_tuple(shape)
    key = Nx.Random.key(System.system_time(:nanosecond))
    {tensor, _} = Nx.Random.uniform(key, min_val, max_val, shape: shape_tuple, type: :f32)
    tensor
  end

  @doc "Create tensor of random normal values"
  def random_normal(shape, mean \\ 0.0, std \\ 1.0) when is_list(shape) do
    shape_tuple = List.to_tuple(shape)
    key = Nx.Random.key(System.system_time(:nanosecond))
    {tensor, _} = Nx.Random.normal(key, mean, std, shape: shape_tuple, type: :f32)
    tensor
  end

  @doc "Xavier/Glorot initialization"
  def xavier_uniform(shape) when is_list(shape) do
    [fan_in | rest] = shape
    fan_out = List.last(rest) || fan_in
    limit = :math.sqrt(6.0 / (fan_in + fan_out))
    random_uniform(shape, -limit, limit)
  end

  @doc "He initialization (for ReLU)"
  def he_normal(shape) when is_list(shape) do
    [fan_in | _] = shape
    std = :math.sqrt(2.0 / fan_in)
    random_normal(shape, 0.0, std)
  end

  # ============================================================================
  # BENCHMARKING
  # ============================================================================

  @doc "Benchmark a tensor operation"
  def benchmark(fun, iterations \\ 100) do
    # Warmup
    Enum.each(1..10, fn _ -> fun.() end)

    # Measure
    {time, _} = :timer.tc(fn ->
      Enum.each(1..iterations, fn _ -> fun.() end)
    end)

    avg_us = time / iterations
    %{
      total_ms: time / 1000,
      avg_us: avg_us,
      ops_per_sec: 1_000_000 / avg_us
    }
  end
end
