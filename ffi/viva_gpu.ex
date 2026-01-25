defmodule VivaGpu do
  @moduledoc """
  GPU-accelerated operations for VIVA using Nx/EXLA.
  Called from Gleam via FFI.
  """

  # ============================================================================
  # DETECTION
  # ============================================================================

  @doc "Check if CUDA GPU is available"
  def cuda_available do
    case System.get_env("XLA_TARGET") do
      target when target in ["cuda", "cuda11", "cuda12"] -> true
      _ ->
        case System.cmd("which", ["nvidia-smi"], stderr_to_stdout: true) do
          {_, 0} -> true
          _ -> false
        end
    end
  end

  @doc "Check if EXLA is available"
  def exla_available do
    Code.ensure_loaded?(EXLA.Backend)
  end

  @doc "Get GPU info"
  def gpu_info do
    if cuda_available() do
      # Parse nvidia-smi output
      case System.cmd("nvidia-smi", ["--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], stderr_to_stdout: true) do
        {output, 0} ->
          [name, memory, compute] = output |> String.trim() |> String.split(", ")
          memory_mb = memory |> String.replace(" MiB", "") |> String.to_integer()

          {:GpuInfo, true, name, memory_mb, compute}

        _ ->
          {:GpuInfo, false, "Unknown", 0, "N/A"}
      end
    else
      {:GpuInfo, false, "None", 0, "N/A"}
    end
  end

  # ============================================================================
  # PAD BATCH OPERATIONS
  # ============================================================================

  @doc "Apply delta to all PADs in batch"
  def batch_apply_delta({:PadBatch, data, count}, {:Pad, dp, da, dd}) do
    # data is [p1,a1,d1, p2,a2,d2, ...]
    # delta is applied per dimension

    delta = Nx.tensor([dp, da, dd], type: :f32)
    |> Nx.tile([count])

    result = Nx.tensor(data, type: :f32)
    |> Nx.add(delta)
    |> Nx.clip(-1.0, 1.0)
    |> Nx.to_flat_list()

    {:PadBatch, result, count}
  end

  @doc "Scale all PADs"
  def batch_scale({:PadBatch, data, count}, factor) do
    result = Nx.tensor(data, type: :f32)
    |> Nx.multiply(factor)
    |> Nx.clip(-1.0, 1.0)
    |> Nx.to_flat_list()

    {:PadBatch, result, count}
  end

  @doc "Lerp all PADs toward target"
  def batch_lerp({:PadBatch, data, count}, {:Pad, tp, ta, td}, t) do
    current = Nx.tensor(data, type: :f32) |> Nx.reshape({count, 3})
    target = Nx.tensor([tp, ta, td], type: :f32) |> Nx.broadcast({count, 3})

    result = current
    |> Nx.add(Nx.multiply(Nx.subtract(target, current), t))
    |> Nx.reshape({count * 3})
    |> Nx.to_flat_list()

    {:PadBatch, result, count}
  end

  # ============================================================================
  # TENSOR OPERATIONS
  # ============================================================================

  @doc "Batch matrix multiplication"
  def batch_matmul(inputs, {:Tensor, weights_data, weights_shape}) do
    [rows, cols] = weights_shape

    # Stack inputs into batch
    input_data = Enum.flat_map(inputs, fn {:Tensor, data, _shape} -> data end)
    batch_size = length(inputs)
    input_size = case inputs do
      [{:Tensor, _, [s]} | _] -> s
      _ -> cols
    end

    # Create Nx tensors
    batch = Nx.tensor(input_data, type: :f32) |> Nx.reshape({batch_size, input_size})
    weights = Nx.tensor(weights_data, type: :f32) |> Nx.reshape({rows, cols})

    # Batch matmul: [batch, in] @ [out, in]^T = [batch, out]
    result = Nx.dot(batch, Nx.transpose(weights))

    # Convert back to list of tensors
    result
    |> Nx.to_batched(1)
    |> Enum.map(fn row ->
      data = row |> Nx.squeeze() |> Nx.to_flat_list()
      {:Tensor, data, [length(data)]}
    end)
  end

  @doc "Batch dense layer forward"
  def batch_dense_forward(inputs, {:Tensor, w_data, w_shape}, {:Tensor, b_data, _b_shape}, activation) do
    [rows, cols] = w_shape
    batch_size = length(inputs)

    input_data = Enum.flat_map(inputs, fn {:Tensor, data, _} -> data end)

    batch = Nx.tensor(input_data, type: :f32) |> Nx.reshape({batch_size, cols})
    weights = Nx.tensor(w_data, type: :f32) |> Nx.reshape({rows, cols})
    biases = Nx.tensor(b_data, type: :f32)

    # Forward: batch @ weights^T + biases
    z = batch
    |> Nx.dot(Nx.transpose(weights))
    |> Nx.add(biases)
    |> apply_activation(activation)

    # Convert back
    z
    |> Nx.to_batched(1)
    |> Enum.map(fn row ->
      data = row |> Nx.squeeze() |> Nx.to_flat_list()
      {:Tensor, data, [length(data)]}
    end)
  end

  defp apply_activation(tensor, :Linear), do: tensor
  defp apply_activation(tensor, :ReLU), do: Nx.max(tensor, 0)
  defp apply_activation(tensor, :Sigmoid), do: Nx.sigmoid(tensor)
  defp apply_activation(tensor, :Tanh), do: Nx.tanh(tensor)
  defp apply_activation(tensor, :Softmax) do
    # Softmax along last axis
    max_val = Nx.reduce_max(tensor, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(tensor, max_val)
    exp_vals = Nx.exp(shifted)
    sum_exp = Nx.sum(exp_vals, axes: [-1], keep_axes: true)
    Nx.divide(exp_vals, sum_exp)
  end

  # ============================================================================
  # RESONANCE
  # ============================================================================

  @doc "Calculate all pairwise resonances"
  def batch_resonance(pads) do
    # Convert list of Pad to Nx tensor [n, 3]
    n = length(pads)

    data = Enum.flat_map(pads, fn {:Pad, p, a, d} -> [p, a, d] end)
    tensor = Nx.tensor(data, type: :f32) |> Nx.reshape({n, 3})

    # Compute pairwise distances using broadcasting
    # dist[i,j] = ||pad_i - pad_j||
    # Expand: [n, 1, 3] - [1, n, 3] = [n, n, 3]
    t1 = Nx.reshape(tensor, {n, 1, 3})
    t2 = Nx.reshape(tensor, {1, n, 3})

    diff = Nx.subtract(t1, t2)
    dist_sq = Nx.sum(Nx.multiply(diff, diff), axes: [2])
    dist = Nx.sqrt(dist_sq)

    # Similarity = 1 - dist / max_dist
    # max_dist for PAD is sqrt(12) ≈ 3.464
    similarity = Nx.subtract(1.0, Nx.divide(dist, 3.464))

    # Convert to nested list
    similarity
    |> Nx.to_batched(1)
    |> Enum.map(fn row ->
      row |> Nx.squeeze() |> Nx.to_flat_list()
    end)
  end
end
