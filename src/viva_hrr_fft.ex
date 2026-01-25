# VivaNxHRR - FFT operations for Holographic Reduced Representations
#
# Uses Nx.fft for O(n log n) circular convolution.
# Required deps: {:nx, "~> 0.7"}

defmodule VivaNxHRR do
  @moduledoc """
  FFT-accelerated HRR operations using Nx.
  Provides circular convolution for bind/unbind operations.
  """

  @doc """
  Circular convolution via FFT.
  conv(a, b) = ifft(fft(a) * fft(b))
  """
  def circular_conv(a, b) when is_list(a) and is_list(b) do
    # Convert to Nx tensors
    t_a = Nx.tensor(a)
    t_b = Nx.tensor(b)

    # FFT both
    fft_a = Nx.fft(t_a)
    fft_b = Nx.fft(t_b)

    # Element-wise multiply in frequency domain
    product = Nx.multiply(fft_a, fft_b)

    # Inverse FFT
    result = Nx.ifft(product)

    # Take real part and convert to list
    result
    |> Nx.real()
    |> Nx.to_flat_list()
  end

  @doc """
  Circular correlation via FFT.
  corr(a, b) = ifft(fft(a) * conj(fft(b)))
  """
  def circular_corr(a, b) when is_list(a) and is_list(b) do
    t_a = Nx.tensor(a)
    t_b = Nx.tensor(b)

    fft_a = Nx.fft(t_a)
    fft_b = Nx.fft(t_b)

    # Conjugate of fft_b
    conj_fft_b = Nx.conjugate(fft_b)

    # Multiply
    product = Nx.multiply(fft_a, conj_fft_b)

    # Inverse FFT
    result = Nx.ifft(product)

    result
    |> Nx.real()
    |> Nx.to_flat_list()
  end

  @doc """
  Batch circular convolution for multiple pairs.
  More efficient than calling circular_conv repeatedly.
  """
  def batch_circular_conv(pairs) when is_list(pairs) do
    Enum.map(pairs, fn {a, b} -> circular_conv(a, b) end)
  end

  @doc """
  Normalize vector to unit length.
  """
  def normalize(v) when is_list(v) do
    t = Nx.tensor(v)
    norm = t |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt()

    case Nx.to_number(norm) > 0.0001 do
      true -> t |> Nx.divide(norm) |> Nx.to_flat_list()
      false -> v
    end
  end

  @doc """
  Cosine similarity between two vectors.
  """
  def cosine_similarity(a, b) when is_list(a) and is_list(b) do
    t_a = Nx.tensor(a)
    t_b = Nx.tensor(b)

    dot = Nx.dot(t_a, t_b) |> Nx.to_number()
    norm_a = t_a |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number()
    norm_b = t_b |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number()

    case norm_a * norm_b > 0.0001 do
      true -> dot / (norm_a * norm_b)
      false -> 0.0
    end
  end
end

# Erlang-compatible module name for @external calls
defmodule :viva_hrr_fft do
  @moduledoc """
  Erlang-compatible wrapper for VivaNxHRR.
  Called from Gleam via @external.
  """

  def circular_conv(a, b), do: VivaNxHRR.circular_conv(a, b)
  def circular_corr(a, b), do: VivaNxHRR.circular_corr(a, b)
  def normalize(v), do: VivaNxHRR.normalize(v)
  def cosine_similarity(a, b), do: VivaNxHRR.cosine_similarity(a, b)
end
