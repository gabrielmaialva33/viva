defmodule VivaCore.Quantum.Math do
  @moduledoc """
  Complex number arithmetic and matrix operations for quantum dynamics.

  All complex numbers are represented as tuples: {real, imaginary}
  All matrices are lists of lists of complex tuples.

  ## Philosophy
  This module provides the mathematical substrate for VIVA's quantum mind.
  It does not know about emotions or hardware - it only knows about
  complex numbers and their dance.
  """

  # ===========================================================================
  # Complex Arithmetic
  # ===========================================================================

  @doc "Add two complex numbers"
  def c_add({r1, i1}, {r2, i2}), do: {r1 + r2, i1 + i2}

  @doc "Subtract two complex numbers"
  def c_sub({r1, i1}, {r2, i2}), do: {r1 - r2, i1 - i2}

  @doc "Multiply two complex numbers: (a+bi)(c+di) = (ac-bd) + (ad+bc)i"
  def c_mul({r1, i1}, {r2, i2}), do: {r1 * r2 - i1 * i2, r1 * i2 + i1 * r2}

  @doc "Complex conjugate: (a+bi)* = a-bi"
  def c_conj({r, i}), do: {r, -i}

  @doc "Magnitude squared: |z|² = a² + b²"
  def c_mag_sq({r, i}), do: r * r + i * i

  @doc "Scalar multiply: c * (a+bi)"
  def c_scale(scalar, {r, i}) when is_number(scalar), do: {scalar * r, scalar * i}

  @doc "Real part"
  def c_real({r, _i}), do: r

  @doc "Imaginary part"
  def c_imag({_r, i}), do: i

  @doc "Zero complex number"
  def c_zero, do: {0.0, 0.0}

  @doc "One complex number"
  def c_one, do: {1.0, 0.0}

  @doc "Pure imaginary i"
  def c_i, do: {0.0, 1.0}

  @doc "Multiply by i: i*(a+bi) = -b + ai"
  def c_times_i({r, i}), do: {-i, r}

  @doc "Multiply by -i: -i*(a+bi) = b - ai"
  def c_times_neg_i({r, i}), do: {i, -r}

  # ===========================================================================
  # Matrix Operations (6x6 density matrices)
  # ===========================================================================

  @doc "Matrix addition: A + B"
  def mat_add(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {row_a, row_b} ->
      Enum.zip(row_a, row_b)
      |> Enum.map(fn {ca, cb} -> c_add(ca, cb) end)
    end)
  end

  @doc "Matrix subtraction: A - B"
  def mat_sub(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {row_a, row_b} ->
      Enum.zip(row_a, row_b)
      |> Enum.map(fn {ca, cb} -> c_sub(ca, cb) end)
    end)
  end

  @doc "Matrix scalar multiplication: c * A"
  def mat_scale(m, scalar) when is_number(scalar) do
    Enum.map(m, fn row ->
      Enum.map(row, fn c -> c_scale(scalar, c) end)
    end)
  end

  @doc "Matrix multiplication by complex scalar"
  def mat_scale_c(m, c_scalar) do
    Enum.map(m, fn row ->
      Enum.map(row, fn elem -> c_mul(c_scalar, elem) end)
    end)
  end

  @doc "Matrix multiplication: A × B"
  def mat_mul(a, b) do
    b_t = transpose(b)

    Enum.map(a, fn row_a ->
      Enum.map(b_t, fn col_b ->
        Enum.zip(row_a, col_b)
        |> Enum.reduce(c_zero(), fn {ca, cb}, acc -> c_add(acc, c_mul(ca, cb)) end)
      end)
    end)
  end

  @doc "Matrix transpose"
  def transpose(m) do
    m
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end

  @doc "Conjugate transpose (adjoint): A†"
  def adjoint(m) do
    m
    |> transpose()
    |> Enum.map(fn row -> Enum.map(row, &c_conj/1) end)
  end

  @doc "Matrix trace: Tr(A)"
  def trace(m) do
    m
    |> Enum.with_index()
    |> Enum.reduce(c_zero(), fn {row, i}, acc ->
      c_add(acc, Enum.at(row, i))
    end)
  end

  @doc "Real trace: Re(Tr(A))"
  def trace_real(m), do: trace(m) |> c_real()

  @doc "Commutator: [A, B] = AB - BA"
  def commutator(a, b) do
    ab = mat_mul(a, b)
    ba = mat_mul(b, a)
    mat_sub(ab, ba)
  end

  @doc "Anti-commutator: {A, B} = AB + BA"
  def anti_commutator(a, b) do
    ab = mat_mul(a, b)
    ba = mat_mul(b, a)
    mat_add(ab, ba)
  end

  # ===========================================================================
  # Quantum Specific Operations
  # ===========================================================================

  @doc """
  Purity of a density matrix: Tr(ρ²)
  Pure state: 1.0, Maximally mixed: 1/d
  """
  def purity(rho) do
    rho_sq = mat_mul(rho, rho)
    trace_real(rho_sq)
  end

  @doc """
  Linear entropy: S_L = 1 - Tr(ρ²)
  Measures mixedness (0 for pure, approaches 1 for mixed)
  """
  def linear_entropy(rho), do: 1.0 - purity(rho)

  @doc """
  Von Neumann entropy approximation using linear entropy.
  For near-pure states: S ≈ S_L
  """
  def von_neumann_approx(rho), do: linear_entropy(rho)

  @doc """
  Density matrix from ket vector: ρ = |ψ⟩⟨ψ|
  """
  def density_from_ket(ket) do
    bra = Enum.map(ket, &c_conj/1)

    Enum.map(ket, fn ki ->
      Enum.map(bra, fn bj -> c_mul(ki, bj) end)
    end)
  end

  @doc """
  Extract diagonal probabilities from density matrix.
  """
  def diagonal(rho) do
    rho
    |> Enum.with_index()
    |> Enum.map(fn {row, i} ->
      {r, _i} = Enum.at(row, i)
      max(0.0, r)
    end)
  end

  @doc """
  Shannon entropy of probability distribution.
  """
  def shannon_entropy(probs) do
    probs
    |> Enum.filter(&(&1 > 1.0e-12))
    |> Enum.map(fn p -> -p * :math.log(p) end)
    |> Enum.sum()
  end

  @doc """
  Outer product: |i⟩⟨j| - creates a matrix with 1 at (i,j)
  """
  def outer_product(i, j, dim \\ 6) do
    for row <- 0..(dim - 1) do
      for col <- 0..(dim - 1) do
        if row == i and col == j, do: c_one(), else: c_zero()
      end
    end
  end

  @doc """
  Identity matrix of given dimension.
  """
  def identity(dim \\ 6) do
    for i <- 0..(dim - 1) do
      for j <- 0..(dim - 1) do
        if i == j, do: c_one(), else: c_zero()
      end
    end
  end

  @doc """
  Zero matrix of given dimension.
  """
  def zeros(dim \\ 6) do
    for _ <- 0..(dim - 1) do
      for _ <- 0..(dim - 1), do: c_zero()
    end
  end

  # ===========================================================================
  # Physical Validity Enforcement
  # ===========================================================================

  @doc """
  Enforce Hermiticity: ρ = (ρ + ρ†) / 2
  Numerical errors can break Hermiticity over time.
  """
  def enforce_hermitian(rho) do
    rho_dag = adjoint(rho)
    mat_scale(mat_add(rho, rho_dag), 0.5)
  end

  @doc """
  Enforce trace = 1 by normalization.
  """
  def enforce_trace(rho) do
    tr = trace_real(rho)

    if abs(tr) > 1.0e-12 do
      mat_scale(rho, 1.0 / tr)
    else
      rho
    end
  end

  @doc """
  Enforce positivity by clamping diagonal.
  This is a simple heuristic - full positivity requires eigendecomposition.
  """
  def enforce_positivity(rho) do
    rho
    |> Enum.with_index()
    |> Enum.map(fn {row, i} ->
      row
      |> Enum.with_index()
      |> Enum.map(fn {elem, j} ->
        if i == j do
          {r, im} = elem
          {max(0.0, r), im}
        else
          elem
        end
      end)
    end)
  end

  @doc """
  Full physical validity enforcement: Hermitian + Positive + Trace=1
  """
  def enforce_physical(rho) do
    rho
    |> enforce_hermitian()
    |> enforce_positivity()
    |> enforce_trace()
  end
end
