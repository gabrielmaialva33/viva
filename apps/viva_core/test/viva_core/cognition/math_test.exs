defmodule VivaCore.Cognition.MathTest do
  use ExUnit.Case
  alias VivaCore.Cognition.{Math, Concept}

  # For DSL tests
  use VivaCore.Cognition.DSL

  describe "Math Primitives" do
    test "similarity: identical vectors should return 1.0" do
      v = [1, 2, 3]
      assert_in_delta Math.similarity(v, v), 1.0, 0.0001
    end

    test "similarity: orthogonal vectors should return 0.0" do
      v1 = [1, 0]
      v2 = [0, 1]
      assert_in_delta Math.similarity(v1, v2), 0.0, 0.0001
    end

    test "similarity: opposite vectors should return -1.0" do
      v1 = [1, 0]
      v2 = [-1, 0]
      assert_in_delta Math.similarity(v1, v2), -1.0, 0.0001
    end

    test "add: vector addition" do
      v1 = [1, 2]
      v2 = [3, 4]
      result = Math.add(v1, v2)
      assert Nx.to_flat_list(result) == [4, 6]
    end

    test "sub: vector subtraction" do
      v1 = [3, 5]
      v2 = [1, 2]
      result = Math.sub(v1, v2)
      assert Nx.to_flat_list(result) == [2, 3]
    end

    test "analogy: simple arithmetic" do
      # King(5) - Man(2) + Woman(4) = 5-2+4 = 7
      # This is a mock 1D analogy
      # Man
      a = [2]
      # King
      b = [5]
      # Woman
      c = [4]
      expected = 7

      result = Math.analogy(a, b, c)
      assert Nx.to_flat_list(result) == [expected]
    end
  end

  describe "Cognitive DSL" do
    test "Operator Overloading: ~d + ~d" do
      # Create concepts with manually injected vectors for testing
      c1 = Concept.new("A", Nx.tensor([1, 2]))
      c2 = Concept.new("B", Nx.tensor([3, 4]))

      # Use DSL operators
      res = c1 + c2

      assert res.content == "Synthesized Concept"
      assert Nx.to_flat_list(res.vector) == [4, 6]
    end

    test "Operator Overloading: ~d - ~d" do
      c1 = Concept.new("A", Nx.tensor([5, 5]))
      c2 = Concept.new("B", Nx.tensor([2, 5]))

      res = c1 - c2

      assert res.content == "Analogy Result"
      assert Nx.to_flat_list(res.vector) == [3, 0]
    end

    test "Standard arithmetic still works" do
      assert 1 + 2 == 3
      assert 5 - 2 == 3
    end
  end
end
