%% viva_hrr_fft - Pure Erlang fallback for HRR FFT operations
%%
%% Used when Nx is not available.
%% Implements naive O(n²) circular convolution.

-module(viva_hrr_fft).
-export([circular_conv/2, circular_corr/2, normalize/1, cosine_similarity/2]).

%% Circular convolution - naive O(n²) implementation
%% For production with large vectors, use the Elixir/Nx version
circular_conv(A, B) when is_list(A), is_list(B) ->
    N = length(A),
    AArr = list_to_tuple(A),
    BArr = list_to_tuple(B),
    [conv_element(K, AArr, BArr, N) || K <- lists:seq(0, N-1)].

conv_element(K, AArr, BArr, N) ->
    lists:foldl(
        fun(J, Acc) ->
            Aj = element(J+1, AArr),
            BIdx = ((K - J) rem N + N) rem N,
            Bkj = element(BIdx+1, BArr),
            Acc + (Aj * Bkj)
        end,
        0.0,
        lists:seq(0, N-1)
    ).

%% Circular correlation = convolution with reversed B (except B[0])
circular_corr(A, B) when is_list(A), is_list(B) ->
    BInv = approximate_inverse(B),
    circular_conv(A, BInv).

approximate_inverse([]) -> [];
approximate_inverse([First | Rest]) -> [First | lists:reverse(Rest)].

%% Normalize vector to unit length
normalize(V) when is_list(V) ->
    Norm = math:sqrt(lists:foldl(fun(X, Acc) -> Acc + X*X end, 0.0, V)),
    case Norm > 0.0001 of
        true -> [X / Norm || X <- V];
        false -> V
    end.

%% Cosine similarity
cosine_similarity(A, B) when is_list(A), is_list(B) ->
    Dot = lists:foldl(
        fun({Ai, Bi}, Acc) -> Acc + Ai * Bi end,
        0.0,
        lists:zip(A, B)
    ),
    NormA = math:sqrt(lists:foldl(fun(X, Acc) -> Acc + X*X end, 0.0, A)),
    NormB = math:sqrt(lists:foldl(fun(X, Acc) -> Acc + X*X end, 0.0, B)),
    case NormA * NormB > 0.0001 of
        true -> Dot / (NormA * NormB);
        false -> 0.0
    end.
