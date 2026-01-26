%% viva_tensor_ffi.erl - NumPy-style O(1) array operations
%% Using Erlang :array for fast random access (like NumPy ndarray)
-module(viva_tensor_ffi).
-export([
    list_to_array/1,
    array_to_list/1,
    array_get/2,
    array_set/3,
    array_size/1,
    % Optimized ops
    array_map/2,
    array_map2/3,
    array_fold/3,
    array_dot/2,
    array_matmul/5,
    strided_get/4
]).

%% Convert list to array (O(n) once, then O(1) access)
list_to_array(List) ->
    array:from_list(List).

%% Convert array back to list
array_to_list(Array) ->
    array:to_list(Array).

%% O(1) random access - THE KEY OPTIMIZATION
array_get(Array, Index) ->
    array:get(Index, Array).

%% O(1) functional update (returns new array)
array_set(Array, Index, Value) ->
    array:set(Index, Value, Array).

%% Array size
array_size(Array) ->
    array:size(Array).

%% ============================================================================
%% OPTIMIZED OPERATIONS (avoid Gleam list overhead)
%% ============================================================================

%% Map over array elements (stays in array land)
array_map(Array, Fun) ->
    array:map(fun(_Idx, Val) -> Fun(Val) end, Array).

%% Map2: element-wise operation on two arrays
array_map2(A, B, Fun) ->
    Size = array:size(A),
    array:from_list([
        Fun(array:get(I, A), array:get(I, B))
        || I <- lists:seq(0, Size - 1)
    ]).

%% Fold over array
array_fold(Array, Acc0, Fun) ->
    array:foldl(fun(_Idx, Val, Acc) -> Fun(Acc, Val) end, Acc0, Array).

%% Dot product - CRITICAL for neural nets
array_dot(A, B) ->
    Size = array:size(A),
    dot_loop(A, B, 0, Size, 0.0).

dot_loop(_A, _B, Idx, Size, Acc) when Idx >= Size -> Acc;
dot_loop(A, B, Idx, Size, Acc) ->
    Val = array:get(Idx, A) * array:get(Idx, B),
    dot_loop(A, B, Idx + 1, Size, Acc + Val).

%% Matrix multiplication with strides - NumPy style!
%% matmul(A, B, M, N, K) where A is MxK, B is KxN
array_matmul(A, B, M, N, K) ->
    Result = [
        begin
            RowStart = I * K,
            lists:foldl(fun(KIdx, Acc) ->
                AVal = array:get(RowStart + KIdx, A),
                BVal = array:get(KIdx * N + J, B),
                Acc + AVal * BVal
            end, 0.0, lists:seq(0, K - 1))
        end
        || I <- lists:seq(0, M - 1),
           J <- lists:seq(0, N - 1)
    ],
    array:from_list(Result).

%% Strided access - THE CORE OF NumPy's POWER
%% Given strides [s0, s1, ...] and indices [i0, i1, ...], compute:
%% offset + i0*s0 + i1*s1 + ...
strided_get(Array, Offset, Strides, Indices) ->
    FlatIdx = compute_strided_index(Offset, Strides, Indices),
    array:get(FlatIdx, Array).

compute_strided_index(Offset, Strides, Indices) ->
    lists:foldl(
        fun({Stride, Idx}, Acc) -> Acc + Stride * Idx end,
        Offset,
        lists:zip(Strides, Indices)
    ).
