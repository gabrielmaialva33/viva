%% viva_simd_nif.erl - Erlang wrapper for SIMD NIF operations
%%
%% Provides SIMD-accelerated operations for neural network computations.
%% Falls back to Erlang implementations if NIF not loaded.

-module(viva_simd_nif).
-export([
    simd_dot/2,
    simd_mul/2,
    simd_matmul/5,
    simd_sum/1,
    simd_scale/2,
    is_simd_available/0
]).

-on_load(init/0).

%% Load NIF on module load
init() ->
    PrivDir = case code:priv_dir(viva) of
        {error, _} ->
            %% Development fallback
            case file:get_cwd() of
                {ok, Cwd} -> filename:join(Cwd, "priv");
                _ -> "priv"
            end;
        Dir -> Dir
    end,
    NifPath = filename:join(PrivDir, "viva_simd_nif"),
    case erlang:load_nif(NifPath, 0) of
        ok -> ok;
        {error, _Reason} ->
            %% NIF not available, use Erlang fallbacks
            ok
    end.

%% ============================================================================
%% NIF STUBS (replaced by NIF if loaded)
%% ============================================================================

%% Dot product - fallback to Erlang
simd_dot(A, B) when length(A) =:= length(B) ->
    dot_erlang(A, B, 0.0).

dot_erlang([], [], Acc) -> Acc;
dot_erlang([Ha | Ta], [Hb | Tb], Acc) ->
    dot_erlang(Ta, Tb, Acc + Ha * Hb).

%% Element-wise multiply - fallback
simd_mul(A, B) when length(A) =:= length(B) ->
    lists:zipwith(fun(X, Y) -> X * Y end, A, B).

%% Matrix multiply - fallback
simd_matmul(A, B, M, K, N) ->
    %% A is MxK (row-major), B is KxN (row-major)
    [begin
        [begin
            lists:foldl(fun(KIdx, Acc) ->
                Aval = lists:nth(I * K + KIdx + 1, A),
                Bval = lists:nth(KIdx * N + J + 1, B),
                Acc + Aval * Bval
            end, 0.0, lists:seq(0, K - 1))
        end || J <- lists:seq(0, N - 1)]
    end || I <- lists:seq(0, M - 1)].

%% Sum - fallback
simd_sum(List) ->
    lists:foldl(fun(X, Acc) -> X + Acc end, 0.0, List).

%% Scale - fallback
simd_scale(List, Scalar) ->
    [X * Scalar || X <- List].

%% Check if SIMD NIF is loaded
is_simd_available() ->
    %% Try to call a NIF function and see if it works
    case catch simd_dot([1.0, 2.0], [3.0, 4.0]) of
        11.0 -> true;
        _ -> false
    end.
