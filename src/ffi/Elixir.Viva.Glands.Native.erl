%% VIVA Glands NIF Wrapper
%% Loads the Rust NIF for GPU-accelerated HRR operations (CUDA + Candle)
-module('Elixir.Viva.Glands.Native').
-export([
    glands_init_gleam/4,
    glands_init/1,
    glands_load_model/2,
    glands_check/0,
    glands_project/2,
    glands_bind/3,
    glands_unbind/3,
    glands_similarity/2,
    glands_batch_similarity/2,
    glands_superpose/1,
    glands_benchmark/2
]).
-on_load(init/0).

-define(NIF_PATH, "native/viva_glands/target/release/libviva_glands").

init() ->
    PrivDir = code:priv_dir(viva),
    NifPath = case PrivDir of
        {error, _} ->
            %% Development mode - use relative path
            ?NIF_PATH;
        Dir ->
            filename:join(Dir, "native/libviva_glands")
    end,
    erlang:load_nif(NifPath, 0).

%% Public API for Gleam - converts params to Elixir struct for NIF
glands_init_gleam(LlmDim, HrrDim, Seed, GpuLayers) ->
    %% Create Elixir struct expected by rustler NifStruct
    Config = #{'__struct__' => 'Elixir.Viva.Glands.Config',
               llm_dim => LlmDim,
               hrr_dim => HrrDim,
               seed => Seed,
               gpu_layers => GpuLayers},
    glands_init(Config).

%% NIF stub - replaced when NIF loads
glands_init(_Config) ->
    erlang:nif_error(nif_not_loaded).

glands_load_model(_Resource, _ModelPath) ->
    erlang:nif_error(nif_not_loaded).

glands_check() ->
    erlang:nif_error(nif_not_loaded).

glands_project(_Handle, _Embedding) ->
    erlang:nif_error(nif_not_loaded).

glands_bind(_Handle, _A, _B) ->
    erlang:nif_error(nif_not_loaded).

glands_unbind(_Handle, _Trace, _Key) ->
    erlang:nif_error(nif_not_loaded).

glands_similarity(_A, _B) ->
    erlang:nif_error(nif_not_loaded).

glands_batch_similarity(_Vectors, _Query) ->
    erlang:nif_error(nif_not_loaded).

glands_superpose(_Vectors) ->
    erlang:nif_error(nif_not_loaded).

glands_benchmark(_Handle, _Iterations) ->
    erlang:nif_error(nif_not_loaded).
