%% VIVA LLM NIF Wrapper
%% llama.cpp integration for knowledge distillation
%% Extracts hidden states and logits from teacher models (Qwen3-32B)
-module('Elixir.Viva.Llm.Native').

-export([
    %% Model management
    llm_load_model/2,
    llm_model_info/1,
    %% Hidden state extraction (for DHC distillation)
    llm_get_hidden_states/3,
    llm_get_logits/3,
    %% Legacy/convenience
    llm_predict/2,
    %% Tokenization
    llm_tokenize/2,
    llm_detokenize/2,
    %% System
    llm_memory_status/0,
    llm_native_check/0
]).

-on_load(init/0).

-define(NIF_PATH, "native/viva_llm/target/release/libviva_llm").

init() ->
    PrivDir = code:priv_dir(viva),
    NifPath = case PrivDir of
        {error, _} ->
            %% Development mode - use relative path
            ?NIF_PATH;
        Dir ->
            %% Try priv/native first, then fallback to priv
            Native = filename:join([Dir, "native", "libviva_llm"]),
            case filelib:is_file(Native ++ ".so") of
                true -> Native;
                false -> filename:join(Dir, "viva_llm")
            end
    end,
    erlang:load_nif(NifPath, 0).

%% =============================================================================
%% Model Management
%% =============================================================================

%% Load a GGUF model into GPU memory
%% Args:
%%   Path: string - absolute path to .gguf file
%%   GpuLayers: integer - number of layers to offload to GPU (0 = CPU only)
%%
%% Returns: ResourceArc handle to model
llm_load_model(_Path, _GpuLayers) ->
    erlang:nif_error(nif_not_loaded).

%% Get model dimensions
%% Returns: {NEmbedding, NVocab}
llm_model_info(_ModelResource) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Hidden State Extraction (DHC Distillation)
%% =============================================================================

%% Extract hidden states from the last layer for a given prompt
%% This is the KEY function for knowledge distillation
%%
%% Args:
%%   ModelResource: ResourceArc handle from llm_load_model
%%   Prompt: string - input text
%%   CtxSize: integer - context window size (e.g., 2048, 4096)
%%
%% Returns: {Embeddings, NTokens}
%%   Embeddings: [float] - flattened [n_tokens, n_embd]
%%   NTokens: integer - number of tokens processed
llm_get_hidden_states(_ModelResource, _Prompt, _CtxSize) ->
    erlang:nif_error(nif_not_loaded).

%% Get logits (output probabilities) for the last token
%% Useful for soft-label distillation (temperature scaling)
%%
%% Args:
%%   ModelResource: ResourceArc handle
%%   Prompt: string - input text
%%   CtxSize: integer - context window size
%%
%% Returns: [float] - logits of shape [n_vocab]
llm_get_logits(_ModelResource, _Prompt, _CtxSize) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Legacy/Convenience
%% =============================================================================

%% Legacy predict function - returns embedding of last token
%% Returns: {Status, Embedding}
llm_predict(_ModelResource, _Prompt) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Tokenization
%% =============================================================================

%% Tokenize text to token IDs
%% Returns: [integer] - token IDs
llm_tokenize(_ModelResource, _Text) ->
    erlang:nif_error(nif_not_loaded).

%% Detokenize token IDs back to text
%% Returns: string
llm_detokenize(_ModelResource, _TokenIds) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% System
%% =============================================================================

%% Get system memory status for proprioception
%% Returns: {TotalMemory, FreeMemory} in bytes
llm_memory_status() ->
    erlang:nif_error(nif_not_loaded).

%% Check native capabilities (AVX512, AVX2, etc.)
%% Returns: string
llm_native_check() ->
    erlang:nif_error(nif_not_loaded).
