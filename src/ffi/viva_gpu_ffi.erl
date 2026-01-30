%% VIVA GPU FFI - GPU Detection and Operations
%% Stub implementation - always returns CPU mode (Pure Gleam)

-module(viva_gpu_ffi).
-export([cuda_available/0, exla_available/0, gpu_info/0]).
-export([batch_apply_delta/2, batch_scale/2, batch_lerp/3]).
-export([batch_matmul/2, batch_dense_forward/4, batch_resonance/1]).

%% Detection - Always return false/CPU for Pure Gleam
cuda_available() -> false.
exla_available() -> false.

gpu_info() ->
    #{available => false,
      name => <<"None">>,
      memory_mb => 0,
      compute_capability => <<"N/A">>}.

%% Batch operations - Return input unchanged (CPU fallback handles it)
batch_apply_delta(Batch, _Delta) -> Batch.
batch_scale(Batch, _Factor) -> Batch.
batch_lerp(Batch, _Target, _T) -> Batch.

%% Tensor operations - Return empty (CPU fallback handles it)
batch_matmul(_Inputs, _Weights) -> [].
batch_dense_forward(_Inputs, _Weights, _Biases, _Activation) -> [].
batch_resonance(_Pads) -> [].
