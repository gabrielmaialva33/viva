%% VIVA Burn NIF Wrapper
%% GPU-accelerated batch neural forward using burn-rs (Rust)
%% RTX 4090 optimized: CUDA 13.1 + Tensor Cores
-module('Elixir.Viva.Burn.Native').

-export([
    burn_check/0,
    burn_batch_forward/3,
    burn_batch_dense/3,
    %% NES Operations (auto GPU/CPU)
    burn_perturb_weights/4,
    burn_nes_gradient/3,
    burn_update_weights/3,
    burn_nes_step/5,
    %% NES GPU-explicit
    burn_perturb_weights_gpu/4,
    burn_nes_gradient_gpu/3,
    burn_update_weights_gpu/3,
    burn_batch_update_weights_gpu/3,
    %% Activations
    burn_batch_sigmoid/1,
    burn_batch_tanh/1,
    burn_batch_relu/1,
    %% Benchmarks
    burn_benchmark/5,
    burn_benchmark_nes/3,
    %% Batch Physics
    burn_batch_physics_simulate/7,
    burn_batch_physics_simulate_with_spin/7,  %% Alias for simulate with spin support
    burn_batch_physics_create_tables/1,
    burn_batch_physics_calculate_fitness/8,
    burn_batch_physics_benchmark/3,
    %% Multi-shot Episode Simulation (KEY OPTIMIZATION)
    burn_batch_simulate_episodes/4,
    burn_batch_evaluate_episodes/4,
    %% Tensor Physics (GPU)
    burn_tensor_physics_simulate/7,
    burn_tensor_physics_benchmark/3,
    %% CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    burn_cma_es_init/3,
    burn_cma_es_init_auto/2,
    burn_cma_es_sample/2,
    burn_cma_es_update/3,
    burn_cma_es_step/4,
    burn_cma_es_get_mean/1,
    burn_cma_es_get_sigma/1,
    burn_cma_es_get_diagnostics/1,
    burn_cma_es_benchmark/3
]).

-on_load(init/0).

-define(NIF_PATH, "native/viva_burn/target/release/libviva_burn").

init() ->
    PrivDir = code:priv_dir(viva),
    NifPath = case PrivDir of
        {error, _} ->
            %% Development mode - use relative path
            ?NIF_PATH;
        Dir ->
            %% Try priv/native first, then fallback to priv
            Native = filename:join([Dir, "native", "libviva_burn"]),
            case filelib:is_file(Native ++ ".so") of
                true -> Native;
                false -> filename:join(Dir, "viva_burn")
            end
    end,
    erlang:load_nif(NifPath, 0).

%% =============================================================================
%% NIF stubs - replaced when NIF loads
%% =============================================================================

burn_check() ->
    erlang:nif_error(nif_not_loaded).

burn_batch_forward(_WeightsList, _InputsList, _Architecture) ->
    erlang:nif_error(nif_not_loaded).

burn_batch_dense(_WeightsBatch, _InputsBatch, _LayerSizes) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% NES Operations (auto-select GPU/CPU)
%% =============================================================================

%% Generate perturbations for NES gradient estimation
%% GPU: Uses cuRAND for parallel random generation
%% CPU: Falls back to Rayon + rand crate
burn_perturb_weights(_BaseWeights, _NumPerturbations, _StdDev, _Seed) ->
    erlang:nif_error(nif_not_loaded).

%% Compute NES gradient from fitness evaluations
%% GPU: Uses GEMM with Tensor Core acceleration
%% CPU: Falls back to Rayon parallel
burn_nes_gradient(_Perturbations, _Fitnesses, _StdDev) ->
    erlang:nif_error(nif_not_loaded).

%% Apply gradient update (SGD step)
burn_update_weights(_Weights, _Gradient, _LearningRate) ->
    erlang:nif_error(nif_not_loaded).

%% Complete NES step (gradient + update in one call)
%% More efficient as it minimizes CPU-GPU transfers
burn_nes_step(_BaseWeights, _Perturbations, _Fitnesses, _StdDev, _LearningRate) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% NES GPU-Explicit Operations (CUDA only)
%% =============================================================================

%% Force GPU perturbation generation
burn_perturb_weights_gpu(_BaseWeights, _NumPerturbations, _StdDev, _Seed) ->
    erlang:nif_error(nif_not_loaded).

%% Force GPU gradient computation
burn_nes_gradient_gpu(_Perturbations, _Fitnesses, _StdDev) ->
    erlang:nif_error(nif_not_loaded).

%% Force GPU weight update
burn_update_weights_gpu(_Weights, _Gradient, _LearningRate) ->
    erlang:nif_error(nif_not_loaded).

%% Batch weight updates on GPU (for population-based training)
burn_batch_update_weights_gpu(_WeightsBatch, _GradientsBatch, _LearningRate) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Activation Functions
%% =============================================================================

burn_batch_sigmoid(_Values) ->
    erlang:nif_error(nif_not_loaded).

burn_batch_tanh(_Values) ->
    erlang:nif_error(nif_not_loaded).

burn_batch_relu(_Values) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Benchmarks
%% =============================================================================

%% Benchmark batch forward pass
burn_benchmark(_PopSize, _InputSize, _HiddenSize, _OutputSize, _Iterations) ->
    erlang:nif_error(nif_not_loaded).

%% Benchmark NES operations (compares GPU vs CPU)
%% Returns formatted string with timing results
burn_benchmark_nes(_WeightCount, _NumPerturbations, _Iterations) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Batch Physics Simulation (NEW)
%% =============================================================================

%% Simulate multiple billiards tables in parallel
%% Args:
%%   PositionsX: [[float]] - ball X positions [batch, 8]
%%   PositionsZ: [[float]] - ball Z positions [batch, 8]
%%   VelocitiesX: [[float]] - ball X velocities [batch, 8]
%%   VelocitiesZ: [[float]] - ball Z velocities [batch, 8]
%%   Pocketed: [[float]] - pocketed flags [batch, 8]
%%   Shots: [[float]] - shots [batch, 4] (angle, power, english, elevation)
%%   MaxSteps: integer - maximum simulation steps
%%
%% Returns: {ok, {FinalPosX, FinalPosZ, FinalPocketed, StepsTaken}}
burn_batch_physics_simulate(_PositionsX, _PositionsZ, _VelocitiesX, _VelocitiesZ,
                           _Pocketed, _Shots, _MaxSteps) ->
    erlang:nif_error(nif_not_loaded).

%% Simulate with spin physics support (full spin physics)
%% Uses english and elevation to apply realistic spin effects
burn_batch_physics_simulate_with_spin(_PositionsX, _PositionsZ, _VelocitiesX, _VelocitiesZ,
                                      _Pocketed, _Shots, _MaxSteps) ->
    erlang:nif_error(nif_not_loaded).

%% Create initial state for batch of sinuca tables
%% Returns: {PositionsX, PositionsZ, VelocitiesX, VelocitiesZ, Pocketed}
burn_batch_physics_create_tables(_BatchSize) ->
    erlang:nif_error(nif_not_loaded).

%% Calculate fitness for batch simulation results
%% Returns: [{Fitness, HitAngle, ScatterRatio}]
burn_batch_physics_calculate_fitness(_BatchSize, _InitialPocketed, _FinalPocketed,
                                    _FinalPosX, _FinalPosZ, _InitialPosX,
                                    _InitialPosZ, _TargetBallIdx) ->
    erlang:nif_error(nif_not_loaded).

%% Benchmark batch physics performance
burn_batch_physics_benchmark(_BatchSize, _MaxSteps, _Iterations) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Multi-shot Episode Simulation (KEY OPTIMIZATION)
%% =============================================================================

%% Simulate complete multi-shot episodes for population of neural networks
%% This reduces 4800 NIF calls per generation to just 1!
%%
%% Args:
%%   PopulationWeights: [[float]] - weights [pop_size, weight_count]
%%   Architecture: [integer] - layer sizes [input, h1, h2, ..., output]
%%   ShotsPerEpisode: integer - number of shots to simulate per episode
%%   MaxStepsPerShot: integer - max physics steps per shot
%%
%% Returns: [{{Fitness, ShotsTaken, BallsPocketed, HitAngle, Scatter},
%%            {FinalPosX, FinalPosZ, FinalPocketed}}]
burn_batch_simulate_episodes(_PopulationWeights, _Architecture,
                             _ShotsPerEpisode, _MaxStepsPerShot) ->
    erlang:nif_error(nif_not_loaded).

%% Evaluate episodes (simplified API for QD training)
%% Returns only fitness + behavior descriptors
%%
%% Returns: [{Fitness, HitAngle, ScatterRatio}]
burn_batch_evaluate_episodes(_PopulationWeights, _Architecture,
                             _ShotsPerEpisode, _MaxStepsPerShot) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% Tensor Physics (GPU) - NEW
%% =============================================================================

%% GPU tensor-based physics simulation
burn_tensor_physics_simulate(_PositionsX, _PositionsZ, _VelocitiesX, _VelocitiesZ,
                             _Pocketed, _Shots, _MaxSteps) ->
    erlang:nif_error(nif_not_loaded).

%% Benchmark tensor physics
burn_tensor_physics_benchmark(_BatchSize, _MaxSteps, _Iterations) ->
    erlang:nif_error(nif_not_loaded).

%% =============================================================================
%% CMA-ES - Covariance Matrix Adaptation Evolution Strategy
%% =============================================================================

%% Initialize CMA-ES optimizer
%% Args:
%%   InitialMean: [float] - starting point in search space
%%   InitialSigma: float - initial step size (0.3 recommended for NN weights)
%%   Lambda: integer | nil - population size (nil = auto = 4 + 3*ln(n))
%%
%% Returns: ResourceArc handle to CMA-ES state
burn_cma_es_init(_InitialMean, _InitialSigma, _Lambda) ->
    erlang:nif_error(nif_not_loaded).

%% Initialize CMA-ES optimizer with auto lambda (4 + 3*ln(n))
%% Args:
%%   InitialMean: [float] - starting point in search space
%%   InitialSigma: float - initial step size
%%
%% Returns: ResourceArc handle to CMA-ES state
burn_cma_es_init_auto(_InitialMean, _InitialSigma) ->
    erlang:nif_error(nif_not_loaded).

%% Sample new population from CMA-ES distribution
%% Args:
%%   CmaState: ResourceArc handle
%%   Seed: integer - random seed for reproducibility
%%
%% Returns: [[float]] - [lambda, n] population of candidate solutions
burn_cma_es_sample(_CmaState, _Seed) ->
    erlang:nif_error(nif_not_loaded).

%% Update CMA-ES state with fitness evaluations
%% Args:
%%   CmaState: ResourceArc handle
%%   Population: [[float]] - [lambda, n] candidate solutions from sample
%%   Fitnesses: [float] - [lambda] fitness values (higher = better)
%%
%% Returns: true on success
burn_cma_es_update(_CmaState, _Population, _Fitnesses) ->
    erlang:nif_error(nif_not_loaded).

%% Complete CMA-ES step: update (if provided) then sample
%% Args:
%%   CmaState: ResourceArc handle
%%   Population: [[float]] - previous population (can be empty)
%%   Fitnesses: [float] - previous fitnesses (can be empty)
%%   Seed: integer - random seed
%%
%% Returns: [[float]] - new population to evaluate
burn_cma_es_step(_CmaState, _Population, _Fitnesses, _Seed) ->
    erlang:nif_error(nif_not_loaded).

%% Get current mean (best estimate) from CMA-ES
%% Returns: [float] - current mean vector
burn_cma_es_get_mean(_CmaState) ->
    erlang:nif_error(nif_not_loaded).

%% Get current step size (sigma) from CMA-ES
%% Returns: float - current sigma value
burn_cma_es_get_sigma(_CmaState) ->
    erlang:nif_error(nif_not_loaded).

%% Get CMA-ES convergence diagnostics
%% Returns: {Sigma, ConditionNumber, NormalizedPsNorm}
burn_cma_es_get_diagnostics(_CmaState) ->
    erlang:nif_error(nif_not_loaded).

%% Benchmark CMA-ES operations
%% Returns: string - benchmark report
burn_cma_es_benchmark(_N, _Lambda, _Iterations) ->
    erlang:nif_error(nif_not_loaded).
