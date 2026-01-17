defmodule VivaBridge.Body do
  @moduledoc """
  VIVA's Body - Hardware Sensing via Rust NIF (Cross-platform Interoception).
  Now powered by Bevy ECS (Headless).

  ## Architecture
  - Singleton Bevy App runs at 60Hz internally
  - Elixir polls via `body_tick/0` at BodyServer's rate (500ms)
  - Soul Channel provides async events via `poll_channel/0`

  ## Active NIFs
  - `alive/0`, `body_tick/0`, `poll_channel/0`, `apply_stimulus/3`
  - `metabolism_init/1`, `metabolism_tick/2`
  - `memory_*` functions

  ## Deprecated Stubs
  All `dynamics_*` and `body_engine_*` functions are deprecated.
  Dynamics now run inside Bevy ECS (Rust side).
  """

  @skip_nif System.get_env("VIVA_SKIP_NIF") == "true"

  unless @skip_nif do
    use Rustler,
      otp_app: :viva_bridge,
      crate: "viva_body"
  end

  # ============================================================================
  # Core Lifecycle (Active NIFs)
  # ============================================================================

  @doc "Checks if VIVA body is alive"
  def alive(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Ticks the Bevy ECS Body one frame.
  Returns BodyState with PAD, hardware metrics, and stress level.
  """
  def body_tick(), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Polls the Soul Channel for async events from Rust ECS"
  def poll_channel(), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Applies emotional stimulus directly to Bevy ECS"
  def apply_stimulus(_p, _a, _d), do: :erlang.nif_error(:nif_not_loaded)

  # ============================================================================
  # Metabolism NIFs (Active)
  # ============================================================================

  def metabolism_init(_tdp), do: :erlang.nif_error(:nif_not_loaded)
  def metabolism_tick(_usage, _temp), do: :erlang.nif_error(:nif_not_loaded)

  # ============================================================================
  # Memory NIFs (Active)
  # ============================================================================

  def memory_init(_path), do: :erlang.nif_error(:nif_not_loaded)
  def memory_store(_vector, _meta), do: :erlang.nif_error(:nif_not_loaded)
  def memory_search(_query, _limit), do: :erlang.nif_error(:nif_not_loaded)
  def memory_save(), do: :erlang.nif_error(:nif_not_loaded)
  def memory_stats(_backend), do: :erlang.nif_error(:nif_not_loaded)

  # ============================================================================
  # Convenience Functions (Elixir wrappers)
  # ============================================================================

  @doc "Get hardware state from last tick"
  def feel_hardware() do
    if @skip_nif, do: %{}, else: body_tick()[:hardware]
  end

  # ============================================================================
  # DEPRECATED - Legacy Stubs (DO NOT USE)
  # These are kept for backwards compatibility but are non-functional.
  # All dynamics now run in Bevy ECS (Rust side).
  # ============================================================================

  @deprecated "Use apply_stimulus/3 instead - O-U dynamics run in Bevy ECS"
  def dynamics_ou_step(_p, _a, _d, _dt, _np, _na, _nd), do: {0.0, 0.0, 0.0}

  @deprecated "Cusp dynamics run in Bevy ECS"
  def dynamics_cusp_equilibria(_c, _y), do: [0.0]

  @deprecated "Cusp dynamics run in Bevy ECS"
  def dynamics_cusp_is_bifurcation(_c, _y), do: false

  @deprecated "Cusp dynamics run in Bevy ECS"
  def dynamics_cusp_mood_step(mood, _ar, _bias, _dt), do: mood

  @deprecated "Use body_tick/0 instead - dynamics run in Bevy ECS"
  def dynamics_step(p, a, d, _dt, _np, _na, _nd, _cusp, _sens, _bias), do: {p, a, d}

  @deprecated "Use body_tick/0 directly - singleton pattern"
  def body_engine_new(), do: :singleton

  @deprecated "Use body_tick/0 directly - singleton pattern"
  def body_engine_new_with_config(_dt, _cusp, _sens, _seed), do: :singleton

  @deprecated "Use body_tick/0 directly"
  def body_engine_tick(_engine), do: body_tick()

  @deprecated "Use body_tick/0 - returns PAD in state"
  def body_engine_get_pad(_engine), do: {0.0, 0.0, 0.0}

  @deprecated "Use apply_stimulus/3 instead"
  def body_engine_set_pad(_engine, _p, _a, _d), do: :ok

  @deprecated "Use apply_stimulus/3 instead"
  def body_engine_apply_stimulus(_engine, _p, _a, _d), do: :ok

  @deprecated "Not implemented - returns zeros"
  def hardware_to_qualia(), do: {0.0, 0.0, 0.0}

  @deprecated "Not implemented"
  def get_cycles(), do: 0

  # ============================================================================
  # Stubs to be migrated (placeholders for future NIFs)
  # ============================================================================

  @doc false
  def brain_init(), do: :stub

  @doc false
  def brain_experience(_txt, _p, _a, _d), do: []

  @doc false
  def mirror_get_self(_path), do: nil

  @doc false
  def mirror_build_identity(), do: {"dev", "0.1.0", "unknown", 0}

  @doc false
  def mirror_list_modules(), do: []

  @doc false
  def mirror_capabilities(), do: {"bevy", "x86_64", true, false, false, false}

  @doc false
  def mirror_feature_flags(), do: {true, false, true, 10}
end
