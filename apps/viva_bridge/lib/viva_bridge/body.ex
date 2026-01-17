defmodule VivaBridge.Body do
  @moduledoc """
  VIVA's Body - Hardware Sensing via Rust NIF (Cross-platform Interoception).
  Now powered by Bevy ECS (Headless).
  """

  @skip_nif System.get_env("VIVA_SKIP_NIF") == "true"

  unless @skip_nif do
    use Rustler,
      otp_app: :viva_bridge,
      crate: "viva_body"
  end

  # ============================================================================
  # Core Lifecycle (Singleton)
  # ============================================================================

  @doc "Checks if VIVA body is alive"
  def alive(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Ticks the Bevy ECS Body one frame.
  Returns map with complete state.
  """
  def body_tick(), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Polls the Soul Channel for updates (from Rust ECS)"
  def poll_channel(), do: :erlang.nif_error(:nif_not_loaded)

  # ============================================================================
  # Legacy / Compatibility Stubs
  # ============================================================================

  def get_cycles(), do: 0

  def feel_hardware() do
    if @skip_nif, do: %{}, else: body_tick()[:hardware]
  end

  def hardware_to_qualia(), do: {0.0, 0.0, 0.0}

  # Dynamics NIFs - Deprecated or Stubbed (ECS handles this internally now)
  def dynamics_ou_step(_p, _a, _d, _dt, _np, _na, _nd), do: {0.0, 0.0, 0.0}
  def dynamics_cusp_equilibria(_c, _y), do: [0.0]
  def dynamics_cusp_is_bifurcation(_c, _y), do: false
  def dynamics_cusp_mood_step(mood, _ar, _bias, _dt), do: mood
  def dynamics_step(p, a, d, _dt, _np, _na, _nd, _cusp, _sens, _bias), do: {p, a, d}

  # Body Engine legacy (Resource based) -> Mapped to Singleton
  def body_engine_new(), do: :singleton
  def body_engine_new_with_config(_dt, _cusp, _sens, _seed), do: :singleton

  def body_engine_tick(_engine) do
    # Redirect to singleton tick
    body_tick()
  end

  @doc "Applies emotional stimulus directly to ECS"
  def apply_stimulus(_p, _a, _d), do: :erlang.nif_error(:nif_not_loaded)

  def body_engine_get_pad(_engine), do: {0.0, 0.0, 0.0}
  def body_engine_set_pad(_engine, _p, _a, _d), do: :ok
  def body_engine_apply_stimulus(_engine, _p, _a, _d), do: :ok

  # ============================================================================
  # Memory NIFs (Still Active)
  # ============================================================================

  def memory_init(_path), do: :erlang.nif_error(:nif_not_loaded)
  def memory_store(_vector, _meta), do: :erlang.nif_error(:nif_not_loaded)
  def memory_search(_query, _limit), do: :erlang.nif_error(:nif_not_loaded)
  def memory_save(), do: :erlang.nif_error(:nif_not_loaded)
  def memory_stats(_backend), do: :erlang.nif_error(:nif_not_loaded)

  # ============================================================================
  # Metabolism / Brain / Mirror Stubs (to be migrated)
  # ============================================================================

  def metabolism_init(_tdp), do: {:ok, "Stub"}
  def metabolism_tick(_usage, _temp), do: {0.0, 0.0, 0.0, false}
  def brain_init(), do: :stub
  def brain_experience(_txt, _p, _a, _d), do: []
  def mirror_get_self(_path), do: nil
  def mirror_build_identity(), do: {"dev", "0.1.0", "unknown", 0}
  def mirror_list_modules(), do: []
  def mirror_capabilities(), do: {"bevy", "x86_64", true, false, false, false}
  def mirror_feature_flags(), do: {true, false, true, 10}
end
