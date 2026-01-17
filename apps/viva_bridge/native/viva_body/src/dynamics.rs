//! Stochastic dynamics for emotional modeling.
//!
//! This module implements mathematical models from affective neuroscience:
//!
//! - **Ornstein-Uhlenbeck Process**: Mean-reverting stochastic dynamics for PAD emotions
//! - **Cusp Catastrophe**: Bifurcation model for sudden emotional transitions (hysteresis)
//!
//! ## References
//!
//! - Ornstein & Uhlenbeck (1930): Original O-U process for Brownian motion with friction
//! - Särkkä & Solin (2019): Applied Stochastic Differential Equations (Cambridge)
//! - Zeeman (1977): Catastrophe Theory - Selected Papers
//! - Wagenmakers et al. (2005): Fitting the Cusp Catastrophe Model
//! - Sashin (1985): Affect tolerance using catastrophe theory

/// Parameters for Ornstein-Uhlenbeck process.
///
/// The O-U process models mean-reverting behavior:
/// ```text
/// dX_t = θ(μ - X_t)dt + σdW_t
/// ```
///
/// Where:
/// - θ (theta): Mean-reversion speed (how fast it returns to equilibrium)
/// - μ (mu): Long-term mean (equilibrium point)
/// - σ (sigma): Volatility (noise amplitude)
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OUParams {
    /// Mean-reversion speed (θ). Higher = faster return to equilibrium.
    /// Typical range: 0.1 to 10.0
    pub theta: f64,
    /// Long-term equilibrium value (μ).
    /// For PAD emotions, typically 0.0 (neutral).
    pub mu: f64,
    /// Volatility / noise amplitude (σ).
    /// Higher = more random fluctuation.
    pub sigma: f64,
}

impl Default for OUParams {
    fn default() -> Self {
        // Half-life formula: t½ = ln(2)/θ
        // θ = 0.0154 → t½ ≈ 45s (psychologically realistic for PAD emotions)
        // Matched with Elixir: VivaCore.Emotional @base_decay_rate
        Self {
            theta: 0.0154, // ~45s half-life (realistic emotional decay)
            mu: 0.0,       // Neutral equilibrium
            sigma: 0.01,   // Low noise (matched with Elixir)
        }
    }
}

impl OUParams {
    /// Create params for fast emotional recovery (high theta).
    /// t½ ≈ 15s - quick bounce-back after stimuli
    pub fn fast_recovery() -> Self {
        Self {
            theta: 0.046, // t½ ≈ 15s
            mu: 0.0,
            sigma: 0.005,
        }
    }

    /// Create params for slow emotional drift (low theta).
    /// t½ ≈ 2min - emotions persist longer (e.g., melancholy)
    pub fn slow_drift() -> Self {
        Self {
            theta: 0.0058, // t½ ≈ 120s
            mu: 0.0,
            sigma: 0.02,
        }
    }

    /// Create params for volatile emotions (high sigma).
    /// Normal decay but more noise - emotional instability
    pub fn volatile() -> Self {
        Self {
            theta: 0.0154, // t½ ≈ 45s (same as default)
            mu: 0.0,
            sigma: 0.05,  // 5x more noise
        }
    }
}

/// Ornstein-Uhlenbeck process step using Euler-Maruyama discretization.
///
/// Updates `state` in-place using:
/// ```text
/// X_{t+dt} = X_t + θ(μ - X_t)·dt + σ·√dt·ε
/// ```
///
/// Where ε ~ N(0,1) is provided as `noise`.
///
/// # Arguments
///
/// * `state` - Current state value (modified in-place)
/// * `params` - O-U parameters (theta, mu, sigma)
/// * `dt` - Time step in seconds
/// * `noise` - Random sample from N(0,1)
///
/// # Example
///
/// ```rust,ignore
/// let mut x = 0.5; // Current pleasure
/// let params = OUParams::default();
/// let noise = rand::random::<f64>() * 2.0 - 1.0; // Approximate N(0,1)
/// ou_step(&mut x, &params, 0.016, noise); // 60Hz update
/// ```
#[inline]
pub fn ou_step(state: &mut f64, params: &OUParams, dt: f64, noise: f64) {
    let drift = params.theta * (params.mu - *state);
    let diffusion = params.sigma * dt.sqrt() * noise;
    *state += drift * dt + diffusion;
}

/// Ornstein-Uhlenbeck step for 3D PAD state.
///
/// Updates all three dimensions (Pleasure, Arousal, Dominance) independently.
/// Each dimension can have different parameters.
///
/// # Arguments
///
/// * `pad` - [pleasure, arousal, dominance] array (modified in-place)
/// * `params` - Array of O-U parameters for each dimension
/// * `dt` - Time step in seconds
/// * `noise` - Array of N(0,1) samples for each dimension
#[inline]
pub fn ou_step_pad(pad: &mut [f64; 3], params: &[OUParams; 3], dt: f64, noise: &[f64; 3]) {
    for i in 0..3 {
        ou_step(&mut pad[i], &params[i], dt, noise[i]);
    }
}

/// Ornstein-Uhlenbeck step with bounds clamping.
///
/// Same as `ou_step` but clamps result to [min, max].
/// Useful for PAD where values must stay in [-1, 1].
#[inline]
pub fn ou_step_clamped(
    state: &mut f64,
    params: &OUParams,
    dt: f64,
    noise: f64,
    min: f64,
    max: f64,
) {
    ou_step(state, params, dt, noise);
    *state = state.clamp(min, max);
}

/// Ornstein-Uhlenbeck step for bounded PAD state [-1, 1].
#[inline]
pub fn ou_step_pad_bounded(pad: &mut [f64; 3], params: &[OUParams; 3], dt: f64, noise: &[f64; 3]) {
    ou_step_pad(pad, params, dt, noise);
    for x in pad.iter_mut() {
        *x = x.clamp(-1.0, 1.0);
    }
}

// ============================================================================
// CUSP CATASTROPHE MODEL
// ============================================================================

/// Cusp catastrophe potential function.
///
/// The cusp model captures sudden transitions (bifurcations) in behavior:
/// ```text
/// V(x) = x⁴/4 - c·x²/2 - y·x
/// ```
///
/// Where:
/// - x: State variable (e.g., mood level)
/// - c: Control parameter (splitting factor) - determines if bifurcation exists
/// - y: Asymmetry parameter (external bias) - pushes toward one attractor
///
/// Equilibria occur where dV/dx = 0:
/// ```text
/// x³ - c·x - y = 0
/// ```
///
/// # Behavior
///
/// - c ≤ 0: Single stable equilibrium (smooth behavior)
/// - c > 0: Potential bifurcation region (can have 2 stable + 1 unstable)
///
/// # Reference
///
/// Zeeman (1977), Wagenmakers et al. (2005)
#[inline]
pub fn cusp_potential(x: f64, c: f64, y: f64) -> f64 {
    x.powi(4) / 4.0 - c * x.powi(2) / 2.0 - y * x
}

/// Cusp catastrophe gradient (negative force).
///
/// Returns dV/dx = x³ - c·x - y
///
/// The system evolves toward equilibrium where gradient = 0.
#[inline]
pub fn cusp_gradient(x: f64, c: f64, y: f64) -> f64 {
    x.powi(3) - c * x - y
}

/// Cusp catastrophe dynamics step.
///
/// Evolves state toward equilibrium using gradient descent:
/// ```text
/// dx/dt = -dV/dx = -(x³ - c·x - y) = -x³ + c·x + y
/// ```
///
/// # Arguments
///
/// * `x` - Current state (modified in-place)
/// * `c` - Control parameter (splitting factor)
/// * `y` - Asymmetry parameter (bias)
/// * `dt` - Time step
/// * `damping` - Damping factor (1.0 = no damping, <1.0 = slower)
///
/// # Example
///
/// ```rust,ignore
/// let mut mood = 0.0;
/// // High control = bifurcation possible, slight positive bias
/// cusp_step(&mut mood, 2.0, 0.1, 0.016, 1.0);
/// ```
#[inline]
pub fn cusp_step(x: &mut f64, c: f64, y: f64, dt: f64, damping: f64) {
    let gradient = cusp_gradient(*x, c, y);
    *x -= damping * gradient * dt;
}

/// Cusp catastrophe step with noise (stochastic cusp).
///
/// Combines cusp dynamics with stochastic noise:
/// ```text
/// dx = (-x³ + c·x + y)dt + σ·dW
/// ```
#[inline]
pub fn cusp_step_stochastic(x: &mut f64, c: f64, y: f64, dt: f64, sigma: f64, noise: f64) {
    let gradient = cusp_gradient(*x, c, y);
    let diffusion = sigma * dt.sqrt() * noise;
    *x += (-gradient * dt) + diffusion;
}

/// Find cusp equilibria analytically (approximate).
///
/// Solves x³ - c·x - y = 0 using Cardano's formula (simplified).
///
/// Returns vector of equilibrium points (1 or 3 depending on c, y).
pub fn cusp_equilibria(c: f64, y: f64) -> Vec<f64> {
    // Discriminant determines number of real roots
    // For x³ - cx - y = 0, discriminant = 4c³ - 27y²
    let discriminant = 4.0 * c.powi(3) - 27.0 * y.powi(2);

    if c <= 0.0 || discriminant <= 0.0 {
        // One real root - use Newton-Raphson
        let mut x = y.signum() * y.abs().cbrt(); // Initial guess
        for _ in 0..10 {
            let f = x.powi(3) - c * x - y;
            let df = 3.0 * x.powi(2) - c;
            if df.abs() < 1e-10 {
                break;
            }
            x -= f / df;
        }
        vec![x]
    } else {
        // Three real roots in bifurcation region
        // Use trigonometric solution for depressed cubic
        let p = -c;
        let q = -y;
        let m = 2.0 * (-p / 3.0).sqrt();
        // Clamp to [-1, 1] to avoid NaN from acos due to floating-point errors
        let acos_arg = (3.0 * q / (p * m)).clamp(-1.0, 1.0);
        let theta = acos_arg.acos() / 3.0;

        let x1 = m * theta.cos();
        let x2 = m * (theta - 2.0 * std::f64::consts::PI / 3.0).cos();
        let x3 = m * (theta + 2.0 * std::f64::consts::PI / 3.0).cos();

        let mut roots = vec![x1, x2, x3];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        roots
    }
}

/// Check if state is in cusp bifurcation region.
///
/// Returns true if the cusp has multiple equilibria (hysteresis possible).
#[inline]
pub fn cusp_is_bifurcation(c: f64, y: f64) -> bool {
    c > 0.0 && 4.0 * c.powi(3) > 27.0 * y.powi(2)
}

/// Cusp catastrophe for emotional hysteresis.
///
/// Models sudden mood transitions where:
/// - `arousal` acts as control parameter (high arousal = bifurcation)
/// - `external_stress` acts as asymmetry (bias toward negative)
/// - `mood` is the state that can jump suddenly
///
/// # Example: Depression-Mania Transition
///
/// ```rust,ignore
/// let mut mood = -0.8; // Depressed
/// let arousal = 2.5;   // High arousal (crisis)
/// let stress = 0.3;    // Positive event
///
/// // With high arousal and positive bias, mood can suddenly jump
/// cusp_mood_step(&mut mood, arousal, stress, 0.1);
/// // mood might jump from -0.8 to +0.6 (mania)
/// ```
#[inline]
pub fn cusp_mood_step(mood: &mut f64, arousal: f64, external_bias: f64, dt: f64) {
    // Map arousal to control parameter (higher arousal = more bifurcation)
    let c = arousal.abs() * 1.5;
    // External bias directly maps to asymmetry
    let y = external_bias;

    cusp_step(mood, c, y, dt, 1.0);
    *mood = mood.clamp(-1.0, 1.0);
}

// ============================================================================
// COMBINED DYNAMICS: DynAffect Model
// ============================================================================

/// DynAffect model combining O-U mean-reversion with cusp bifurcation.
///
/// This is the full emotional dynamics model from the VIVA architecture:
/// - Base dynamics: Ornstein-Uhlenbeck (mean-reverting toward neutral)
/// - Bifurcation overlay: Cusp catastrophe (sudden mood transitions)
///
/// The model captures both:
/// - Gradual emotional drift/recovery (O-U)
/// - Sudden mood swings under high arousal (Cusp)
pub struct DynAffect {
    /// O-U parameters for each PAD dimension
    pub ou_params: [OUParams; 3],
    /// Whether cusp dynamics are enabled
    pub cusp_enabled: bool,
    /// Cusp sensitivity (how much arousal affects bifurcation)
    pub cusp_sensitivity: f64,
}

impl Default for DynAffect {
    fn default() -> Self {
        Self {
            ou_params: [OUParams::default(); 3],
            cusp_enabled: true,
            cusp_sensitivity: 1.5,
        }
    }
}

impl DynAffect {
    /// Create DynAffect with custom O-U parameters.
    pub fn with_ou_params(params: [OUParams; 3]) -> Self {
        Self {
            ou_params: params,
            ..Default::default()
        }
    }

    /// Step the full DynAffect model.
    ///
    /// # Arguments
    ///
    /// * `pad` - [pleasure, arousal, dominance] state
    /// * `dt` - Time step in seconds
    /// * `noise` - [n1, n2, n3] N(0,1) samples
    /// * `external_bias` - External emotional influence
    pub fn step(&self, pad: &mut [f64; 3], dt: f64, noise: &[f64; 3], external_bias: f64) {
        // Step 1: O-U dynamics for all dimensions
        ou_step_pad_bounded(pad, &self.ou_params, dt, noise);

        // Step 2: Cusp overlay on pleasure (mood) if enabled
        if self.cusp_enabled {
            let arousal = pad[1];
            let c = arousal.abs() * self.cusp_sensitivity;
            let y = external_bias;

            // Only apply cusp if in potential bifurcation region
            if c > 0.5 {
                cusp_step(&mut pad[0], c, y, dt * 0.5, 0.8);
                pad[0] = pad[0].clamp(-1.0, 1.0);
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ou_converges_to_equilibrium() {
        let params = OUParams {
            theta: 2.0, // Fast convergence
            mu: 0.5,    // Target
            sigma: 0.0, // No noise for deterministic test
        };
        let mut x = 0.0;

        // Run for 100 steps
        for _ in 0..100 {
            ou_step(&mut x, &params, 0.1, 0.0);
        }

        // Should converge near mu
        assert!(
            (x - params.mu).abs() < 0.01,
            "x={} should be near mu={}",
            x,
            params.mu
        );
    }

    #[test]
    fn test_ou_pad_stays_bounded() {
        let params = [OUParams::default(); 3];
        let mut pad = [0.5, 0.5, 0.5];

        // Run with extreme noise
        for i in 0..1000 {
            let noise = [(i as f64 % 3.0) - 1.0; 3]; // Varies -1 to 2
            ou_step_pad_bounded(&mut pad, &params, 0.1, &noise);
        }

        // All values should be in [-1, 1]
        for (i, &v) in pad.iter().enumerate() {
            assert!(v >= -1.0 && v <= 1.0, "pad[{}]={} out of bounds", i, v);
        }
    }

    #[test]
    fn test_cusp_single_equilibrium() {
        // c <= 0: always single equilibrium
        let eq = cusp_equilibria(-1.0, 0.5);
        assert_eq!(eq.len(), 1);
    }

    #[test]
    fn test_cusp_bifurcation_region() {
        // c > 0, small y: three equilibria
        let eq = cusp_equilibria(3.0, 0.1);
        assert_eq!(eq.len(), 3, "Expected 3 equilibria in bifurcation region");

        // Middle equilibrium should be unstable (between outer two)
        assert!(eq[0] < eq[1] && eq[1] < eq[2]);
    }

    #[test]
    fn test_cusp_gradient_at_equilibrium() {
        let eq = cusp_equilibria(2.0, 0.0);
        for x in eq {
            let grad = cusp_gradient(x, 2.0, 0.0);
            assert!(
                grad.abs() < 0.01,
                "Gradient at equilibrium should be ~0, got {}",
                grad
            );
        }
    }

    #[test]
    fn test_cusp_step_converges() {
        let mut x = 0.1;
        let c = 2.0;
        let y = 0.5;

        // Run until convergence
        for _ in 0..1000 {
            cusp_step(&mut x, c, y, 0.01, 1.0);
        }

        // Should be at an equilibrium (gradient ~= 0)
        let grad = cusp_gradient(x, c, y);
        assert!(
            grad.abs() < 0.1,
            "Should converge to equilibrium, grad={}",
            grad
        );
    }

    #[test]
    fn test_cusp_is_bifurcation() {
        // Bifurcation region: c > 0 and 4c³ > 27y²
        assert!(cusp_is_bifurcation(3.0, 0.1));
        assert!(!cusp_is_bifurcation(-1.0, 0.1));
        assert!(!cusp_is_bifurcation(1.0, 2.0)); // y too large
    }

    #[test]
    fn test_dynaffect_integration() {
        let model = DynAffect::default();
        let mut pad = [0.0, 0.8, 0.0]; // High arousal

        // Run several steps
        for _ in 0..100 {
            model.step(&mut pad, 0.016, &[0.0, 0.0, 0.0], 0.3);
        }

        // All values should remain bounded
        for (i, &v) in pad.iter().enumerate() {
            assert!(v >= -1.0 && v <= 1.0, "pad[{}]={} out of bounds", i, v);
        }
    }
}

// ============================================================================
// PROPERTY-BASED TESTS (proptest)
// ============================================================================

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: O-U with zero noise always converges to equilibrium
        #[test]
        fn ou_converges_deterministic(
            init in -10.0f64..10.0,
            mu in -5.0f64..5.0,
            theta in 0.1f64..5.0,
        ) {
            let params = OUParams { theta, mu, sigma: 0.0 };
            let mut x = init;

            // Run until convergence (deterministic case)
            for _ in 0..500 {
                ou_step(&mut x, &params, 0.1, 0.0);
            }

            // Should be within 1% of equilibrium
            prop_assert!(
                (x - mu).abs() < 0.05,
                "O-U should converge: x={}, mu={}, diff={}",
                x, mu, (x - mu).abs()
            );
        }

        /// Property: PAD values always stay bounded in [-1, 1]
        #[test]
        fn pad_always_bounded(
            p in -2.0f64..2.0,
            a in -2.0f64..2.0,
            d in -2.0f64..2.0,
            n1 in -3.0f64..3.0,
            n2 in -3.0f64..3.0,
            n3 in -3.0f64..3.0,
        ) {
            let params = [OUParams::default(); 3];
            let mut pad = [p.clamp(-1.0, 1.0), a.clamp(-1.0, 1.0), d.clamp(-1.0, 1.0)];
            let noise = [n1, n2, n3];

            // Multiple steps with potentially extreme noise
            for _ in 0..100 {
                ou_step_pad_bounded(&mut pad, &params, 0.1, &noise);
            }

            for (i, &v) in pad.iter().enumerate() {
                prop_assert!(
                    v >= -1.0 && v <= 1.0,
                    "PAD[{}] out of bounds: {}",
                    i, v
                );
            }
        }

        /// Property: Cusp gradient is zero at equilibria
        #[test]
        fn cusp_equilibria_have_zero_gradient(
            c in 0.1f64..5.0,
            y in -1.0f64..1.0,
        ) {
            // Skip if y is too large for bifurcation (single eq case)
            let eq = cusp_equilibria(c, y);

            for x in eq {
                let grad = cusp_gradient(x, c, y);
                prop_assert!(
                    grad.abs() < 0.1,
                    "Gradient at equilibrium should be ~0: x={}, grad={}",
                    x, grad
                );
            }
        }

        /// Property: DynAffect maintains PAD bounds under any conditions
        #[test]
        fn dynaffect_maintains_bounds(
            p in -1.0f64..1.0,
            a in -1.0f64..1.0,
            d in -1.0f64..1.0,
            bias in -2.0f64..2.0,
            n1 in -3.0f64..3.0,
            n2 in -3.0f64..3.0,
            n3 in -3.0f64..3.0,
        ) {
            let model = DynAffect::default();
            let mut pad = [p, a, d];
            let noise = [n1, n2, n3];

            // Run many steps
            for _ in 0..200 {
                model.step(&mut pad, 0.016, &noise, bias);
            }

            for (i, &v) in pad.iter().enumerate() {
                prop_assert!(
                    v >= -1.0 && v <= 1.0,
                    "DynAffect PAD[{}] out of bounds: {}",
                    i, v
                );
            }
        }

        /// Property: Cusp gradient is derivative of potential
        #[test]
        fn cusp_gradient_is_derivative(x in -2.0f64..2.0, c in -2.0f64..2.0, y in -2.0f64..2.0) {
            let eps = 1e-6;
            let v1 = cusp_potential(x - eps, c, y);
            let v2 = cusp_potential(x + eps, c, y);

            // Numerical derivative: dV/dx ≈ (V(x+ε) - V(x-ε)) / 2ε
            let numerical_grad = (v2 - v1) / (2.0 * eps);
            let analytical_grad = cusp_gradient(x, c, y);

            prop_assert!(
                (numerical_grad - analytical_grad).abs() < 0.001,
                "Gradient should match numerical derivative: analytical={}, numerical={}",
                analytical_grad, numerical_grad
            );
        }

        /// Property: Cusp bifurcation check is consistent with equilibria count
        #[test]
        fn cusp_bifurcation_consistent(c in -2.0f64..5.0, y in -2.0f64..2.0) {
            let is_bif = cusp_is_bifurcation(c, y);
            let eq = cusp_equilibria(c, y);

            if is_bif {
                // In bifurcation region, should have 3 equilibria
                prop_assert!(
                    eq.len() == 3,
                    "Bifurcation region should have 3 equilibria, got {}",
                    eq.len()
                );
            } else {
                // Outside bifurcation, should have 1 equilibrium
                prop_assert!(
                    eq.len() == 1,
                    "Outside bifurcation should have 1 equilibrium, got {}",
                    eq.len()
                );
            }
        }
    }
}
