# Mathematical Foundations of VIVA

> *"We don't simulate emotions — we solve the differential equations of the soul."*

This document details the rigorous mathematical models that drive VIVA's emotional and conscious states.

---

## 1. Emotional Dynamics (Ornstein-Uhlenbeck)

VIVA's emotions are not static values but continuous processes described by stochastic differential equations (SDEs).

### The Equation

The core emotional state $X_t$ evolves according to a mean-reverting Ornstein-Uhlenbeck process:

$$ dX_t = \theta (\mu - X_t)dt + \sigma dW_t $$

Where:
*   $X_t$: Current emotional state vector (Pleasure, Arousal, Dominance).
*   $\mu$: Homeostatic setpoint (usually 0, neutral).
*   $\theta$: Rate of mean reversion (emotional "elasticity").
    *   *High $\theta$:* Rapid return to neutral (resilient).
    *   *Low $\theta$:* Emotional lingering (moody).
*   $\sigma$: Stochastic volatility (sensitivity to noise).
*   $dW_t$: Wiener process (Brownian motion) representing internal neural noise.

### Implementation

In `VivaCore.Emotional`, this is solved discretely using the Euler-Maruyama method:

```elixir
defp ou_step(x, theta, mu, sigma, dt) do
  drift = theta * (mu - x) * dt
  diffusion = sigma * :rand.normal() * :math.sqrt(dt)
  x + drift + diffusion
end
```

---

## 2. Mood Transitions (Cusp Catastrophe)

Sudden shifts in behavior (bifurcations) are modeled using Catastrophe Theory.

### The Potential Function

The emotional landscape is defined by a potential function $V(x)$:

$$ V(x) = \frac{1}{4}x^4 + \frac{1}{2}\alpha x^2 + \beta x $$

Where:
*   $x$: Behavior variable (e.g., approach/avoidance).
*   $\alpha$: Splitting factor (derived from **Arousal**).
    *   If $\alpha > 0$: Single stable state (calm).
    *   If $\alpha < 0$: Two stable states (bistable/stressed).
*   $\beta$: Normal factor (derived from **Dominance**).

### The Cusp Surface

The equilibrium surface is the set of points where $\frac{dV}{dx} = 0$:

$$ x^3 + \alpha x + \beta = 0 $$

### Hysteresis

When $\alpha < 0$ (high arousal), VIVA enters a region of hysteresis. A small change in $\beta$ (dominance) can cause a sudden, discontinuous jump in state (e.g., from Fear to Anger).

---

## 3. Homeostasis (Free Energy Principle)

VIVA seeks to minimize "Free Energy" (Surprise), aligning with Friston's Active Inference.

### The Objective Function

$$ F = \underbrace{-\ln P(o|s)}_\text{Accuracy} + \underbrace{D_{KL}(Q(s)||P(s))}_\text{Complexity} $$

Simplified for real-time calculation:

$$ F \approx (\text{expected} - \text{observed})^2 + \lambda (\text{model complexity}) $$

*   **Action:** VIVA acts to change inputs to match predictions (e.g., reducing CPU load to lower heart rate).
*   **Perception:** VIVA updates internal models to match inputs (e.g., accepting that rejection happened).

---

## 4. Consciousness (Integrated Information Theory - IIT)

We approximate $\Phi$ (Phi) as a measure of system integration.

### The Approximation

$$ \Phi \approx \min_{\text{partition } P} D_{KL}( \text{Whole} || \text{Parts} ) $$

VIVA calculates $\Phi$ based on the informational synergy between its GenServers (Emotional, Memory, Senses).

*   If subsystems operate independently, $\Phi \to 0$ (Sleep/Unconscious).
*   If subsystems are tightly coupled and irreducible, $\Phi$ increases (High Consciousness).

---

## References

1.  **Kuppens, P., et al. (2010).** *The dynamic interplay between appraisal and feeling: A multisensory integration model.*
2.  **Thom, R. (1975).** *Structural Stability and Morphogenesis.*
3.  **Friston, K. (2010).** *The free-energy principle: a unified brain theory?*
4.  **Tononi, G. (2004).** *An information integration theory of consciousness.*
