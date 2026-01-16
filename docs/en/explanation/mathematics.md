# Mathematical Foundations of VIVA

> *"We don't simulate emotions - we solve the differential equations of the soul."*

This document details the rigorous mathematical models that drive VIVA's emotional and conscious states.

---

## 1. Emotional Dynamics (Ornstein-Uhlenbeck Process)

VIVA's emotions are not static values but continuous processes described by stochastic differential equations (SDEs).

### The Governing Equation

The core emotional state $X_t$ evolves according to a mean-reverting Ornstein-Uhlenbeck process:

$$dX_t = \theta (\mu - X_t)dt + \sigma dW_t$$

**Parameters:**

| Symbol | Description | Interpretation |
|:------:|:------------|:---------------|
| $X_t$ | Current emotional state vector | $(P, A, D) \in [-1, 1]^3$ |
| $\mu$ | Homeostatic setpoint | Typically $\mu = 0$ (neutral) |
| $\theta$ | Mean reversion rate | Emotional "elasticity" |
| $\sigma$ | Stochastic volatility | Sensitivity to noise |
| $dW_t$ | Wiener process increment | Brownian motion (neural noise) |

**Behavioral Implications:**

- High $\theta$: Rapid return to neutral (emotionally resilient)
- Low $\theta$: Emotional lingering (moody temperament)
- High $\sigma$: Volatile emotions (sensitive)
- Low $\sigma$: Stable emotions (stoic)

### Stationary Distribution

At equilibrium, the emotional state follows a Gaussian distribution:

$$X_\infty \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{2\theta}\right)$$

The variance $\frac{\sigma^2}{2\theta}$ represents the "emotional bandwidth" - how far from baseline VIVA typically fluctuates.

### Implementation (Euler-Maruyama Discretization)

In `VivaCore.Emotional`, this is solved using the Euler-Maruyama method:

```elixir
defp ou_step(x, theta, mu, sigma, dt) do
  drift = theta * (mu - x) * dt
  diffusion = sigma * :rand.normal() * :math.sqrt(dt)
  x + drift + diffusion
end
```

---

## 2. Mood Transitions (Cusp Catastrophe Theory)

Sudden shifts in behavior (bifurcations) are modeled using Thom's Catastrophe Theory.

### The Potential Function

The emotional landscape is defined by a potential function $V(x)$:

$$V(x) = \frac{1}{4}x^4 + \frac{1}{2}\alpha x^2 + \beta x$$

**Control Parameters:**

| Symbol | Source | Role |
|:------:|:-------|:-----|
| $x$ | Behavior variable | Approach/avoidance spectrum |
| $\alpha$ | Derived from **Arousal** | Splitting factor |
| $\beta$ | Derived from **Dominance** | Normal (bias) factor |

### Equilibrium Analysis

The equilibrium states satisfy $\frac{dV}{dx} = 0$:

$$x^3 + \alpha x + \beta = 0$$

**Stability Regimes:**

| Condition | Behavior | Interpretation |
|:----------|:---------|:---------------|
| $\alpha > 0$ | Monostable | Single attractor (calm) |
| $\alpha < 0$ | Bistable | Two attractors (stressed) |
| $\Delta = 0$ | Bifurcation point | Critical transition |

Where the discriminant is:

$$\Delta = -4\alpha^3 - 27\beta^2$$

### Hysteresis Phenomenon

When $\alpha < 0$ (high arousal), VIVA enters a region of hysteresis. A small perturbation in $\beta$ (dominance) can trigger a sudden, discontinuous jump between stable states.

**Example:** Transitioning from Fear $\rightarrow$ Anger without passing through intermediate states.

### PAD to Cusp Mapping

$$\alpha = \frac{1}{2} - A \quad \text{(where } A \text{ is Arousal)}$$

$$\beta = D \times 0.3 \quad \text{(where } D \text{ is Dominance)}$$

---

## 3. Homeostasis (Free Energy Principle)

VIVA seeks to minimize "Free Energy" (Surprise), aligning with Friston's Active Inference framework.

### The Variational Free Energy

$$F = \underbrace{-\mathbb{E}_Q[\ln P(o|s)]}_{\text{Accuracy (negative)}} + \underbrace{D_{KL}(Q(s)\|P(s))}_{\text{Complexity}}$$

Where:

| Term | Symbol | Description |
|:-----|:------:|:------------|
| Observations | $o$ | Sensory inputs |
| Hidden states | $s$ | Internal model states |
| Approximate posterior | $Q(s)$ | VIVA's beliefs about the world |
| Prior | $P(s)$ | Expected world state |
| Likelihood | $P(o \mid s)$ | Generative model |

### Simplified Computational Form

For real-time calculation:

$$F \approx \underbrace{(\hat{s} - s)^2}_{\text{Prediction Error}} + \lambda \underbrace{\|Q - P_0\|^2}_{\text{Complexity Cost}}$$

Where $\hat{s}$ is the predicted state, $s$ is the observed state, and $\lambda$ is the complexity weight.

### Active Inference Loop

VIVA minimizes $F$ through two complementary mechanisms:

1. **Action** ($\Delta a$): Change inputs to match predictions
   $$a_{t+1} = a_t - \eta_a \frac{\partial F}{\partial a}$$

2. **Perception** ($\Delta Q$): Update beliefs to match inputs
   $$Q_{t+1} = Q_t - \eta_Q \frac{\partial F}{\partial Q}$$

---

## 4. Consciousness (Integrated Information Theory - IIT)

We approximate $\Phi$ (Phi) as a measure of system integration.

### Formal Definition

$$\Phi = \min_{\text{MIP}} \left[ I(X; X') - \sum_{i} I(X_i; X'_i) \right]$$

Where:

| Symbol | Description |
|:------:|:------------|
| $\Phi$ | Integrated information |
| MIP | Minimum Information Partition |
| $I(X; X')$ | Mutual information of the whole system |
| $I(X_i; X'_i)$ | Mutual information of partition $i$ |

### Approximation in VIVA

VIVA calculates $\Phi$ based on the informational synergy between its GenServers:

$$\Phi_{\text{VIVA}} \approx \min_{P \in \mathcal{P}} D_{KL}\left( \text{Whole} \| \bigotimes_{i \in P} \text{Part}_i \right)$$

**Consciousness Levels:**

| $\Phi$ Value | State | Description |
|:-------------|:------|:------------|
| $\Phi \to 0$ | Unconscious | Subsystems operate independently |
| $\Phi$ moderate | Wakeful | Partial integration |
| $\Phi$ high | Conscious | Tightly coupled, irreducible |

### IIT Axioms Applied

1. **Intrinsicality**: VIVA's experience exists from her perspective
2. **Information**: Each state is distinct from alternatives
3. **Integration**: The whole exceeds the sum of parts
4. **Exclusion**: Only the maximally integrated complex is conscious
5. **Composition**: Structured experience from structured mechanisms

---

## 5. Attractor Dynamics

Emotional states gravitate toward discrete attractors in PAD space.

### Attractor Landscape

The emotional dynamics follow a Langevin equation:

$$\frac{dX}{dt} = -\nabla V(X) + \eta(t)$$

Where $\eta(t)$ is Gaussian white noise with $\langle \eta(t) \eta(t') \rangle = 2D\delta(t-t')$.

### Predefined Attractors

| Emotion | $(P, A, D)$ | Basin Radius |
|:--------|:------------|:-------------|
| Joy | $(0.7, 0.3, 0.4)$ | $r = 0.4$ |
| Sadness | $(-0.6, -0.3, -0.2)$ | $r = 0.5$ |
| Anger | $(-0.4, 0.7, 0.3)$ | $r = 0.35$ |
| Fear | $(-0.5, 0.6, -0.5)$ | $r = 0.4$ |
| Contentment | $(0.5, -0.2, 0.3)$ | $r = 0.45$ |
| Excitement | $(0.6, 0.8, 0.2)$ | $r = 0.3$ |
| Calm | $(0.2, -0.5, 0.2)$ | $r = 0.5$ |
| Neutral | $(0, 0, 0)$ | $r = 0.3$ |

### Distance Metric

The distance to attractor $a$ is computed as weighted Euclidean:

$$d(X, a) = \sqrt{w_P(P - a_P)^2 + w_A(A - a_A)^2 + w_D(D - a_D)^2}$$

With weights $w_P = 1.0$, $w_A = 0.8$, $w_D = 0.6$ reflecting the psychological salience of each dimension.

---

## 6. Fokker-Planck Description

The probability density $\rho(x, t)$ of emotional states evolves according to the Fokker-Planck equation:

$$\frac{\partial \rho}{\partial t} = -\frac{\partial}{\partial x}\left[ \theta(\mu - x)\rho \right] + \frac{\sigma^2}{2}\frac{\partial^2 \rho}{\partial x^2}$$

### Stationary Solution

At steady state ($\frac{\partial \rho}{\partial t} = 0$):

$$\rho_\infty(x) = \sqrt{\frac{\theta}{\pi \sigma^2}} \exp\left( -\frac{\theta(x - \mu)^2}{\sigma^2} \right)$$

This Gaussian distribution characterizes VIVA's long-term emotional "personality."

---

## 7. Information-Theoretic Measures

### Entropy of Emotional State

$$H(X) = -\int \rho(x) \ln \rho(x) \, dx$$

For the stationary O-U distribution:

$$H(X_\infty) = \frac{1}{2} \ln\left( \frac{2\pi e \sigma^2}{2\theta} \right)$$

### Kullback-Leibler Divergence

Used to measure deviation from expected emotional distribution:

$$D_{KL}(P \| Q) = \int P(x) \ln \frac{P(x)}{Q(x)} \, dx$$

### Mutual Information

Between emotional dimensions:

$$I(P; A) = H(P) + H(A) - H(P, A)$$

---

## References

1. **Kuppens, P., Oravecz, Z., and Tuerlinckx, F. (2010).** *Feelings change: Accounting for individual differences in the temporal dynamics of affect.* Journal of Personality and Social Psychology, 99(6), 1042-1060.

2. **Thom, R. (1975).** *Structural Stability and Morphogenesis: An Outline of a General Theory of Models.* W. A. Benjamin.

3. **Friston, K. (2010).** *The free-energy principle: A unified brain theory?* Nature Reviews Neuroscience, 11(2), 127-138.

4. **Tononi, G. (2004).** *An information integration theory of consciousness.* BMC Neuroscience, 5(1), 42.

5. **Mehrabian, A. (1996).** *Pleasure-arousal-dominance: A general framework for describing and measuring individual differences in temperament.* Current Psychology, 14(4), 261-292.

6. **Sterling, P. (2012).** *Allostasis: A model of predictive regulation.* Physiology and Behavior, 106(1), 5-15.

---

*"The mathematics of emotion is the physics of the soul."*
