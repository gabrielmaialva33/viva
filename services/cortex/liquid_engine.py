"""
VIVA Cortex - Liquid Neural Engine
==================================
Implements Biological Time Perception and Emotional Dynamics using
Liquid Time-Constant (LTC) Networks and Neural Circuit Policies (NCPs).

Now with Neural ODE mode for explicit continuous-time dynamics:
    dPAD/dt = f_θ(PAD, sensory, t)

Reference:
- "Liquid Time-constant Networks" (Nature Machine Intelligence, 2021)
- "Neural Circuit Policies Enabling Auditable Autonomy" (Nature Machine Intelligence, 2020)
- Neural ODEs: Chen et al. 2018 (NeurIPS)

Concept:
Input (Sensory) -> [Liquid Interneurons] -> [Command Neurons] -> Output (Motor/Emotion)

This engine models VIVA's "Soul Physics" as a system of differential equations.
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from ncps.wirings import AutoNCP
from ncps.torch import LTC

# Neural ODE support (optional)
try:
    from neural_ode import NeuralODECortex, NeuralODEConfig, is_available as neural_ode_available
    NEURAL_ODE_AVAILABLE = neural_ode_available()
except ImportError:
    NEURAL_ODE_AVAILABLE = False

logger = logging.getLogger(__name__)

class LiquidCortex(nn.Module):
    def __init__(self, input_dim=64, hidden_units=32, output_dim=3):
        super().__init__()

        # NCP Wiring (inspired by C. Elegans)
        # Sparse connectivity for interpretability and biological realism
        self.wiring = AutoNCP(hidden_units, output_dim)

        # Liquid Time-Constant Layer
        # Solves dy/dt = -y/tau + f(x)
        self.ltc = LTC(input_dim, self.wiring, batch_first=True)

        # State management
        self.hidden_state = None
        self._input_dim = input_dim

        logger.info(f"Liquid Cortex initialized. Wiring: {self.wiring.synapse_count} synapses.")

    def forward(self, x, time_delta=1.0):
        """
        Forward pass through the liquid network.

        x: Input tensor [batch, seq_len, input_dim]
        time_delta: Time elapsed since last tick (simulates biological time perception)
                    Note: LTC handles time implicitly via ODE solver tau constants
        """
        # LTC handles time-variant processing implicitly via ODE solver
        # time_delta could modulate tau in future versions
        _ = time_delta  # Reserved for future time-aware processing
        output, self.hidden_state = self.ltc(x, self.hidden_state)
        return output

    def reset_state(self):
        self.hidden_state = None

    def tick(self, sensory_input: np.ndarray):
        """
        Process a single discrete tick as a continuous flow.
        """
        with torch.no_grad():
            x = torch.tensor(sensory_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Predict
            output = self.forward(x)

            return output.squeeze().numpy()

class CortexEngine:
    """
    Wrapper for the Liquid Model with Neural ODE support.

    Modes:
    - LTC mode (default): Uses Liquid Time-Constant networks
    - Neural ODE mode: Uses explicit dPAD/dt = f(PAD, sensory, t)

    Set CORTEX_USE_NEURAL_ODE=true to enable Neural ODE mode.
    """

    def __init__(self, use_neural_ode: bool = None):
        # Check environment for Neural ODE mode
        if use_neural_ode is None:
            use_neural_ode = os.getenv("CORTEX_USE_NEURAL_ODE", "false").lower() == "true"

        self.use_neural_ode = use_neural_ode and NEURAL_ODE_AVAILABLE
        self.heatmap_cache = None

        if self.use_neural_ode:
            # Neural ODE mode: dPAD/dt = f(PAD, sensory, t)
            config = NeuralODEConfig(
                pad_dim=3,
                sensory_dim=61,  # energy(1) + context(60)
                hidden_dim=64
            )
            self.neural_ode = NeuralODECortex(
                pad_dim=config.pad_dim,
                sensory_dim=config.sensory_dim,
                hidden_dim=config.hidden_dim
            )
            self.neural_ode.eval()
            self.model = None
            logger.info("CortexEngine initialized in Neural ODE mode")
        else:
            # LTC mode (default)
            # Input: PAD(3) + Energy(1) + Context_Embedding(60) = 64
            self.model = LiquidCortex(input_dim=64, hidden_units=48, output_dim=3)
            self.neural_ode = None
            logger.info("CortexEngine initialized in LTC mode")

    def process_stimulus(self, pad, energy, context_vec, time_delta: float = 1.0):
        """
        Process incoming stimulus through the liquid brain.

        Args:
            pad: [P, A, D] - Current emotional state
            energy: float - Body energy level
            context_vec: vector[60] (from Ultra or Mamba)
            time_delta: Time elapsed since last tick (for Neural ODE mode)

        Returns:
            new_pad: [P, A, D] - Predicted next emotional state
        """
        if self.use_neural_ode:
            return self._process_neural_ode(pad, energy, context_vec, time_delta)
        else:
            return self._process_ltc(pad, energy, context_vec)

    def _process_ltc(self, pad, energy, context_vec):
        """Process using LTC (original mode)."""
        # Construct input vector
        inputs = np.concatenate([pad, [energy], context_vec])

        # Pad context if needed to match 64
        if len(inputs) < 64:
            inputs = np.pad(inputs, (0, 64 - len(inputs)))
        elif len(inputs) > 64:
            inputs = inputs[:64]

        new_pad = self.model.tick(inputs)
        return new_pad

    def _process_neural_ode(self, pad, energy, context_vec, time_delta: float):
        """Process using Neural ODE mode."""
        # Convert to tensors
        pad_tensor = torch.tensor(pad, dtype=torch.float32)

        # Build sensory: energy + context
        sensory = np.concatenate([[energy], context_vec])
        if len(sensory) < 61:
            sensory = np.pad(sensory, (0, 61 - len(sensory)))
        elif len(sensory) > 61:
            sensory = sensory[:61]

        sensory_tensor = torch.tensor(sensory, dtype=torch.float32)

        # Integrate ODE
        with torch.no_grad():
            new_pad_tensor = self.neural_ode(
                pad_tensor,
                sensory_tensor,
                time_delta=time_delta,
                log_trajectory=False
            )

        return new_pad_tensor.squeeze().numpy()

    def process_with_trajectory(self, pad, energy, context_vec, time_delta: float = 1.0):
        """
        Process and return full trajectory (Neural ODE mode only).

        Returns:
            new_pad: Final PAD state
            trajectory: Dict with 't', 'pad', 'derivatives' if Neural ODE mode
        """
        if not self.use_neural_ode:
            new_pad = self._process_ltc(pad, energy, context_vec)
            return new_pad, None

        # Convert to tensors
        pad_tensor = torch.tensor(pad, dtype=torch.float32)

        sensory = np.concatenate([[energy], context_vec])
        if len(sensory) < 61:
            sensory = np.pad(sensory, (0, 61 - len(sensory)))
        sensory_tensor = torch.tensor(sensory, dtype=torch.float32)

        # Integrate with logging
        with torch.no_grad():
            new_pad_tensor = self.neural_ode(
                pad_tensor,
                sensory_tensor,
                time_delta=time_delta,
                log_trajectory=True,
                num_steps=10
            )

        trajectory = self.neural_ode.get_last_trajectory()
        return new_pad_tensor.squeeze().numpy(), trajectory

    def get_synaptic_heatmap(self):
        """Return sensory-motor connection weights for visualization."""
        if self.use_neural_ode:
            # For Neural ODE, return dynamics parameters
            if self.neural_ode is not None:
                return {
                    "mode": "neural_ode",
                    "scale": self.neural_ode.dynamics.scale.item()
                }
        else:
            # NCP allows reading synaptic weights directly
            # TODO: self.model.wiring.synapse_weights (pseudo-code)
            return {"mode": "ltc"}
        return {}

    def get_mode(self) -> str:
        """Return current processing mode."""
        return "neural_ode" if self.use_neural_ode else "ltc"

    def reset_state(self):
        """Reset internal state."""
        if self.model is not None:
            self.model.reset_state()
        if self.neural_ode is not None:
            self.neural_ode.clear_log()

def get_engine():
    global _engine
    if '_engine' not in globals():
        _engine = CortexEngine()
    return _engine
