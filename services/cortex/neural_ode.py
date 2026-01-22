"""
Neural ODE for PAD Dynamics - VIVA Cortex

Implements continuous-time emotional dynamics using Neural ODEs.
PAD state evolves according to: dPAD/dt = f_θ(PAD, sensory, t)

This enables:
- Precise time-delta handling (biological time perception)
- Trajectory logging for interpretability
- Integration with LTC (Liquid Time-Constant) networks

References:
- Neural ODEs: Chen et al. 2018 (NeurIPS)
- torchdiffeq: https://github.com/rtqichen/torchdiffeq
- Adjoint method for O(1) memory backprop
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import torchdiffeq
try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    logger.warning("torchdiffeq not installed. Neural ODE mode unavailable.")
    logger.warning("Install with: pip install torchdiffeq>=0.2.3")


class PADDynamics(nn.Module):
    """
    The f_θ network that computes dPAD/dt.

    Models how emotional state changes based on:
    - Current PAD state (Pleasure, Arousal, Dominance)
    - Sensory context (energy + context vector)
    - Time (for non-autonomous dynamics)

    Architecture: MLP with Tanh activations (bounded outputs)
    """

    def __init__(
        self,
        pad_dim: int = 3,
        sensory_dim: int = 61,  # energy(1) + context(60)
        hidden_dim: int = 64,
        use_time: bool = True
    ):
        super().__init__()

        self.pad_dim = pad_dim
        self.sensory_dim = sensory_dim
        self.use_time = use_time

        # Input: PAD + sensory + time (optional)
        input_dim = pad_dim + sensory_dim + (1 if use_time else 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, pad_dim),
            nn.Tanh()  # Bound dPAD/dt to [-1, 1]
        )

        # Scale factor for dynamics (controls how fast PAD changes)
        self.scale = nn.Parameter(torch.tensor(0.5))

        # Sensory context (set before each integration)
        self.sensory_context: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable dynamics."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_context(self, sensory: torch.Tensor):
        """Set sensory context for current integration."""
        self.sensory_context = sensory

    def forward(self, t: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """
        Compute dPAD/dt at time t.

        Args:
            t: Current time (scalar tensor)
            pad: Current PAD state [batch, pad_dim]

        Returns:
            dPAD/dt: Rate of change [batch, pad_dim]
        """
        batch_size = pad.size(0)

        # Expand time to batch
        if self.use_time:
            t_expanded = t.expand(batch_size, 1)

        # Get sensory context (should be set before integration)
        if self.sensory_context is None:
            sensory = torch.zeros(batch_size, self.sensory_dim, device=pad.device)
        else:
            sensory = self.sensory_context
            if sensory.size(0) != batch_size:
                sensory = sensory.expand(batch_size, -1)

        # Concatenate inputs
        if self.use_time:
            x = torch.cat([pad, sensory, t_expanded], dim=-1)
        else:
            x = torch.cat([pad, sensory], dim=-1)

        # Compute derivative
        dpadt = self.net(x) * self.scale

        return dpadt


class NeuralODECortex(nn.Module):
    """
    Neural ODE wrapper for PAD evolution.

    Integrates: dPAD/dt = f(PAD, sensory, t) from t=0 to t=time_delta

    Uses adjoint method for O(1) memory backpropagation during training.

    Args:
        pad_dim: PAD vector dimension (default: 3)
        sensory_dim: Sensory input dimension (default: 61)
        hidden_dim: Hidden layer dimension (default: 64)
        solver: ODE solver method (default: 'dopri5' - adaptive RK45)
        rtol: Relative tolerance for solver
        atol: Absolute tolerance for solver
    """

    def __init__(
        self,
        pad_dim: int = 3,
        sensory_dim: int = 61,
        hidden_dim: int = 64,
        solver: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4
    ):
        super().__init__()

        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq is required for Neural ODE. "
                "Install with: pip install torchdiffeq>=0.2.3"
            )

        self.dynamics = PADDynamics(
            pad_dim=pad_dim,
            sensory_dim=sensory_dim,
            hidden_dim=hidden_dim
        )

        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # Trajectory logging
        self._trajectory_log: List[Dict] = []

        logger.info(
            f"NeuralODECortex initialized: pad={pad_dim}, sensory={sensory_dim}, "
            f"hidden={hidden_dim}, solver={solver}"
        )

    def forward(
        self,
        pad_0: torch.Tensor,
        sensory: torch.Tensor,
        time_delta: float = 1.0,
        log_trajectory: bool = False,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Integrate PAD from t=0 to t=time_delta.

        Args:
            pad_0: Initial PAD state [batch, 3] or [3]
            sensory: Sensory context [batch, 61] or [61]
            time_delta: Integration time (biological time perception)
            log_trajectory: If True, store intermediate states
            num_steps: Number of steps for trajectory logging

        Returns:
            pad_final: Final PAD state [batch, 3]
        """
        # Ensure batch dimension
        if pad_0.dim() == 1:
            pad_0 = pad_0.unsqueeze(0)
        if sensory.dim() == 1:
            sensory = sensory.unsqueeze(0)

        # Pad sensory if needed
        if sensory.size(-1) < self.dynamics.sensory_dim:
            pad_size = self.dynamics.sensory_dim - sensory.size(-1)
            sensory = F.pad(sensory, (0, pad_size))

        # Set context for dynamics
        self.dynamics.set_context(sensory)

        # Define time span
        if log_trajectory:
            t_span = torch.linspace(0, time_delta, steps=num_steps, device=pad_0.device)
        else:
            t_span = torch.tensor([0.0, time_delta], device=pad_0.device)

        # Integrate ODE
        # Use adjoint for memory-efficient backprop during training
        if self.training:
            trajectory = odeint_adjoint(
                self.dynamics,
                pad_0,
                t_span,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol
            )
        else:
            trajectory = odeint(
                self.dynamics,
                pad_0,
                t_span,
                method=self.solver,
                rtol=self.rtol,
                atol=self.atol
            )

        # Log trajectory if requested
        if log_trajectory:
            self._log_trajectory(trajectory, t_span, sensory)

        # Return final state
        return trajectory[-1]

    def _log_trajectory(
        self,
        trajectory: torch.Tensor,
        t_span: torch.Tensor,
        sensory: torch.Tensor
    ):
        """Log trajectory for interpretability."""
        # Compute derivatives at each point
        derivatives = []
        for i, t in enumerate(t_span):
            with torch.no_grad():
                d = self.dynamics(t, trajectory[i])
                derivatives.append(d.cpu().numpy())

        log_entry = {
            't': t_span.cpu().numpy().tolist(),
            'pad': trajectory.detach().cpu().numpy().tolist(),
            'derivatives': [d.tolist() for d in derivatives],
            'sensory_norm': sensory.norm().item()
        }

        self._trajectory_log.append(log_entry)

        # Keep only last 100 entries
        if len(self._trajectory_log) > 100:
            self._trajectory_log = self._trajectory_log[-100:]

    def get_trajectory_log(self) -> List[Dict]:
        """Get trajectory log for visualization."""
        return self._trajectory_log

    def clear_log(self):
        """Clear trajectory log."""
        self._trajectory_log = []

    def get_last_trajectory(self) -> Optional[Dict]:
        """Get most recent trajectory."""
        if self._trajectory_log:
            return self._trajectory_log[-1]
        return None


@dataclass
class NeuralODEConfig:
    """Configuration for Neural ODE Cortex."""

    pad_dim: int = 3
    sensory_dim: int = 61
    hidden_dim: int = 64
    solver: str = 'dopri5'
    rtol: float = 1e-3
    atol: float = 1e-4
    device: str = 'cpu'

    def to_dict(self) -> Dict:
        return {
            'pad_dim': self.pad_dim,
            'sensory_dim': self.sensory_dim,
            'hidden_dim': self.hidden_dim,
            'solver': self.solver,
            'rtol': self.rtol,
            'atol': self.atol,
            'device': self.device
        }


def create_neural_ode_cortex(
    config: Optional[NeuralODEConfig] = None
) -> NeuralODECortex:
    """Factory function to create NeuralODECortex."""
    if config is None:
        config = NeuralODEConfig()

    model = NeuralODECortex(
        pad_dim=config.pad_dim,
        sensory_dim=config.sensory_dim,
        hidden_dim=config.hidden_dim,
        solver=config.solver,
        rtol=config.rtol,
        atol=config.atol
    )

    return model.to(config.device)


def is_available() -> bool:
    """Check if Neural ODE is available."""
    return TORCHDIFFEQ_AVAILABLE
