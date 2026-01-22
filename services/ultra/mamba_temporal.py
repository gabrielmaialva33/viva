"""
Mamba-2 Temporal Memory Processor - VIVA Ultra

Implements O(n) linear-time sequence processing for memory history using
State Space Models (SSM). Alternative to Transformer attention (O(n²)).

Key Benefits:
- Linear complexity: Can process 100+ memories without VRAM explosion
- Implicit memory: Hidden state captures temporal patterns
- Efficient inference: Single pass, no KV cache

Architecture:
Memory embeddings [t-100:t] → Mamba-2 → context[60] → Cortex

Reference:
- Mamba-2: Gu & Dao, 2024. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- https://github.com/state-spaces/mamba
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Check for mamba-ssm availability
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("mamba-ssm not installed. Mamba temporal processor unavailable.")
    logger.warning("Install with: pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0")


@dataclass
class MambaConfig:
    """Configuration for Mamba Temporal Processor."""
    d_model: int = 384          # Input dimension (MiniLM embedding)
    d_state: int = 64           # SSM state dimension
    n_layers: int = 2           # Number of Mamba layers
    output_dim: int = 60        # Context vector dimension (for Cortex)
    max_seq_len: int = 128      # Maximum sequence length
    dropout: float = 0.1        # Dropout rate
    device: str = 'cpu'


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MambaTemporalProcessor(nn.Module):
    """
    Mamba-2 based temporal processor for memory sequences.

    Processes a sequence of memory embeddings and outputs a compact
    context vector that captures temporal patterns and semantic content.

    Input: [batch, seq_len, 384] (memory embeddings from MiniLM)
    Output: [batch, 60] (context vector for Cortex)
    """

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 64,
        n_layers: int = 2,
        output_dim: int = 60,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for MambaTemporalProcessor. "
                "Install with: pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.output_dim = output_dim

        # Positional encoding (temporal awareness)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Mamba layers with layer normalization
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(n_layers):
            self.mamba_layers.append(
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,       # Causal convolution width
                    expand=2,       # Expansion factor for inner dimension
                )
            )
            self.layer_norms.append(nn.LayerNorm(d_model))

        # Final projection to context vector
        self.output_norm = nn.LayerNorm(d_model)
        self.context_proj = nn.Linear(d_model, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"MambaTemporalProcessor initialized: d_model={d_model}, "
            f"d_state={d_state}, layers={n_layers}, output={output_dim}"
        )

    def forward(
        self,
        memory_embeddings: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Process memory sequence through Mamba layers.

        Args:
            memory_embeddings: [batch, seq_len, d_model] or [seq_len, d_model]
            return_sequence: If True, return full sequence instead of last state

        Returns:
            If return_sequence=False: [batch, output_dim] context vector
            If return_sequence=True: [batch, seq_len, output_dim] full sequence
        """
        # Ensure batch dimension
        if memory_embeddings.dim() == 2:
            memory_embeddings = memory_embeddings.unsqueeze(0)

        # Add positional encoding
        x = self.pos_encoding(memory_embeddings)

        # Process through Mamba layers (residual connections)
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            # Pre-norm residual connection
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x)
            x = x + residual

        # Final normalization
        x = self.output_norm(x)

        if return_sequence:
            # Return full sequence projected to output_dim
            return self.context_proj(x)
        else:
            # Return only final state (captures full temporal context)
            final_state = x[:, -1, :]  # [batch, d_model]
            return self.context_proj(final_state)  # [batch, output_dim]

    def process_memory_sequence(
        self,
        embeddings: List[List[float]],
        timestamps: Optional[List[float]] = None
    ) -> Tuple[List[float], dict]:
        """
        Process a sequence of memory embeddings.

        Convenience method for inference.

        Args:
            embeddings: List of embedding vectors [[e1], [e2], ...]
            timestamps: Optional list of timestamps for temporal weighting

        Returns:
            (context_vector, metadata)
        """
        with torch.no_grad():
            # Convert to tensor
            x = torch.tensor(embeddings, dtype=torch.float32)

            # Add batch dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(0)

            # Process
            context = self.forward(x)

            # Convert back to list
            context_list = context.squeeze().tolist()

            metadata = {
                "seq_len": len(embeddings),
                "d_model": self.d_model,
                "output_dim": self.output_dim,
                "has_timestamps": timestamps is not None
            }

            return context_list, metadata


class MambaFallback(nn.Module):
    """
    Fallback processor when mamba-ssm is not available.

    Uses simple averaging with decay (exponentially weighted mean).
    Not as good as Mamba but still captures temporal patterns.
    """

    def __init__(
        self,
        d_model: int = 384,
        output_dim: int = 60,
        decay: float = 0.9
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.decay = decay

        # Simple projection
        self.proj = nn.Linear(d_model, output_dim)

        logger.warning("Using MambaFallback (mamba-ssm not installed)")

    def forward(self, memory_embeddings: torch.Tensor) -> torch.Tensor:
        """Exponentially weighted mean fallback."""
        if memory_embeddings.dim() == 2:
            memory_embeddings = memory_embeddings.unsqueeze(0)

        batch, seq_len, d_model = memory_embeddings.shape

        # Exponentially weighted mean (more recent = higher weight)
        weights = torch.tensor([
            self.decay ** (seq_len - 1 - i)
            for i in range(seq_len)
        ], device=memory_embeddings.device, dtype=memory_embeddings.dtype)

        weights = weights / weights.sum()  # Normalize
        weights = weights.view(1, seq_len, 1)  # [1, seq_len, 1]

        # Weighted mean
        weighted_mean = (memory_embeddings * weights).sum(dim=1)  # [batch, d_model]

        return self.proj(weighted_mean)  # [batch, output_dim]

    def process_memory_sequence(
        self,
        embeddings: List[List[float]],
        timestamps: Optional[List[float]] = None
    ) -> Tuple[List[float], dict]:
        """Process using fallback method."""
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            context = self.forward(x)
            context_list = context.squeeze().tolist()

            return context_list, {
                "seq_len": len(embeddings),
                "method": "exponential_weighted_mean_fallback"
            }


# Singleton instance
_mamba_processor: Optional[nn.Module] = None


def create_mamba_processor(config: Optional[MambaConfig] = None) -> nn.Module:
    """
    Factory function to create Mamba processor.

    Returns MambaTemporalProcessor if mamba-ssm is available,
    otherwise returns MambaFallback.
    """
    if config is None:
        config = MambaConfig()

    if MAMBA_AVAILABLE:
        processor = MambaTemporalProcessor(
            d_model=config.d_model,
            d_state=config.d_state,
            n_layers=config.n_layers,
            output_dim=config.output_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
    else:
        processor = MambaFallback(
            d_model=config.d_model,
            output_dim=config.output_dim
        )

    return processor.to(config.device)


def get_mamba_processor() -> nn.Module:
    """Get or create the Mamba processor singleton."""
    global _mamba_processor
    if _mamba_processor is None:
        _mamba_processor = create_mamba_processor()
        _mamba_processor.eval()
    return _mamba_processor


def is_available() -> bool:
    """Check if Mamba-2 is available."""
    return MAMBA_AVAILABLE


def process_memory_history(
    embeddings: List[List[float]],
    timestamps: Optional[List[float]] = None
) -> Tuple[List[float], dict]:
    """
    Convenience function for processing memory history.

    Args:
        embeddings: List of memory embeddings [[e1], [e2], ...]
        timestamps: Optional timestamps for temporal ordering

    Returns:
        (context_vector[60], metadata)
    """
    processor = get_mamba_processor()
    return processor.process_memory_sequence(embeddings, timestamps)
