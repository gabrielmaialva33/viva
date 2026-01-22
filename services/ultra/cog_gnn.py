"""
CogGNN: Cognitive Graph Neural Network for VIVA

A 3-layer GNN architecture inspired by NeuCFlow (arXiv:1905.13049)
that models consciousness as message passing through a knowledge graph.

Architecture:
    Layer 1 (Unconscious): Background sensory fusion via GAT (4 heads)
    Layer 2 (Conscious): Active reasoning with emotional modulation (2 heads)
    Layer 3 (Attention): Focus selection for Global Workspace broadcast

The network integrates PAD emotional state into all node representations,
allowing emotional context to modulate graph attention patterns.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

logger = logging.getLogger(__name__)


class CogGNN(nn.Module):
    """
    Cognitive Graph Neural Network - 3-layer NeuCFlow-inspired architecture.

    Processes a knowledge graph with emotional context to produce:
    - Updated node embeddings (after message passing)
    - Attention scores (for consciousness focus selection)

    Args:
        in_dim: Input embedding dimension (default: 384 for MiniLM)
        hidden_dim: Hidden layer dimension (default: 64)
        pad_dim: PAD emotional state dimension (default: 3)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 64,
        pad_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.pad_dim = pad_dim

        # Layer 1: Unconscious Flow (background sensory fusion)
        # 4 attention heads → captures diverse relational patterns
        self.unconscious = GATConv(
            in_channels=in_dim + pad_dim,
            out_channels=hidden_dim,
            heads=4,
            dropout=dropout,
            concat=True,  # Output: hidden_dim * 4
        )

        # Layer 2: Conscious Flow (active reasoning)
        # 2 attention heads → focused deliberation
        self.conscious = GATConv(
            in_channels=hidden_dim * 4,
            out_channels=hidden_dim,
            heads=2,
            dropout=dropout,
            concat=True,  # Output: hidden_dim * 2
        )

        # Layer 3: Attention Flow (focus selection)
        # Single scalar attention score per node
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Optional: Emotion-aware projection
        self.emotion_gate = nn.Sequential(
            nn.Linear(pad_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self._init_weights()
        logger.info(
            f"CogGNN initialized: in={in_dim}, hidden={hidden_dim}, pad={pad_dim}"
        )

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pad_state: torch.Tensor,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the cognitive GNN.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            pad_state: PAD emotional state [3] or [1, 3]
            return_attention: Whether to compute attention scores

        Returns:
            updated_x: Updated node embeddings [num_nodes, hidden_dim * 2]
            attention: Node attention scores [num_nodes, 1] (if return_attention)
        """
        num_nodes = x.size(0)

        # Ensure PAD state is properly shaped
        if pad_state.dim() == 1:
            pad_state = pad_state.unsqueeze(0)

        # Expand PAD to all nodes: [1, 3] → [num_nodes, 3]
        pad_expanded = pad_state.expand(num_nodes, -1)

        # Concatenate emotional context to node features
        x = torch.cat([x, pad_expanded], dim=-1)  # [num_nodes, in_dim + 3]

        # Layer 1: Unconscious Flow
        x = self.unconscious(x, edge_index)
        x = F.elu(x)

        # Layer 2: Conscious Flow
        x = self.conscious(x, edge_index)
        x = F.elu(x)

        # Apply emotion gating (modulate by arousal/valence)
        emotion_weight = self.emotion_gate(pad_expanded)  # [num_nodes, hidden_dim]
        # Expand to match x dimensions
        emotion_weight = emotion_weight.repeat(1, 2)  # [num_nodes, hidden_dim * 2]
        x = x * emotion_weight  # Element-wise modulation

        # Layer 3: Attention scoring
        attention = None
        if return_attention:
            attention = torch.sigmoid(self.attention(x))  # [num_nodes, 1]

        return x, attention

    def propagate_with_query(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pad_state: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate with a query concept to find relevant nodes.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            pad_state: PAD emotional state [3]
            query_embedding: Query concept embedding [in_dim]

        Returns:
            updated_x: Updated node embeddings
            relevance: Query-conditioned attention scores
        """
        # Get base attention
        updated_x, base_attention = self.forward(x, edge_index, pad_state)

        # Compute query similarity
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        # Cosine similarity between query and updated nodes
        # Note: Only use the first hidden_dim dimensions for comparison
        node_norm = F.normalize(updated_x[:, : self.hidden_dim], dim=-1)
        query_projected = F.normalize(
            query_embedding[:, : self.hidden_dim]
            if query_embedding.size(-1) >= self.hidden_dim
            else F.pad(query_embedding, (0, self.hidden_dim - query_embedding.size(-1))),
            dim=-1,
        )

        # Similarity: [num_nodes, 1]
        similarity = torch.mm(node_norm, query_projected.t())

        # Combine base attention with query relevance
        relevance = base_attention * 0.5 + similarity * 0.5

        return updated_x, relevance


class CogGNNConfig:
    """Configuration for CogGNN model."""

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 64,
        pad_dim: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.pad_dim = pad_dim
        self.dropout = dropout
        self.device = device

    def to_dict(self) -> Dict:
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "pad_dim": self.pad_dim,
            "dropout": self.dropout,
            "device": self.device,
        }


def create_cog_gnn(config: Optional[CogGNNConfig] = None) -> CogGNN:
    """Factory function to create CogGNN with optional config."""
    if config is None:
        config = CogGNNConfig()

    model = CogGNN(
        in_dim=config.in_dim,
        hidden_dim=config.hidden_dim,
        pad_dim=config.pad_dim,
        dropout=config.dropout,
    )

    return model.to(config.device)
