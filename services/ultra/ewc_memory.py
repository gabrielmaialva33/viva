"""
EWC Memory Protection - VIVA Ultra

Implements Elastic Weight Consolidation for protecting important memories
during continuous learning. Prevents catastrophic forgetting by penalizing
changes to embeddings that are critical for consolidated memories.

Reference: Kirkpatrick et al. 2017 - "Overcoming catastrophic forgetting in neural networks"

Key Concepts:
- Fisher Information: Measures importance of each embedding dimension
- Consolidation Score: How important a memory is (from Dreamer DRE scoring)
- EWC Penalty: L_ewc = λ/2 × Σ F_i × (θ_i - θ*_i)²

Integration:
- Called by Dreamer during memory consolidation
- Protects memories with consolidation_score > 0.7
- Stores Fisher info in Qdrant payload
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProtectedMemory:
    """A memory protected by EWC."""
    memory_id: str
    baseline_embedding: np.ndarray  # θ* (embedding at consolidation time)
    fisher_diagonal: np.ndarray     # F_i (importance per dimension)
    consolidation_score: float      # DRE score from Dreamer
    consolidated_at: float          # Timestamp


@dataclass
class EWCConfig:
    """Configuration for EWC Memory Manager."""
    lambda_ewc: float = 0.4           # Regularization strength
    min_consolidation_score: float = 0.7  # Minimum score to protect
    max_protected_memories: int = 1000    # Memory limit
    decay_rate: float = 0.01          # Fisher info decay per consolidation cycle
    embedding_dim: int = 384          # MiniLM embedding dimension


class EWCMemoryManager:
    """
    Manages EWC protection for consolidated memories.

    Workflow:
    1. Dreamer consolidates memory with score > 0.7
    2. compute_fisher_diagonal() calculates importance per dimension
    3. protect_memory() stores baseline + Fisher info
    4. On new memories: compute_ewc_penalty() penalizes changes to protected dims

    The penalty ensures that learning new memories doesn't destroy
    the embedding relationships that important memories depend on.
    """

    def __init__(self, config: Optional[EWCConfig] = None):
        self.config = config or EWCConfig()
        self.protected_memories: Dict[str, ProtectedMemory] = {}
        self.total_fisher: Optional[np.ndarray] = None  # Accumulated Fisher

        logger.info(
            f"EWCMemoryManager initialized: λ={self.config.lambda_ewc}, "
            f"min_score={self.config.min_consolidation_score}"
        )

    def compute_fisher_diagonal(
        self,
        embedding: np.ndarray,
        related_embeddings: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute Fisher Information diagonal for an embedding.

        Fisher Information measures how sensitive the model is to changes
        in each embedding dimension. High Fisher = changing this dimension
        would significantly affect related memories.

        Approximation: F_i ≈ 1 / (1 + variance_i)
        Low variance in a dimension across related memories means it's
        more important for that concept cluster.

        Args:
            embedding: The memory embedding to protect [384]
            related_embeddings: Nearby memories in embedding space [N, 384]
            weights: Optional importance weights for related memories [N]

        Returns:
            Fisher diagonal [384] - importance per dimension
        """
        if len(related_embeddings) == 0:
            # No context: uniform importance
            return np.ones(self.config.embedding_dim) * 0.5

        # Weighted variance across related embeddings
        if weights is not None:
            # Normalize weights
            weights = weights / (weights.sum() + 1e-8)
            # Weighted mean
            mean = np.average(related_embeddings, axis=0, weights=weights)
            # Weighted variance
            diff = related_embeddings - mean
            variance = np.average(diff ** 2, axis=0, weights=weights)
        else:
            variance = np.var(related_embeddings, axis=0)

        # Fisher ≈ 1 / (1 + variance)
        # Low variance → high importance (consistent across cluster)
        fisher = 1.0 / (1.0 + variance + 1e-8)

        # Normalize to [0, 1]
        fisher = fisher / (fisher.max() + 1e-8)

        return fisher.astype(np.float32)

    def protect_memory(
        self,
        memory_id: str,
        embedding: np.ndarray,
        related_embeddings: np.ndarray,
        consolidation_score: float,
        related_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Protect a consolidated memory with EWC.

        Called by Dreamer after memory consolidation.

        Args:
            memory_id: Qdrant point ID
            embedding: Memory embedding [384]
            related_embeddings: Embeddings of related memories [N, 384]
            consolidation_score: DRE score from Dreamer (0-1)
            related_weights: Importance weights for related memories

        Returns:
            Dict with protection status and payload for Qdrant
        """
        # Check minimum score
        if consolidation_score < self.config.min_consolidation_score:
            return {
                "protected": False,
                "reason": f"score {consolidation_score:.2f} < min {self.config.min_consolidation_score}"
            }

        # Check capacity
        if len(self.protected_memories) >= self.config.max_protected_memories:
            self._evict_weakest_memory()

        # Compute Fisher diagonal
        fisher = self.compute_fisher_diagonal(
            embedding, related_embeddings, related_weights
        )

        # Create protected memory
        protected = ProtectedMemory(
            memory_id=memory_id,
            baseline_embedding=embedding.copy(),
            fisher_diagonal=fisher,
            consolidation_score=consolidation_score,
            consolidated_at=self._timestamp()
        )

        self.protected_memories[memory_id] = protected

        # Update total Fisher (for global penalty)
        self._update_total_fisher(fisher, consolidation_score)

        logger.debug(
            f"Protected memory {memory_id}: score={consolidation_score:.2f}, "
            f"fisher_mean={fisher.mean():.3f}"
        )

        # Return Qdrant payload
        return {
            "protected": True,
            "qdrant_payload": {
                "ewc_fisher_info": fisher.tolist(),
                "ewc_baseline_embedding": embedding.tolist(),
                "ewc_consolidation_score": consolidation_score,
                "ewc_consolidated_at": protected.consolidated_at
            }
        }

    def compute_ewc_penalty(
        self,
        new_embedding: np.ndarray,
        affected_memory_ids: Optional[List[str]] = None
    ) -> Tuple[float, Dict]:
        """
        Compute EWC penalty for a new/modified embedding.

        L_ewc = λ/2 × Σ_i F_i × (θ_i - θ*_i)²

        Used to penalize changes that would affect important memories.

        Args:
            new_embedding: The new embedding to evaluate [384]
            affected_memory_ids: Specific memories to check (None = all)

        Returns:
            (penalty_value, details_dict)
        """
        if not self.protected_memories:
            return 0.0, {"reason": "no_protected_memories"}

        # Get memories to check
        if affected_memory_ids:
            memories = [
                self.protected_memories[mid]
                for mid in affected_memory_ids
                if mid in self.protected_memories
            ]
        else:
            memories = list(self.protected_memories.values())

        if not memories:
            return 0.0, {"reason": "no_affected_memories"}

        total_penalty = 0.0
        contributions = []

        for mem in memories:
            # Squared difference from baseline
            diff = new_embedding - mem.baseline_embedding
            diff_squared = diff ** 2

            # Weighted by Fisher (importance) and consolidation score
            weighted_penalty = mem.consolidation_score * np.sum(
                mem.fisher_diagonal * diff_squared
            )
            total_penalty += weighted_penalty

            contributions.append({
                "memory_id": mem.memory_id,
                "penalty": float(weighted_penalty),
                "score": mem.consolidation_score
            })

        # Apply lambda and normalize
        final_penalty = (self.config.lambda_ewc / 2.0) * total_penalty

        return float(final_penalty), {
            "total_memories_checked": len(memories),
            "top_contributions": sorted(
                contributions, key=lambda x: x["penalty"], reverse=True
            )[:5]
        }

    def get_protection_status(self, memory_id: str) -> Optional[Dict]:
        """Get protection status for a memory."""
        if memory_id not in self.protected_memories:
            return None

        mem = self.protected_memories[memory_id]
        return {
            "memory_id": mem.memory_id,
            "consolidation_score": mem.consolidation_score,
            "fisher_mean": float(mem.fisher_diagonal.mean()),
            "fisher_std": float(mem.fisher_diagonal.std()),
            "consolidated_at": mem.consolidated_at
        }

    def get_stats(self) -> Dict:
        """Get EWC manager statistics."""
        if not self.protected_memories:
            return {
                "protected_count": 0,
                "total_fisher": None
            }

        scores = [m.consolidation_score for m in self.protected_memories.values()]
        return {
            "protected_count": len(self.protected_memories),
            "avg_consolidation_score": float(np.mean(scores)),
            "max_consolidation_score": float(np.max(scores)),
            "total_fisher_mean": float(self.total_fisher.mean()) if self.total_fisher is not None else None,
            "lambda_ewc": self.config.lambda_ewc
        }

    def decay_fisher_info(self):
        """
        Apply decay to Fisher information over time.

        Called periodically to allow some plasticity for old memories.
        """
        for mem in self.protected_memories.values():
            mem.fisher_diagonal *= (1.0 - self.config.decay_rate)

        if self.total_fisher is not None:
            self.total_fisher *= (1.0 - self.config.decay_rate)

        logger.debug(f"Applied Fisher decay: rate={self.config.decay_rate}")

    def load_from_qdrant_payload(self, memory_id: str, payload: Dict) -> bool:
        """
        Load protection from Qdrant payload.

        Called on startup to restore protected memories from Qdrant.
        """
        if "ewc_fisher_info" not in payload:
            return False

        try:
            protected = ProtectedMemory(
                memory_id=memory_id,
                baseline_embedding=np.array(payload["ewc_baseline_embedding"], dtype=np.float32),
                fisher_diagonal=np.array(payload["ewc_fisher_info"], dtype=np.float32),
                consolidation_score=payload["ewc_consolidation_score"],
                consolidated_at=payload.get("ewc_consolidated_at", 0.0)
            )
            self.protected_memories[memory_id] = protected
            self._update_total_fisher(protected.fisher_diagonal, protected.consolidation_score)
            return True
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to load EWC from payload: {e}")
            return False

    def _update_total_fisher(self, fisher: np.ndarray, score: float):
        """Update accumulated Fisher information."""
        weighted_fisher = fisher * score
        if self.total_fisher is None:
            self.total_fisher = weighted_fisher.copy()
        else:
            self.total_fisher += weighted_fisher

    def _evict_weakest_memory(self):
        """Remove the memory with lowest consolidation score."""
        if not self.protected_memories:
            return

        weakest_id = min(
            self.protected_memories.keys(),
            key=lambda mid: self.protected_memories[mid].consolidation_score
        )
        del self.protected_memories[weakest_id]
        logger.debug(f"Evicted weakest protected memory: {weakest_id}")

    def _timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()


# Singleton instance
_ewc_manager: Optional[EWCMemoryManager] = None


def get_ewc_manager() -> EWCMemoryManager:
    """Get or create the EWC manager singleton."""
    global _ewc_manager
    if _ewc_manager is None:
        _ewc_manager = EWCMemoryManager()
    return _ewc_manager


def compute_ewc_for_consolidation(
    memory_id: str,
    embedding: np.ndarray,
    related_embeddings: np.ndarray,
    consolidation_score: float
) -> Dict:
    """
    Convenience function for Dreamer integration.

    Call this during memory consolidation to protect important memories.

    Args:
        memory_id: Qdrant point ID
        embedding: Memory embedding
        related_embeddings: Related memory embeddings
        consolidation_score: DRE score from Dreamer

    Returns:
        Dict with 'protected' status and 'qdrant_payload' if protected
    """
    manager = get_ewc_manager()
    return manager.protect_memory(
        memory_id=memory_id,
        embedding=embedding,
        related_embeddings=related_embeddings,
        consolidation_score=consolidation_score
    )
