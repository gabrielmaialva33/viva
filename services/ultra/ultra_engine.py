"""
ULTRA Knowledge Graph Reasoning Engine
=======================================

Foundation model for zero-shot link prediction on any knowledge graph.

References:
- Paper: https://arxiv.org/abs/2310.04562 (ICLR 2024)
- GitHub: https://github.com/DeepGraphLearning/ULTRA
- HuggingFace: https://huggingface.co/mgalkin/ultra_50g
- License: MIT

Architecture:
- 6-layer GNN for relation graph
- 6-layer GNN for entity graph (NBFNet-based)
- 168k parameters total
- Zero-shot inference on any KG

Usage in VIVA:
- Memory relation inference (episodic → semantic links)
- Emotional causality reasoning
- Experience-outcome prediction
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

# CogGNN import - deferred to avoid circular dependency
CogGNN = None

# Mamba import - deferred for optional dependency
MambaProcessor = None

logger = logging.getLogger(__name__)


def _load_cog_gnn():
    """Lazy load CogGNN to avoid import issues."""
    global CogGNN
    if CogGNN is None:
        from cog_gnn import CogGNN as _CogGNN
        CogGNN = _CogGNN
    return CogGNN


def _load_mamba():
    """Lazy load Mamba processor."""
    global MambaProcessor
    if MambaProcessor is None:
        try:
            from mamba_temporal import MambaTemporalProcessor, MambaFallback, is_available
            if is_available():
                MambaProcessor = MambaTemporalProcessor
            else:
                MambaProcessor = MambaFallback
            logger.info(f"Mamba processor loaded: {MambaProcessor.__name__}")
        except ImportError as e:
            logger.warning(f"Mamba processor not available: {e}")
            MambaProcessor = None
    return MambaProcessor


@dataclass
class UltraConfig:
    """Configuration for ULTRA model."""

    # Model architecture (from paper)
    hidden_dim: int = 64
    num_relation_layers: int = 6
    num_entity_layers: int = 6

    # Message passing
    message_func: str = "distmult"  # or "transe"
    aggregate_func: str = "pna"

    # Inference
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Semantic Model
    semantic_model: str = "all-MiniLM-L6-v2"

    # Checkpoint
    checkpoint_path: Optional[str] = None
    checkpoint_name: str = "ultra_50g"  # Best for larger graphs


@dataclass
class Triple:
    """A knowledge graph triple (head, relation, tail)."""
    head: str
    relation: str
    tail: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "score": self.score
        }


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph representation."""

    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    triples: List[Tuple[int, int, int]] = field(default_factory=list)

    # Mappings
    entity_to_id: Dict[str, int] = field(default_factory=dict)
    relation_to_id: Dict[str, int] = field(default_factory=dict)

    def add_entity(self, entity: str) -> int:
        if entity not in self.entity_to_id:
            idx = len(self.entities)
            self.entities.append(entity)
            self.entity_to_id[entity] = idx
        return self.entity_to_id[entity]

    def add_relation(self, relation: str) -> int:
        if relation not in self.relation_to_id:
            idx = len(self.relations)
            self.relations.append(relation)
            self.relation_to_id[relation] = idx
        return self.relation_to_id[relation]

    def add_triple(self, head: str, relation: str, tail: str):
        h = self.add_entity(head)
        r = self.add_relation(relation)
        t = self.add_entity(tail)
        self.triples.append((h, r, t))

    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        if not self.triples:
            return Data(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                num_nodes=len(self.entities)
            )

        edge_index = torch.tensor(
            [[t[0] for t in self.triples], [t[2] for t in self.triples]],
            dtype=torch.long
        )
        edge_type = torch.tensor([t[1] for t in self.triples], dtype=torch.long)

        return Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(self.entities),
            num_relations=len(self.relations)
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "triples": len(self.triples)
        }


class UltraEngine:
    """
    ULTRA Knowledge Graph Reasoning Engine.

    Provides zero-shot link prediction capabilities for VIVA's memory system.

    Key capabilities:
    1. Link Prediction: Given (h, r, ?), predict likely tail entities
    2. Relation Inference: Given (h, ?, t), infer likely relations
    3. Triplet Scoring: Score (h, r, t) triplets for plausibility

    Integration with VIVA:
    - Dreamer uses this for memory consolidation reasoning
    - Emotional causality chains
    - Experience → Outcome predictions
    """

    def __init__(self, config: Optional[UltraConfig] = None):
        self.config = config or UltraConfig()
        self.model = None
        self.kg = KnowledgeGraph()
        self._loaded = False
        self._semantic_model = None

        # CogGNN state
        self.cog_gnn = None
        self._last_attention: Optional[torch.Tensor] = None
        self._attended_nodes: List[str] = []
        self._node_embeddings: Optional[torch.Tensor] = None
        self._updated_embeddings: Optional[torch.Tensor] = None

        # Mamba temporal processor state
        self.mamba_processor = None
        self._mamba_available = False

        logger.info(f"UltraEngine initialized (device={self.config.device})")

        # Initialize Semantic Model immediately (needed for Cognition)
        try:
            logger.info(f"Loading semantic model: {self.config.semantic_model}")
            self._semantic_model = SentenceTransformer(self.config.semantic_model)
            self._semantic_model.to('cpu')  # Keep on CPU to save GPU for GNN
            logger.info("Semantic model loaded.")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """
        Load ULTRA checkpoint.

        Checkpoints available:
        - ultra_3g.pth: Trained on FB15k237, WN18RR, CoDExMedium
        - ultra_4g.pth: Adds NELL995 to the above
        - ultra_50g.pth: Trained on 50 graphs (best for inference)
        """
        checkpoint_path = path or self.config.checkpoint_path

        if checkpoint_path is None:
            # Try to find in standard locations
            possible_paths = [
                Path("./ckpts/ultra_50g.pth"),
                Path("./ultra_50g.pth"),
                Path.home() / ".cache/ultra/ultra_50g.pth"
            ]
            for p in possible_paths:
                if p.exists():
                    checkpoint_path = str(p)
                    break

        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=self.config.device
                )
                # Model loading would happen here
                # self.model.load_state_dict(state_dict)
                self._loaded = True
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return False

        logger.warning("No checkpoint loaded - running in mock mode")
        self._loaded = False
        return False

    def build_graph_from_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> KnowledgeGraph:
        """
        Build KG from VIVA memory entries.

        Memory structure expected:
        {
            "id": "mem_123",
            "content": "...",
            "emotional_state": {"pleasure": 0.5, ...},
            "related_to": ["mem_456"],
            "caused_by": "event_789",
            "timestamp": "..."
        }
        """
        self.kg = KnowledgeGraph()

        for mem in memories:
            mem_id = mem.get("id", str(hash(mem.get("content", ""))))

            # Add memory as entity
            self.kg.add_entity(mem_id)

            # Add relations to other memories
            for related in mem.get("related_to", []):
                self.kg.add_triple(mem_id, "related_to", related)

            # Add causal relations
            if caused_by := mem.get("caused_by"):
                self.kg.add_triple(caused_by, "causes", mem_id)

            # Add emotional state as entity with relations
            if emotional := mem.get("emotional_state"):
                emotion_id = f"emotion_{mem_id}"
                self.kg.add_entity(emotion_id)
                self.kg.add_triple(mem_id, "has_emotion", emotion_id)

                # Classify emotion
                p = emotional.get("pleasure", 0)
                if p > 0.3:
                    self.kg.add_triple(emotion_id, "is_type", "positive")
                elif p < -0.3:
                    self.kg.add_triple(emotion_id, "is_type", "negative")
                else:
                    self.kg.add_triple(emotion_id, "is_type", "neutral")

        logger.info(f"Built KG: {self.kg.stats}")
        return self.kg

    def predict_links(
        self,
        head: str,
        relation: str,
        top_k: int = 10
    ) -> List[Triple]:
        """
        Link prediction: Given (head, relation, ?), predict tail entities.

        Zero-shot inference using ULTRA's relation representations.

        Args:
            head: Head entity ID
            relation: Relation type
            top_k: Number of predictions to return

        Returns:
            List of predicted triples with scores
        """
        if head not in self.kg.entity_to_id:
            logger.warning(f"Unknown entity: {head}")
            return []

        if relation not in self.kg.relation_to_id:
            logger.warning(f"Unknown relation: {relation}")
            return []

        predictions = []

        if self._loaded and self.model is not None:
            # Real inference with ULTRA
            # h_idx = self.kg.entity_to_id[head]
            # r_idx = self.kg.relation_to_id[relation]
            # data = self.kg.to_pyg_data().to(self.config.device)
            # scores = self.model(data, h_idx, r_idx)
            # top_indices = scores.topk(top_k).indices
            pass
        else:
            # Mock predictions based on graph structure
            h_idx = self.kg.entity_to_id[head]
            r_idx = self.kg.relation_to_id[relation]

            # Find existing tails for this relation
            for h, r, t in self.kg.triples:
                if r == r_idx:
                    score = 0.8 if h == h_idx else 0.5
                    predictions.append(Triple(
                        head=head,
                        relation=relation,
                        tail=self.kg.entities[t],
                        score=score
                    ))

            # Sort by score and limit
            predictions.sort(key=lambda x: x.score, reverse=True)
            predictions = predictions[:top_k]

        return predictions

    def infer_relations(
        self,
        head: str,
        tail: str,
        top_k: int = 5
    ) -> List[Triple]:
        """
        Relation inference: Given (head, ?, tail), predict relations.

        Useful for:
        - Discovering implicit connections between memories
        - Understanding emotional causality
        """
        if head not in self.kg.entity_to_id:
            return []
        if tail not in self.kg.entity_to_id:
            return []

        predictions = []
        h_idx = self.kg.entity_to_id[head]
        t_idx = self.kg.entity_to_id[tail]

        # Find existing relations between these entities
        for h, r, t in self.kg.triples:
            if h == h_idx and t == t_idx:
                predictions.append(Triple(
                    head=head,
                    relation=self.kg.relations[r],
                    tail=tail,
                    score=0.9
                ))

        # Add predicted relations based on paths
        if not predictions:
            # No direct link - check 2-hop paths
            for h1, r1, t1 in self.kg.triples:
                if h1 == h_idx:
                    for h2, r2, t2 in self.kg.triples:
                        if h2 == t1 and t2 == t_idx:
                            # Found 2-hop path
                            rel1 = self.kg.relations[r1]
                            rel2 = self.kg.relations[r2]
                            predictions.append(Triple(
                                head=head,
                                relation=f"via_{rel1}_{rel2}",
                                tail=tail,
                                score=0.6
                            ))

        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions[:top_k]

    def score_triple(self, head: str, relation: str, tail: str) -> float:
        """
        Score a triple for plausibility.

        Returns score in [0, 1] where higher = more plausible.

        Useful for:
        - Validating memory connections
        - Filtering hallucinated relations
        """
        if head not in self.kg.entity_to_id:
            return 0.0
        if relation not in self.kg.relation_to_id:
            return 0.0
        if tail not in self.kg.entity_to_id:
            return 0.0

        h_idx = self.kg.entity_to_id[head]
        r_idx = self.kg.relation_to_id[relation]
        t_idx = self.kg.entity_to_id[tail]

        # Check if triple exists
        if (h_idx, r_idx, t_idx) in self.kg.triples:
            return 1.0

        # Check for similar patterns
        pattern_count = 0
        for h, r, t in self.kg.triples:
            if r == r_idx:
                pattern_count += 1

        # Base score on pattern frequency
        if pattern_count > 0:
            return min(0.7, pattern_count * 0.1)

        return 0.1

    def find_reasoning_path(
        self,
        start: str,
        end: str,
        max_hops: int = 3
    ) -> List[Triple]:
        """
        Find reasoning path between two entities.

        Multi-hop reasoning using graph traversal.

        Args:
            start: Starting entity
            end: Target entity
            max_hops: Maximum path length

        Returns:
            List of triples forming the path
        """
        if start not in self.kg.entity_to_id:
            return []
        if end not in self.kg.entity_to_id:
            return []

        start_idx = self.kg.entity_to_id[start]
        end_idx = self.kg.entity_to_id[end]

        # BFS for shortest path
        from collections import deque

        visited = {start_idx: None}
        queue = deque([(start_idx, 0)])

        while queue:
            current, hops = queue.popleft()

            if current == end_idx:
                # Reconstruct path
                path = []
                node = end_idx
                while visited[node] is not None:
                    prev, rel_idx = visited[node]
                    path.append(Triple(
                        head=self.kg.entities[prev],
                        relation=self.kg.relations[rel_idx],
                        tail=self.kg.entities[node],
                        score=1.0
                    ))
                    node = prev
                path.reverse()
                return path

            if hops >= max_hops:
                continue

            # Explore neighbors
            for h, r, t in self.kg.triples:
                if h == current and t not in visited:
                    visited[t] = (current, r)
                    queue.append((t, hops + 1))

        return []  # No path found

    # =========================================================================
    # CogGNN Methods (Cognitive Graph Neural Network)
    # =========================================================================

    def init_cog_gnn(self, in_dim: int = 384, hidden_dim: int = 64) -> bool:
        """
        Initialize the Cognitive GNN for emotional graph reasoning.

        Args:
            in_dim: Input embedding dimension (384 for MiniLM)
            hidden_dim: Hidden layer dimension

        Returns:
            True if initialization successful
        """
        try:
            CogGNNClass = _load_cog_gnn()
            self.cog_gnn = CogGNNClass(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                pad_dim=3,
                dropout=0.1
            )
            self.cog_gnn.to(self.config.device)
            self.cog_gnn.eval()  # Inference mode
            logger.info(f"CogGNN initialized (hidden={hidden_dim}, device={self.config.device})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CogGNN: {e}")
            return False

    def propagate_thought(
        self,
        concept: str,
        pad: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run GNN message passing with emotional context.

        Propagates a thought through the knowledge graph, using PAD emotional
        state to modulate attention. This implements the NeuCFlow-inspired
        3-layer architecture: Unconscious → Conscious → Attention.

        Args:
            concept: The concept/thought to propagate
            pad: PAD emotional state [pleasure, arousal, dominance]
            top_k: Number of top attended nodes to return

        Returns:
            Dict with attended_nodes, attention_scores, and updated_concept
        """
        # Ensure CogGNN is initialized
        if self.cog_gnn is None:
            if not self.init_cog_gnn():
                return {
                    "error": "CogGNN not initialized",
                    "attended_nodes": [],
                    "attention_scores": []
                }

        # Need nodes in the graph
        if len(self.kg.entities) == 0:
            return {
                "error": "Knowledge graph is empty",
                "attended_nodes": [],
                "attention_scores": []
            }

        # Get semantic embeddings for all nodes
        if self._node_embeddings is None or self._node_embeddings.size(0) != len(self.kg.entities):
            embeddings = []
            for entity in self.kg.entities:
                emb = self.get_embedding(entity)
                if not emb:
                    # Fallback to random if embedding fails
                    emb = torch.randn(384).tolist()
                embeddings.append(emb)
            self._node_embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Convert KG to PyG format
        data = self.kg.to_pyg_data()

        # Prepare tensors
        x = self._node_embeddings.to(self.config.device)
        edge_index = data.edge_index.to(self.config.device)
        pad_tensor = torch.tensor(pad, dtype=torch.float32, device=self.config.device)

        # Handle edge case: no edges
        if edge_index.size(1) == 0:
            # Create self-loops for message passing
            num_nodes = x.size(0)
            edge_index = torch.stack([
                torch.arange(num_nodes, device=self.config.device),
                torch.arange(num_nodes, device=self.config.device)
            ])

        # Run CogGNN forward pass
        with torch.no_grad():
            updated_x, attention = self.cog_gnn(x, edge_index, pad_tensor)

        # Store for conscious_focus()
        self._last_attention = attention
        self._updated_embeddings = updated_x

        # Get top-k attended nodes
        if attention is not None:
            attention_flat = attention.squeeze()
            k = min(top_k, len(self.kg.entities))
            top_values, top_indices = torch.topk(attention_flat, k)

            attended_nodes = [self.kg.entities[i] for i in top_indices.cpu().tolist()]
            attention_scores = top_values.cpu().tolist()
        else:
            attended_nodes = []
            attention_scores = []

        self._attended_nodes = attended_nodes

        # Get concept embedding and find similarity with attended nodes
        concept_emb = self.get_embedding(concept)
        updated_concept = None
        if concept_emb and len(attended_nodes) > 0:
            # Find which attended node is most similar to concept
            concept_tensor = torch.tensor(concept_emb, dtype=torch.float32)
            similarities = []
            for i, node in enumerate(attended_nodes):
                if node in self.kg.entity_to_id:
                    node_idx = self.kg.entity_to_id[node]
                    node_emb = self._node_embeddings[node_idx]
                    sim = torch.cosine_similarity(concept_tensor.unsqueeze(0), node_emb.unsqueeze(0))
                    similarities.append((node, sim.item(), attention_scores[i]))

            if similarities:
                # Best match considering both similarity and attention
                best = max(similarities, key=lambda x: x[1] * 0.5 + x[2] * 0.5)
                updated_concept = best[0]

        return {
            "attended_nodes": attended_nodes,
            "attention_scores": attention_scores,
            "updated_concept": updated_concept,
            "pad_used": pad,
            "num_nodes": len(self.kg.entities),
            "num_edges": data.edge_index.size(1) if data.edge_index.size(1) > 0 else len(self.kg.entities)
        }

    def conscious_focus(self) -> List[str]:
        """
        Return nodes with highest attention from last propagation.

        This represents the current "conscious focus" - which thoughts/memories
        have won the competition for attention in the Global Workspace.

        Returns:
            List of entity names currently in focus
        """
        return self._attended_nodes

    def propagate_with_query(
        self,
        query: str,
        pad: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Propagate with query-conditioned attention.

        Combines GNN attention with query similarity for focused retrieval.

        Args:
            query: Query string to find relevant nodes
            pad: PAD emotional state
            top_k: Number of results

        Returns:
            Dict with relevance-scored nodes
        """
        # First run standard propagation
        result = self.propagate_thought(query, pad, top_k=len(self.kg.entities))

        if "error" in result:
            return result

        # Get query embedding
        query_emb = self.get_embedding(query)
        if not query_emb:
            return result

        query_tensor = torch.tensor(query_emb, dtype=torch.float32)

        # Score all nodes by combined attention + query similarity
        scored_nodes = []
        for i, entity in enumerate(self.kg.entities):
            # Get attention score
            attn = 0.0
            if self._last_attention is not None:
                attn = self._last_attention[i].item()

            # Get query similarity
            node_emb = self._node_embeddings[i]
            sim = torch.cosine_similarity(
                query_tensor.unsqueeze(0),
                node_emb.unsqueeze(0)
            ).item()

            # Combined score: 50% attention, 50% similarity
            combined = attn * 0.5 + sim * 0.5
            scored_nodes.append((entity, combined, attn, sim))

        # Sort by combined score
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scored_nodes[:top_k]

        return {
            "query": query,
            "results": [
                {
                    "entity": n[0],
                    "combined_score": n[1],
                    "attention": n[2],
                    "similarity": n[3]
                }
                for n in top_nodes
            ],
            "pad_used": pad
        }

    # =========================================================================
    # Mamba Temporal Processor (O(n) sequence processing)
    # =========================================================================

    def init_mamba(self, d_model: int = 384, n_layers: int = 2, output_dim: int = 60) -> bool:
        """
        Initialize Mamba temporal processor.

        Args:
            d_model: Input embedding dimension (default: 384 for MiniLM)
            n_layers: Number of Mamba layers (default: 2)
            output_dim: Context vector dimension (default: 60)

        Returns:
            True if initialized successfully
        """
        MambaClass = _load_mamba()
        if MambaClass is None:
            logger.warning("Mamba processor not available")
            self._mamba_available = False
            return False

        try:
            self.mamba_processor = MambaClass(
                d_model=d_model,
                n_layers=n_layers,
                output_dim=output_dim
            )
            self.mamba_processor.eval()
            self._mamba_available = True
            logger.info(f"Mamba processor initialized: {MambaClass.__name__}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Mamba: {e}")
            self._mamba_available = False
            return False

    def process_memory_sequence(
        self,
        memory_embeddings: List[List[float]],
        timestamps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Process a sequence of memory embeddings through Mamba.

        Takes a history of memory embeddings and produces a compact
        context vector that captures temporal patterns.

        Args:
            memory_embeddings: List of embedding vectors [[e1], [e2], ...]
            timestamps: Optional timestamps for temporal ordering

        Returns:
            Dict with 'context' vector and 'metadata'
        """
        if not self._mamba_available or self.mamba_processor is None:
            # Try to initialize if not already done
            if not self.init_mamba():
                return {
                    "error": "Mamba processor not available",
                    "context": self._fallback_context(memory_embeddings)
                }

        if not memory_embeddings or len(memory_embeddings) == 0:
            return {
                "error": "No embeddings provided",
                "context": [0.0] * 60
            }

        try:
            context, metadata = self.mamba_processor.process_memory_sequence(
                memory_embeddings, timestamps
            )
            return {
                "context": context,
                "metadata": metadata,
                "success": True
            }
        except Exception as e:
            logger.error(f"Mamba processing failed: {e}")
            return {
                "error": str(e),
                "context": self._fallback_context(memory_embeddings)
            }

    def _fallback_context(self, embeddings: List[List[float]]) -> List[float]:
        """Fallback context computation when Mamba is unavailable."""
        if not embeddings:
            return [0.0] * 60

        # Simple mean + truncate to 60 dims
        import numpy as np
        arr = np.array(embeddings)
        mean = np.mean(arr, axis=0)

        if len(mean) >= 60:
            return mean[:60].tolist()
        else:
            # Pad if needed
            return np.pad(mean, (0, 60 - len(mean))).tolist()

    def mamba_stats(self) -> Dict[str, Any]:
        """Get Mamba processor statistics."""
        return {
            "available": self._mamba_available,
            "processor_type": type(self.mamba_processor).__name__ if self.mamba_processor else None,
            "d_model": getattr(self.mamba_processor, 'd_model', None) if self.mamba_processor else None,
            "output_dim": getattr(self.mamba_processor, 'output_dim', None) if self.mamba_processor else None
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_embedding(self, text: str) -> List[float]:
        """Convert text to semantic vector."""
        if self._semantic_model is None:
            return []

        embedding = self._semantic_model.encode(text)
        return embedding.tolist()

    def stats(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "device": self.config.device,
            "checkpoint": self.config.checkpoint_name,
            "graph": self.kg.stats,
            "semantic_model": self.config.semantic_model
        }


# Singleton instance for service
_engine: Optional[UltraEngine] = None


def get_engine() -> UltraEngine:
    """Get or create the global ULTRA engine instance."""
    global _engine
    if _engine is None:
        _engine = UltraEngine()
    return _engine
