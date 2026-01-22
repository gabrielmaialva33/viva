"""
ULTRA Service - FastAPI Server
==============================

HTTP API for ULTRA Knowledge Graph Reasoning.

Reference: https://github.com/DeepGraphLearning/ULTRA

Endpoints:
- POST /graph/build     - Build KG from memories
- POST /predict/link    - Link prediction (h, r, ?)
- POST /predict/relation - Relation inference (h, ?, t)
- POST /score           - Score triple plausibility
- POST /path            - Find reasoning path
- GET  /health          - Health check
- GET  /stats           - Engine statistics

Integration:
- VIVA calls this from VivaBridge.Ultra (Elixir)
- Runs on port 8765 by default
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ultra_engine import get_engine, Triple, UltraConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ultra.server")


# ============================================================================
# Request/Response Models
# ============================================================================

class MemoryEntry(BaseModel):
    """A memory entry from VIVA."""
    id: str
    content: Optional[str] = None
    emotional_state: Optional[Dict[str, float]] = None
    related_to: List[str] = Field(default_factory=list)
    caused_by: Optional[str] = None
    timestamp: Optional[str] = None


class BuildGraphRequest(BaseModel):
    """Request to build KG from memories."""
    memories: List[MemoryEntry]


class LinkPredictionRequest(BaseModel):
    """Request for link prediction (h, r, ?)."""
    head: str
    relation: str
    top_k: int = Field(default=10, ge=1, le=100)


class RelationInferenceRequest(BaseModel):
    """Request for relation inference (h, ?, t)."""
    head: str
    tail: str
    top_k: int = Field(default=5, ge=1, le=50)


class ScoreTripleRequest(BaseModel):
    """Request to score a triple."""
    head: str
    relation: str
    tail: str


class FindPathRequest(BaseModel):
    """Request to find reasoning path."""
    start: str
    end: str
    max_hops: int = Field(default=3, ge=1, le=10)


class TripleResponse(BaseModel):
    """A triple with score."""
    head: str
    relation: str
    tail: str
    score: float


class GraphStats(BaseModel):
    """Knowledge graph statistics."""
    entities: int
    relations: int
    triples: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine_loaded: bool
    device: str
    checkpoint: str


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting ULTRA service...")

    engine = get_engine()

    # Try to load checkpoint
    if engine.load_checkpoint():
        logger.info("ULTRA checkpoint loaded successfully")
    else:
        logger.warning("Running without checkpoint (mock mode)")

    yield

    logger.info("ULTRA service shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ULTRA KG Reasoning Service",
    description="""
    Foundation model for zero-shot knowledge graph reasoning.

    **References:**
    - Paper: [Towards Foundation Models for Knowledge Graph Reasoning](https://arxiv.org/abs/2310.04562) (ICLR 2024)
    - GitHub: [DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA)
    - Model: 168k parameters, zero-shot on any KG

    **Integration with VIVA:**
    - Memory relation inference
    - Emotional causality reasoning
    - Experience-outcome prediction
    """,
    version="0.1.0",
    lifespan=lifespan
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and engine status."""
    engine = get_engine()
    stats = engine.stats()
    return HealthResponse(
        status="healthy",
        engine_loaded=stats["loaded"],
        device=stats["device"],
        checkpoint=stats["checkpoint"]
    )


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get engine and graph statistics."""
    engine = get_engine()
    return engine.stats()


@app.post("/graph/build", response_model=GraphStats)
async def build_graph(request: BuildGraphRequest):
    """
    Build knowledge graph from VIVA memory entries.

    This creates the graph structure that ULTRA will reason over.
    Call this when memories are updated in the Dreamer.
    """
    engine = get_engine()

    memories = [m.model_dump() for m in request.memories]
    kg = engine.build_graph_from_memories(memories)

    return GraphStats(**kg.stats)


@app.post("/predict/link", response_model=List[TripleResponse])
async def predict_links(request: LinkPredictionRequest):
    """
    Link prediction: Given (head, relation, ?), predict tail entities.

    **Use case:** "What memories might be related to this one?"
    """
    engine = get_engine()

    predictions = engine.predict_links(
        head=request.head,
        relation=request.relation,
        top_k=request.top_k
    )

    return [TripleResponse(**p.to_dict()) for p in predictions]


@app.post("/predict/relation", response_model=List[TripleResponse])
async def predict_relations(request: RelationInferenceRequest):
    """
    Relation inference: Given (head, ?, tail), predict relations.

    **Use case:** "How are these two memories connected?"
    """
    engine = get_engine()

    predictions = engine.infer_relations(
        head=request.head,
        tail=request.tail,
        top_k=request.top_k
    )

    return [TripleResponse(**p.to_dict()) for p in predictions]


@app.post("/score", response_model=Dict[str, float])
async def score_triple(request: ScoreTripleRequest):
    """
    Score a triple for plausibility.

    Returns score in [0, 1] where higher = more plausible.

    **Use case:** "Is this memory connection valid?"
    """
    engine = get_engine()

    score = engine.score_triple(
        head=request.head,
        relation=request.relation,
        tail=request.tail
    )

    return {"score": score}


@app.post("/path", response_model=List[TripleResponse])
async def find_path(request: FindPathRequest):
    """
    Find reasoning path between two entities.

    Multi-hop reasoning to discover how entities connect.

    **Use case:** "How did this event lead to this emotion?"
    """
    engine = get_engine()

    path = engine.find_reasoning_path(
        start=request.start,
        end=request.end,
        max_hops=request.max_hops
    )

    return [TripleResponse(**p.to_dict()) for p in path]


@app.post("/add_triple")
async def add_triple(head: str, relation: str, tail: str):
    """Add a single triple to the knowledge graph."""
    engine = get_engine()
    engine.kg.add_triple(head, relation, tail)
    return {"status": "added", "graph": engine.kg.stats}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
        log_level="info"
    )
