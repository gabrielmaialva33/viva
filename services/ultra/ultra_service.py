#!/usr/bin/env python3
"""
ULTRA Service - Knowledge Graph Reasoning with EWC Memory Protection

Handles:
- Knowledge Graph link prediction (ULTRA arXiv:2310.04562)
- CogGNN emotional reasoning
- EWC memory protection (Kirkpatrick et al. 2017)
"""
import sys
import json
import logging
import traceback
import numpy as np
from ultra_engine import get_engine, Triple
from ewc_memory import get_ewc_manager
from dora_finetuning import get_finetuner, create_sample, is_available as dora_available

# Configure logging to stderr so it doesn't mess up stdout (used for communication)
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='[ULTRA Service] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting ULTRA Stdio Service...")
    engine = get_engine()

    # Attempt to load checkpoint on startup
    if engine.load_checkpoint():
        logger.info("ULTRA checkpoint loaded.")
    else:
        logger.warning("Running in mock mode (no checkpoint found).")

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            command = request.get("command")
            args = request.get("args", {})
            req_id = request.get("id", None)

            response_data = None

            if command == "ping":
                response_data = {"status": "pong", "loaded": engine.is_loaded}

            elif command == "build_graph":
                memories_raw = args.get("memories", [])
                # Convert raw dicts to engine format expectations if needed,
                # but build_graph_from_memories takes List[Dict] usually.
                kg = engine.build_graph_from_memories(memories_raw)
                response_data = {"stats": kg.stats}

            elif command == "predict_links":
                # args: head, relation, top_k
                head = args.get("head")
                relation = args.get("relation")
                top_k = args.get("top_k", 10)

                preds = engine.predict_links(head, relation, top_k)
                response_data = {"triples": [p.to_dict() for p in preds]}

            elif command == "infer_relations":
                # args: head, tail, top_k
                head = args.get("head")
                tail = args.get("tail")
                top_k = args.get("top_k", 5)

                preds = engine.infer_relations(head, tail, top_k)
                response_data = {"triples": [p.to_dict() for p in preds]}

            elif command == "score_triple":
                head = args.get("head")
                relation = args.get("relation")
                tail = args.get("tail")

                score = engine.score_triple(head, relation, tail)
                response_data = {"score": score}

            elif command == "find_path":
                start = args.get("start")
                end = args.get("end")
                max_hops = args.get("max_hops", 3)

                path = engine.find_reasoning_path(start, end, max_hops)
                response_data = {"path": [p.to_dict() for p in path]}

            elif command == "embed":
                text = args.get("text")
                embedding = engine.get_embedding(text)
                response_data = {"embedding": embedding}

            # =========================================================
            # CogGNN Commands (Cognitive Graph Neural Network)
            # =========================================================

            elif command == "init_cog_gnn":
                in_dim = args.get("in_dim", 384)
                hidden_dim = args.get("hidden_dim", 64)
                success = engine.init_cog_gnn(in_dim, hidden_dim)
                response_data = {"success": success}

            elif command == "propagate":
                # args: concept, pad (list of 3 floats: [pleasure, arousal, dominance])
                concept = args.get("concept")
                pad = args.get("pad", [0.0, 0.0, 0.0])
                top_k = args.get("top_k", 5)

                result = engine.propagate_thought(concept, pad, top_k)
                response_data = result

            elif command == "conscious_focus":
                focus = engine.conscious_focus()
                response_data = {"focus": focus}

            elif command == "propagate_query":
                # Query-conditioned propagation
                query = args.get("query")
                pad = args.get("pad", [0.0, 0.0, 0.0])
                top_k = args.get("top_k", 5)

                result = engine.propagate_with_query(query, pad, top_k)
                response_data = result

            # =========================================================
            # EWC Commands (Elastic Weight Consolidation)
            # =========================================================

            elif command == "protect_memory":
                # Protect a consolidated memory with EWC
                # args: memory_id, embedding, related_embeddings, consolidation_score
                memory_id = args.get("memory_id")
                embedding = np.array(args.get("embedding"), dtype=np.float32)
                related_embeddings = np.array(
                    args.get("related_embeddings", []),
                    dtype=np.float32
                )
                consolidation_score = args.get("consolidation_score", 0.0)

                ewc_manager = get_ewc_manager()
                result = ewc_manager.protect_memory(
                    memory_id=memory_id,
                    embedding=embedding,
                    related_embeddings=related_embeddings,
                    consolidation_score=consolidation_score
                )
                response_data = result

            elif command == "ewc_penalty":
                # Compute EWC penalty for a new embedding
                # args: embedding, affected_memory_ids (optional)
                embedding = np.array(args.get("embedding"), dtype=np.float32)
                affected_ids = args.get("affected_memory_ids")

                ewc_manager = get_ewc_manager()
                penalty, details = ewc_manager.compute_ewc_penalty(
                    new_embedding=embedding,
                    affected_memory_ids=affected_ids
                )
                response_data = {
                    "penalty": penalty,
                    "details": details
                }

            elif command == "ewc_stats":
                # Get EWC manager statistics
                ewc_manager = get_ewc_manager()
                response_data = ewc_manager.get_stats()

            elif command == "ewc_decay":
                # Apply Fisher decay (called periodically)
                ewc_manager = get_ewc_manager()
                ewc_manager.decay_fisher_info()
                response_data = {"status": "decay_applied"}

            elif command == "ewc_load":
                # Load protection from Qdrant payload
                # args: memory_id, payload
                memory_id = args.get("memory_id")
                payload = args.get("payload", {})

                ewc_manager = get_ewc_manager()
                success = ewc_manager.load_from_qdrant_payload(memory_id, payload)
                response_data = {"loaded": success}

            elif command == "ewc_protection_status":
                # Get protection status for a specific memory
                memory_id = args.get("memory_id")
                ewc_manager = get_ewc_manager()
                status = ewc_manager.get_protection_status(memory_id)
                response_data = status or {"protected": False}

            # =========================================================
            # Mamba Commands (Temporal Memory Processing)
            # =========================================================

            elif command == "init_mamba":
                # Initialize Mamba processor
                d_model = args.get("d_model", 384)
                n_layers = args.get("n_layers", 2)
                output_dim = args.get("output_dim", 60)
                success = engine.init_mamba(d_model, n_layers, output_dim)
                response_data = {"success": success}

            elif command == "process_sequence":
                # Process memory sequence through Mamba
                # args: embeddings, timestamps (optional)
                embeddings = args.get("embeddings", [])
                timestamps = args.get("timestamps")
                result = engine.process_memory_sequence(embeddings, timestamps)
                response_data = result

            elif command == "mamba_stats":
                # Get Mamba processor statistics
                response_data = engine.mamba_stats()

            # =========================================================
            # DoRA Commands (Fine-tuning Embeddings)
            # =========================================================

            elif command == "dora_available":
                # Check if DoRA is available
                response_data = {"available": dora_available()}

            elif command == "dora_setup":
                # Setup DoRA fine-tuner
                finetuner = get_finetuner()
                success = finetuner.setup_model()
                response_data = {"success": success}

            elif command == "dora_train":
                # Train with emotional samples
                # args: samples (list of {text, pad, label?})
                samples_data = args.get("samples", [])
                samples = [
                    create_sample(
                        text=s.get("text", ""),
                        pad=s.get("pad", [0.0, 0.0, 0.0]),
                        label=s.get("label")
                    )
                    for s in samples_data
                ]

                finetuner = get_finetuner()
                result = finetuner.train(samples)
                response_data = result

            elif command == "dora_encode":
                # Encode texts with fine-tuned model
                texts = args.get("texts", [])
                finetuner = get_finetuner()
                embeddings = finetuner.encode(texts)
                response_data = {"embeddings": embeddings}

            elif command == "dora_save":
                # Save DoRA weights
                path = args.get("path")
                finetuner = get_finetuner()
                success = finetuner.save(path)
                response_data = {"success": success}

            elif command == "dora_load":
                # Load DoRA weights
                path = args.get("path")
                finetuner = get_finetuner()
                success = finetuner.load(path)
                response_data = {"success": success}

            elif command == "dora_stats":
                # Get DoRA statistics
                finetuner = get_finetuner()
                response_data = finetuner.get_stats()

            else:
                response_data = {"error": f"Unknown command: {command}"}

            # Send response
            response = {"id": req_id, "result": response_data}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            sys.stdout.write(json.dumps({"error": "Invalid JSON"}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error processing {command}: {e}")
            sys.stdout.write(json.dumps({"id": req_id if 'req_id' in locals() else None, "error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
