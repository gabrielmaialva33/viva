#!/usr/bin/env python3
import sys
import json
import logging
import traceback
from ultra_engine import get_engine, Triple

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
