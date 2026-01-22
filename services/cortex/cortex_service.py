#!/usr/bin/env python3
import sys
import json
import logging
import numpy as np
from liquid_engine import get_engine

# Configure logging to stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='[Cortex Service] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting VIVA Liquid Cortex Service...")
    engine = get_engine()

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
                response_data = {"status": "pong", "type": "liquid_ncp"}

            elif command == "tick":
                # args: pad=[float, float, float], energy=float, context=[float...]
                pad = args.get("pad", [0.0, 0.0, 0.0])
                energy = args.get("energy", 0.5)
                context = args.get("context", [])

                # Convert to numpy
                pad_np = np.array(pad, dtype=np.float32)
                context_np = np.array(context, dtype=np.float32)

                new_pad = engine.process_stimulus(pad_np, energy, context_np)

                # Convert back to list for JSON
                response_data = {"pad": new_pad.tolist()}

            elif command == "reset":
                engine.model.reset_state()
                response_data = {"status": "reset"}

            else:
                response_data = {"error": f"Unknown command: {command}"}

            # Send response
            response = {"id": req_id, "result": response_data}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"Error processing {command}: {e}")
            sys.stdout.write(json.dumps({"id": req_id if 'req_id' in locals() else None, "error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
