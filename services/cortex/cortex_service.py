#!/usr/bin/env python3
"""
VIVA Cortex Service - Liquid Neural Network with Neural ODE Support

Handles PAD emotional dynamics through stdio JSON protocol.
Supports two modes:
- LTC mode (default): Liquid Time-Constant networks
- Neural ODE mode: Explicit dPAD/dt = f(PAD, sensory, t)

Set CORTEX_USE_NEURAL_ODE=true to enable Neural ODE mode.
"""
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
    logger.info(f"Running in {engine.get_mode().upper()} mode")

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
                response_data = {
                    "status": "pong",
                    "type": "liquid_ncp",
                    "mode": engine.get_mode()
                }

            elif command == "tick":
                # args: pad=[P,A,D], energy=float, context=[float...], time_delta=float
                pad = args.get("pad", [0.0, 0.0, 0.0])
                energy = args.get("energy", 0.5)
                context = args.get("context", [])
                time_delta = args.get("time_delta", 1.0)  # Biological time perception

                # Convert to numpy
                pad_np = np.array(pad, dtype=np.float32)
                context_np = np.array(context, dtype=np.float32)

                new_pad = engine.process_stimulus(pad_np, energy, context_np, time_delta)

                # Convert back to list for JSON
                response_data = {
                    "pad": new_pad.tolist(),
                    "mode": engine.get_mode()
                }

            elif command == "tick_with_trajectory":
                # Neural ODE mode: Return full trajectory for visualization
                # args: pad=[P,A,D], energy=float, context=[float...], time_delta=float
                pad = args.get("pad", [0.0, 0.0, 0.0])
                energy = args.get("energy", 0.5)
                context = args.get("context", [])
                time_delta = args.get("time_delta", 1.0)

                pad_np = np.array(pad, dtype=np.float32)
                context_np = np.array(context, dtype=np.float32)

                new_pad, trajectory = engine.process_with_trajectory(
                    pad_np, energy, context_np, time_delta
                )

                response_data = {
                    "pad": new_pad.tolist(),
                    "trajectory": trajectory,  # None if LTC mode
                    "mode": engine.get_mode()
                }

            elif command == "get_mode":
                response_data = {
                    "mode": engine.get_mode(),
                    "heatmap": engine.get_synaptic_heatmap()
                }

            elif command == "reset":
                engine.reset_state()
                response_data = {"status": "reset", "mode": engine.get_mode()}

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
