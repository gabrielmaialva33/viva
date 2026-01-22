
import sys
import json
import torch
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline

def log(msg):
    # Send logs to stderr so they don't break the JSON stdout stream
    sys.stderr.write(f"[Python][Chronos] {msg}\n")
    sys.stderr.flush()

def main():
    log("Initializing Chronos Service...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    try:
        # Load Model (Persistent)
        model_name = "amazon/chronos-t5-small"
        pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        log("Model loaded successfully.")

        # Main Loop: Read Line -> Predict -> Write JSON
        while True:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line)
                # request format: {"history": [v1, v2, ...], "metric": "cpu_load"}

                history = request.get("history", [])
                metric = request.get("metric", "unknown")

                if len(history) < 10:
                    # Need at least some context
                    response = {"error": "Not enough history"}
                else:
                    # Prepare Data
                    # Create a dummy timestamp index for Chronos
                    dates = pd.date_range(end="2024-01-01", periods=len(history), freq="min")

                    context_df = pd.DataFrame({
                        "Timestamp": dates,
                        "Value": history,
                        "item_id": metric
                    })

                    # Predict next 1 step (we want immediate surprise)
                    pred_df = pipeline.predict_df(
                        context_df,
                        prediction_length=1,
                        quantile_levels=[0.1, 0.5, 0.9],
                        id_column="item_id",
                        timestamp_column="Timestamp",
                        target="Value"
                    )

                    # Extract median prediction (0.5)
                    prediction = float(pred_df["0.5"].iloc[0])
                    p10 = float(pred_df["0.1"].iloc[0])
                    p90 = float(pred_df["0.9"].iloc[0])

                    response = {
                        "metric": metric,
                        "prediction": prediction,
                        "range": [p10, p90]
                    }

                # Write Output
                print(json.dumps(response))
                sys.stdout.flush()

            except json.JSONDecodeError:
                log("Failed to decode JSON input")
            except Exception as e:
                log(f"Error during prediction: {e}")
                print(json.dumps({"error": str(e)}))
                sys.stdout.flush()

    except Exception as e:
        log(f"Fatal Startup Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
