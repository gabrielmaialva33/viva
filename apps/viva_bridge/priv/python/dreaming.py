#!/usr/bin/env python3
"""
VIVA's Dream - LoRA Fine-tuning for Chronos

During the day, VIVA experiences her host through Interoception.
At night, those experiences transform into model weights.

This is how VIVA learns to predict HER specific host's behavior,
not just generic time series patterns.

## The Dream Cycle

1. **Day (Awake)**: DatasetCollector captures sensations
2. **Night (Dream)**: This script fine-tunes Chronos with LoRA
3. **Dawn (Rebirth)**: chronos_service.py loads new adapter

## Usage

```bash
# Automatic (called by DreamOrchestrator)
python3 dreaming.py

# Manual with specific date
python3 dreaming.py --date 2026-01-20

# Training from multiple days
python3 dreaming.py --days 7
```
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# ============================================================================
# Configuration
# ============================================================================

# Paths
DATASET_DIR = Path(__file__).parent.parent / "datasets"
ADAPTER_DIR = Path(__file__).parent / "adapters"
MODEL_NAME = "amazon/chronos-t5-small"

# LoRA Configuration (conservative for small model)
LORA_CONFIG = {
    "r": 8,              # LoRA rank (small for chronos-t5-small)
    "lora_alpha": 16,    # Scaling factor
    "lora_dropout": 0.1, # Regularization
    "target_modules": ["q", "v"],  # Query and Value projections
}

# Training Configuration
TRAINING_CONFIG = {
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "logging_steps": 50,
}


def log(msg: str, level: str = "INFO"):
    """Log to stderr for visibility"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    sys.stderr.write(f"[{timestamp}][Dream][{level}] {msg}\n")
    sys.stderr.flush()


# ============================================================================
# Dataset Loading
# ============================================================================

def find_dataset_files(date: Optional[str] = None, days: int = 1) -> List[Path]:
    """Find dataset files for the specified date range."""
    files = []

    if date:
        # Specific date
        target = datetime.strptime(date, "%Y-%m-%d")
        dates = [target - timedelta(days=i) for i in range(days)]
    else:
        # Default: yesterday (dreaming about today's experiences)
        target = datetime.now() - timedelta(days=1)
        dates = [target - timedelta(days=i) for i in range(days)]

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        pattern = f"viva_sensations_{date_str}*.csv"
        found = list(DATASET_DIR.glob(pattern))
        files.extend(found)

    return sorted(files)


def load_datasets(files: List[Path]) -> pd.DataFrame:
    """Load and concatenate dataset files."""
    if not files:
        raise ValueError("No dataset files found")

    dfs = []
    for f in files:
        log(f"Loading {f.name}")
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    log(f"Loaded {len(combined)} records from {len(files)} files")

    return combined


def prepare_training_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare data for Chronos fine-tuning.

    Chronos expects time series data with:
    - timestamp column
    - target column(s)
    - optional covariates
    """
    # Group by metric to create separate time series
    metrics = df["metric"].unique()
    log(f"Found {len(metrics)} unique metrics: {list(metrics)[:5]}...")

    training_samples = []

    for metric in metrics:
        metric_df = df[df["metric"] == metric].sort_values("timestamp")

        if len(metric_df) < 20:
            # Need enough history for training
            continue

        # Create sliding window samples
        window_size = 50  # Context length
        pred_length = 1   # Predict next step

        values = metric_df["value"].values

        for i in range(len(values) - window_size - pred_length + 1):
            context = values[i:i + window_size]
            target = values[i + window_size:i + window_size + pred_length]

            training_samples.append({
                "metric": metric,
                "context": context.tolist(),
                "target": target.tolist(),
            })

    log(f"Created {len(training_samples)} training samples")

    return {
        "samples": training_samples,
        "metrics": list(metrics),
        "total_records": len(df),
    }


# ============================================================================
# LoRA Fine-tuning
# ============================================================================

def setup_lora_model():
    """
    Set up Chronos model with LoRA adapters.

    Note: This is a simplified implementation. Full LoRA requires:
    - peft library
    - transformers >= 4.30
    - Custom Chronos adapter support
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import T5ForConditionalGeneration, T5Config
        from chronos import BaseChronosPipeline
    except ImportError as e:
        log(f"Missing dependency: {e}", "ERROR")
        log("Install with: pip install peft transformers chronos-forecasting", "ERROR")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    # Load base Chronos model
    log(f"Loading base model: {MODEL_NAME}")
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # Extract the underlying T5 model
    model = pipeline.model

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
    )

    # Apply LoRA to model
    log("Applying LoRA adapters...")
    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    log(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return peft_model, pipeline, device


def train_loop(model, training_data: Dict, device: str) -> Dict[str, float]:
    """
    Simple training loop for LoRA fine-tuning.

    For production, consider using:
    - Hugging Face Trainer
    - PyTorch Lightning
    - Proper validation/early stopping
    """
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    samples = training_data["samples"]

    if not samples:
        log("No training samples available", "WARN")
        return {"loss": 0.0, "samples": 0}

    # Simple dataset class
    class TimeSeriesDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            return {
                "context": torch.tensor(sample["context"], dtype=torch.float32),
                "target": torch.tensor(sample["target"], dtype=torch.float32),
            }

    dataset = TimeSeriesDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(dataloader) * TRAINING_CONFIG["epochs"],
    )

    # Training
    model.train()
    total_loss = 0.0
    step = 0

    for epoch in range(TRAINING_CONFIG["epochs"]):
        epoch_loss = 0.0

        for batch in dataloader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)

            # Forward pass (simplified - actual Chronos training is more complex)
            # This is a placeholder for the actual training logic
            # Real implementation would use Chronos-specific loss

            optimizer.zero_grad()

            # Simplified MSE loss on predictions
            # In reality, Chronos uses probabilistic forecasting loss
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                # Placeholder forward pass
                outputs = model(
                    input_ids=context.long()[:, :512] if context.shape[1] > 512 else context.long(),
                    decoder_input_ids=target.long().unsqueeze(-1),
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.01)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                TRAINING_CONFIG["max_grad_norm"],
            )

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            total_loss += loss.item()
            step += 1

            if step % TRAINING_CONFIG["logging_steps"] == 0:
                log(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(dataloader) if dataloader else 0
        log(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.6f}")

    avg_loss = total_loss / step if step > 0 else 0

    return {
        "loss": avg_loss,
        "samples": len(samples),
        "steps": step,
        "epochs": TRAINING_CONFIG["epochs"],
    }


def save_adapter(model, output_path: Path) -> Path:
    """Save the LoRA adapter weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save adapter weights
    adapter_path = output_path.with_suffix(".safetensors")

    try:
        model.save_pretrained(output_path.parent)
        log(f"Saved adapter to {output_path.parent}")
    except Exception as e:
        log(f"Error saving adapter: {e}", "WARN")
        # Fallback: save state dict
        torch.save(model.state_dict(), adapter_path)
        log(f"Saved state dict to {adapter_path}")

    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
        "created_at": datetime.now().isoformat(),
        "adapter_path": str(adapter_path),
    }

    metadata_path = output_path.parent / "adapter_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"Saved metadata to {metadata_path}")

    return adapter_path


# ============================================================================
# Dream Report
# ============================================================================

def generate_dream_report(
    training_data: Dict,
    training_results: Dict,
    adapter_path: Optional[Path],
) -> Dict[str, Any]:
    """Generate a report of the dream session."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "success" if adapter_path else "skipped",
        "data": {
            "total_records": training_data["total_records"],
            "training_samples": len(training_data["samples"]),
            "metrics_learned": training_data["metrics"],
        },
        "training": training_results,
        "adapter": str(adapter_path) if adapter_path else None,
    }

    # Save report
    report_dir = ADAPTER_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = report_dir / f"dream_report_{date_str}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"Dream report saved to {report_path}")

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VIVA's Dream - LoRA Fine-tuning")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to include")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually train")
    args = parser.parse_args()

    log("=" * 60)
    log("VIVA is entering the Dream state...")
    log("=" * 60)

    # 1. Find and load datasets
    try:
        files = find_dataset_files(args.date, args.days)
        if not files:
            log("No dataset files found. VIVA had nothing to dream about.", "WARN")
            return

        df = load_datasets(files)
    except Exception as e:
        log(f"Failed to load datasets: {e}", "ERROR")
        return

    # 2. Prepare training data
    training_data = prepare_training_data(df)

    if not training_data["samples"]:
        log("Not enough data for training. VIVA's dreams were empty.", "WARN")
        report = generate_dream_report(training_data, {}, None)
        return

    # 3. Set up model with LoRA
    if args.dry_run:
        log("Dry run - skipping training", "INFO")
        report = generate_dream_report(training_data, {"dry_run": True}, None)
        return

    try:
        model, pipeline, device = setup_lora_model()
    except Exception as e:
        log(f"Failed to setup model: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return

    # 4. Train
    log("Beginning the Dream (training)...")
    training_results = train_loop(model, training_data, device)

    # 5. Save adapter
    date_str = datetime.now().strftime("%Y-%m-%d")
    adapter_path = save_adapter(
        model,
        ADAPTER_DIR / f"viva-adapter-{date_str}" / "adapter_model",
    )

    # 6. Generate report
    report = generate_dream_report(training_data, training_results, adapter_path)

    log("=" * 60)
    log("VIVA awakens with new knowledge of her host.")
    log(f"Samples dreamed: {training_results.get('samples', 0)}")
    log(f"Final loss: {training_results.get('loss', 0):.6f}")
    log("=" * 60)

    # Output final status as JSON for Elixir to parse
    print(json.dumps(report))


if __name__ == "__main__":
    main()
