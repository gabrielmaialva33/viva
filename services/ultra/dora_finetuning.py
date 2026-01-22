"""
DoRA Fine-Tuning for VIVA Embeddings

Implements Weight-Decomposed Low-Rank Adaptation (DoRA) for fine-tuning
MiniLM embedding model on VIVA's emotional semantic space.

DoRA = LoRA + Weight Decomposition:
- Decomposes weights into magnitude and direction components
- More stable training than vanilla LoRA
- Better preservation of pre-trained features

Use Case:
- Adapt MiniLM embeddings to VIVA's emotional vocabulary
- Contrastive learning: similar emotions → similar embeddings
- ~9% trainable parameters (2M / 22M)

Reference:
- DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)
- LoRA: Low-Rank Adaptation of LLMs (Hu et al., 2021)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Check for peft availability
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft not installed. DoRA fine-tuning unavailable.")
    logger.warning("Install with: pip install peft>=0.10.0")

# Check for transformers
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed")


@dataclass
class DoRAConfig:
    """Configuration for DoRA fine-tuning."""
    # Model
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LoRA config
    r: int = 8                  # LoRA rank
    lora_alpha: int = 16        # LoRA alpha (scaling factor)
    lora_dropout: float = 0.1   # Dropout for LoRA layers
    use_dora: bool = True       # Enable weight decomposition
    target_modules: List[str] = None  # Target modules for LoRA

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 32
    epochs: int = 3
    warmup_steps: int = 100

    # Loss
    temperature: float = 0.07   # InfoNCE temperature
    margin: float = 0.5         # Margin for contrastive loss

    # Paths
    save_path: str = "./dora_checkpoints"
    device: str = "cpu"

    def __post_init__(self):
        if self.target_modules is None:
            # Default: attention layers
            self.target_modules = ["query", "key", "value"]


@dataclass
class EmotionalSample:
    """A training sample with emotional context."""
    text: str
    pad: List[float]  # [pleasure, arousal, dominance]
    label: Optional[str] = None  # Optional categorical label


class EmotionalDataset(Dataset):
    """Dataset for emotional contrastive learning."""

    def __init__(self, samples: List[EmotionalSample], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize text
        encoded = self.tokenizer(
            sample.text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'pad': torch.tensor(sample.pad, dtype=torch.float32)
        }


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with PAD-based similarity weighting.

    Pulls together embeddings of emotionally similar texts,
    pushes apart embeddings of emotionally different texts.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        pad_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            embeddings: [batch, dim] normalized embeddings
            pad_vectors: [batch, 3] PAD emotional states

        Returns:
            Loss scalar
        """
        batch_size = embeddings.size(0)

        # Compute embedding similarity matrix
        sim_matrix = embeddings @ embeddings.T  # [batch, batch]
        sim_matrix = sim_matrix / self.temperature

        # Compute PAD similarity (closer PAD → positive pair)
        pad_dists = torch.cdist(pad_vectors, pad_vectors)  # Euclidean distance
        pad_sim = 1 - pad_dists / (pad_dists.max() + 1e-8)  # Normalize to [0, 1]

        # Create soft labels from PAD similarity
        # Diagonal = 1 (self), off-diagonal = PAD similarity
        labels = pad_sim

        # InfoNCE loss: cross-entropy with soft labels
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class DoRAFineTuner:
    """
    DoRA-based fine-tuning for MiniLM embeddings.

    Adapts the embedding model to VIVA's emotional semantic space
    using weight-decomposed low-rank adaptation.
    """

    def __init__(self, config: Optional[DoRAConfig] = None):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for DoRA fine-tuning. "
                "Install with: pip install peft>=0.10.0"
            )
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers"
            )

        self.config = config or DoRAConfig()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.loss_fn = InfoNCELoss(self.config.temperature)

        self._training_stats = {
            "epochs_completed": 0,
            "total_steps": 0,
            "best_loss": float('inf')
        }

        logger.info(f"DoRAFineTuner initialized: model={self.config.model_name}")

    def setup_model(self) -> bool:
        """
        Initialize model with DoRA/LoRA adapters.

        Returns:
            True if successful
        """
        try:
            # Load base model
            base_model = AutoModel.from_pretrained(self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Configure LoRA with DoRA (weight decomposition)
            peft_config = LoraConfig(
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                use_dora=self.config.use_dora,  # Enable weight decomposition
                task_type=TaskType.FEATURE_EXTRACTION
            )

            # Apply LoRA to model
            self.model = get_peft_model(base_model, peft_config)
            self.model.to(self.config.device)

            # Log trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Model setup complete: {trainable_params:,} / {total_params:,} params "
                f"({100 * trainable_params / total_params:.1f}%)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            return False

    def train(
        self,
        train_samples: List[EmotionalSample],
        val_samples: Optional[List[EmotionalSample]] = None
    ) -> Dict[str, Any]:
        """
        Train the model with contrastive learning.

        Args:
            train_samples: Training samples with text and PAD
            val_samples: Optional validation samples

        Returns:
            Training statistics
        """
        if self.model is None:
            if not self.setup_model():
                return {"error": "Model setup failed"}

        # Create datasets
        train_dataset = EmotionalDataset(train_samples, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

        # Training loop
        self.model.train()
        total_loss = 0.0
        steps = 0

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            for batch in train_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                pad_vectors = batch['pad'].to(self.config.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Mean pooling for sentence embedding
                embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                embeddings = F.normalize(embeddings, p=2, dim=1)

                # Compute loss
                loss = self.loss_fn(embeddings, pad_vectors)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                steps += 1

            avg_epoch_loss = epoch_loss / len(train_loader)
            total_loss += avg_epoch_loss

            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}: loss={avg_epoch_loss:.4f}")

            self._training_stats["epochs_completed"] = epoch + 1
            self._training_stats["total_steps"] = steps

            if avg_epoch_loss < self._training_stats["best_loss"]:
                self._training_stats["best_loss"] = avg_epoch_loss

        return {
            "epochs": self.config.epochs,
            "final_loss": total_loss / self.config.epochs,
            "best_loss": self._training_stats["best_loss"],
            "total_steps": steps
        }

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling for sentence embedding."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts using the fine-tuned model.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            logger.warning("Model not initialized")
            return []

        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for text in texts:
                encoded = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.config.device)

                outputs = self.model(**encoded)
                emb = self._mean_pooling(
                    outputs.last_hidden_state,
                    encoded['attention_mask']
                )
                emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb.squeeze().cpu().tolist())

        return embeddings

    def save(self, path: Optional[str] = None) -> bool:
        """Save the adapter weights."""
        if self.model is None:
            return False

        save_path = Path(path or self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            self.model.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load adapter weights."""
        try:
            base_model = AutoModel.from_pretrained(self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = PeftModel.from_pretrained(base_model, path)
            self.model.to(self.config.device)
            self.model.eval()
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get fine-tuning statistics."""
        return {
            "model": self.config.model_name,
            "use_dora": self.config.use_dora,
            "rank": self.config.r,
            "alpha": self.config.lora_alpha,
            "training": self._training_stats,
            "model_initialized": self.model is not None
        }


# Singleton
_finetuner: Optional[DoRAFineTuner] = None


def get_finetuner() -> DoRAFineTuner:
    """Get or create the DoRA fine-tuner singleton."""
    global _finetuner
    if _finetuner is None:
        _finetuner = DoRAFineTuner()
    return _finetuner


def is_available() -> bool:
    """Check if DoRA fine-tuning is available."""
    return PEFT_AVAILABLE and TRANSFORMERS_AVAILABLE


def create_sample(text: str, pad: List[float], label: Optional[str] = None) -> EmotionalSample:
    """Convenience function to create training samples."""
    return EmotionalSample(text=text, pad=pad, label=label)
