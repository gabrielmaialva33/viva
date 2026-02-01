#!/usr/bin/env python3
"""
VIVA Distillation with Ollama (no dependencies)

Uses Ollama's embedding models as teachers for knowledge distillation.
"""

import argparse
import json
import urllib.request
import random
import math
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"

def get_embedding(prompt: str, model: str = "nomic-embed-text") -> list[float]:
    """Get embedding from Ollama."""
    data = json.dumps({"model": model, "prompt": prompt}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["embedding"]

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def compute_cka(X: list[float], Y: list[float]) -> float:
    """Compute Centered Kernel Alignment between two vectors."""
    # Center the vectors
    mean_x = mean(X)
    mean_y = mean(Y)
    X_centered = [x - mean_x for x in X]
    Y_centered = [y - mean_y for y in Y]

    # Compute CKA
    numerator = dot(X_centered, Y_centered) ** 2
    denominator = dot(X_centered, X_centered) * dot(Y_centered, Y_centered)

    if denominator == 0:
        return 0.0
    return numerator / denominator

def progressive_schedule(epoch: int, max_epochs: int) -> tuple[float, float]:
    """Compute progressive distillation parameters."""
    progress = epoch / max_epochs
    alpha_kd = 0.3 + 0.5 * progress
    temperature = 4.0 - 2.5 * progress
    return alpha_kd, temperature

def random_vec(dim: int) -> list[float]:
    """Generate random unit vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x*x for x in v))
    return [x / norm for x in v]

def main():
    parser = argparse.ArgumentParser(description="VIVA Distillation with Ollama")
    parser.add_argument("prompt", nargs="?", help="Single prompt to embed")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama model")
    parser.add_argument("--test", action="store_true", help="Run distillation test")
    args = parser.parse_args()

    if args.test:
        print("=== VIVA Distillation Test (Ollama) ===\n")

        # 1. Get embedding
        print("1. Getting embedding from Ollama...")
        prompt = "What is consciousness?"
        embedding = get_embedding(prompt, args.model)
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5: {[f'{x:.4f}' for x in embedding[:5]]}\n")

        # 2. CKA test
        print("2. CKA Similarity Tests...")
        cka_self = compute_cka(embedding, embedding)
        print(f"   CKA(self, self): {cka_self:.6f}")

        random_v = random_vec(len(embedding))
        cka_random = compute_cka(embedding, random_v)
        print(f"   CKA(teacher, random): {cka_random:.6f}")
        print(f"   CKA Loss: {1 - cka_random:.6f}\n")

        # 3. Progressive schedule
        print("3. Progressive Distillation Schedule...")
        for epoch in [0, 5, 10]:
            alpha, temp = progressive_schedule(epoch, 10)
            print(f"   Epoch {epoch}: alpha_kd={alpha:.2f} temp={temp:.2f}")
        print()

        # 4. Multi-prompt test
        print("4. Multi-prompt Comparison...")
        prompts = [
            "What is consciousness?",
            "What is awareness?",  # Similar
            "How do I bake a cake?",  # Different
        ]
        embeddings = [get_embedding(p, args.model) for p in prompts]

        print(f"   Similar concepts:")
        cka_similar = compute_cka(embeddings[0], embeddings[1])
        print(f"   '{prompts[0][:30]}...' vs '{prompts[1][:30]}...'")
        print(f"   CKA: {cka_similar:.6f}")

        print(f"\n   Different concepts:")
        cka_diff = compute_cka(embeddings[0], embeddings[2])
        print(f"   '{prompts[0][:30]}...' vs '{prompts[2][:30]}...'")
        print(f"   CKA: {cka_diff:.6f}")
        print()

        # 5. Show that similar concepts have higher CKA
        print("5. Validation...")
        if cka_similar > cka_diff:
            print("   ✅ Similar concepts have higher CKA than different ones")
        else:
            print("   ❌ Unexpected: different concepts have higher CKA")
        print()

        print("=== Test Complete ===")
        return

    if args.prompt:
        embedding = get_embedding(args.prompt, args.model)
        result = {
            "prompt": args.prompt,
            "model": args.model,
            "embedding": embedding,
            "dimension": len(embedding)
        }
        print(json.dumps(result))

if __name__ == "__main__":
    main()
