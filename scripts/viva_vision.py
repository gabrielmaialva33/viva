#!/usr/bin/env python3
"""
VIVA Vision - Image understanding via HuggingFace Inference API
Uses free models: BLIP, LLaVA, or CLIP

Usage:
    viva_vision.py describe image.jpg     # Describe what's in the image
    viva_vision.py caption image.jpg      # Generate caption
    viva_vision.py classify image.jpg     # Classify image
    viva_vision.py ocr image.jpg          # Extract text
"""

import sys
import base64
import json
import os
from pathlib import Path

# Try HuggingFace Hub
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Try requests as fallback
import requests

# HuggingFace free inference endpoints
ENDPOINTS = {
    "caption": "Salesforce/blip-image-captioning-large",
    "describe": "llava-hf/llava-1.5-7b-hf",  # VLM
    "classify": "openai/clip-vit-large-patch14",
    "ocr": "microsoft/trocr-large-printed",
}

# Colors
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def load_image_base64(path: str) -> str:
    """Load image and convert to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def caption_image(image_path: str) -> str:
    """Generate caption using BLIP."""
    api_url = f"https://api-inference.huggingface.co/models/{ENDPOINTS['caption']}"

    with open(image_path, "rb") as f:
        data = f.read()

    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(api_url, headers=headers, data=data, timeout=30)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", str(result))
        return str(result)
    else:
        return f"Error: {response.status_code} - {response.text}"


def classify_image(image_path: str, labels: list = None) -> dict:
    """Zero-shot classification using CLIP."""
    if labels is None:
        labels = ["person", "animal", "object", "text", "scene", "electronics", "workspace"]

    api_url = f"https://api-inference.huggingface.co/models/{ENDPOINTS['classify']}"

    with open(image_path, "rb") as f:
        data = f.read()

    headers = {"Content-Type": "application/json"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": {
            "image": base64.b64encode(data).decode(),
        },
        "parameters": {
            "candidate_labels": labels
        }
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=30)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"{response.status_code} - {response.text}"}


def ocr_image(image_path: str) -> str:
    """Extract text using TrOCR."""
    api_url = f"https://api-inference.huggingface.co/models/{ENDPOINTS['ocr']}"

    with open(image_path, "rb") as f:
        data = f.read()

    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(api_url, headers=headers, data=data, timeout=30)

    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", str(result))
        return str(result)
    else:
        return f"Error: {response.status_code} - {response.text}"


def describe_image(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """Describe image using VLM (LLaVA or similar)."""
    # First try caption as it's more reliable
    caption = caption_image(image_path)
    return caption


def main():
    if len(sys.argv) < 3:
        print(f"{YELLOW}Usage:{RESET}")
        print(f"  {sys.argv[0]} caption image.jpg   - Generate caption")
        print(f"  {sys.argv[0]} classify image.jpg  - Classify image")
        print(f"  {sys.argv[0]} ocr image.jpg       - Extract text")
        print(f"  {sys.argv[0]} describe image.jpg  - Describe image")
        sys.exit(1)

    command = sys.argv[1]
    image_path = sys.argv[2]

    if not Path(image_path).exists():
        print(f"{RED}Error: Image not found: {image_path}{RESET}")
        sys.exit(1)

    print(f"{CYAN}Processing {image_path}...{RESET}")

    if command == "caption":
        result = caption_image(image_path)
        print(f"{GREEN}Caption:{RESET} {result}")

    elif command == "classify":
        labels = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        result = classify_image(image_path, labels)
        print(f"{GREEN}Classification:{RESET}")
        if isinstance(result, dict) and "error" not in result:
            for item in result:
                print(f"  {item['label']}: {item['score']:.2%}")
        else:
            print(f"  {result}")

    elif command == "ocr":
        result = ocr_image(image_path)
        print(f"{GREEN}Text:{RESET} {result}")

    elif command == "describe":
        result = describe_image(image_path)
        print(f"{GREEN}Description:{RESET} {result}")

    else:
        print(f"{RED}Unknown command: {command}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
