#!/usr/bin/env python3
"""
VIVA See - Vision via NVIDIA NIM API
Uses Llama 3.2 Vision or Phi-3.5 Vision

Usage:
    viva_see.py image.jpg              # Describe image
    viva_see.py image.jpg "prompt"     # Custom prompt
"""

import sys
import base64
import os
import requests
from pathlib import Path

# NVIDIA API
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "microsoft/phi-3.5-vision-instruct"  # or meta/llama-3.2-11b-vision-instruct

def get_api_key():
    """Get NVIDIA API key from environment."""
    for i in range(1, 6):
        key = os.environ.get(f"NVIDIA_API_KEY_{i}")
        if key:
            return key
    return os.environ.get("NVIDIA_API_KEY")


def encode_image(path: str) -> str:
    """Encode image to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def see(image_path: str, prompt: str = "Describe what you see in this image in detail.") -> str:
    """Analyze image using NVIDIA VLM."""
    api_key = get_api_key()
    if not api_key:
        return "Error: No NVIDIA API key found"

    image_b64 = encode_image(image_path)

    # Detect image type
    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: viva_see.py image.jpg [prompt]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe what you see in this image in detail."

    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"Analyzing {image_path}...")
    result = see(image_path, prompt)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
