import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: 'huggingface_hub' not found.")
    print("Please run: pip install huggingface_hub")
    exit(1)

REPO_ID = "LiquidAI/LFM2.5-Audio-1.5B-GGUF"
DEST_DIR = Path("models")
FILES = [
    "LFM2.5-Audio-1.5B-Q4_0.gguf",
    "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf",
    "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf",
    "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf",
]

def download_models():
    DEST_DIR.mkdir(exist_ok=True)
    print(f"Downloading models from {REPO_ID} to {DEST_DIR.absolute()}...")
    
    for filename in FILES:
        if (DEST_DIR / filename).exists():
            print(f" - {filename} already exists, skipping.")
            continue
            
        print(f" - Downloading {filename}...")
        try:
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=DEST_DIR)
            print(f"   Success.")
        except Exception as e:
            print(f"   Failed to download {filename}: {e}")
            return False
            
    print("\nAll models downloaded successfully.")
    return True

if __name__ == "__main__":
    download_models()
