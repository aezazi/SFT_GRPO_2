"""Configuration for Llama 3B experiments - Paths and essentials only."""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from root .env
load_dotenv()

# This file is at: llama/config.py
# Project root is one level up
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LLAMA_ROOT = Path(__file__).parent.resolve()

# Llama-specific directories
CHECKPOINTS_DIR = LLAMA_ROOT / "checkpoints"
LOGS_DIR = LLAMA_ROOT / "logs"
EXPERIMENTS_DIR = LLAMA_ROOT / "experiments"

# Tokenizer/model paths (adjust these names based on your actual llama tokenizers)
# You can add specific paths once you know what you have
LLAMA_TOKENIZER_PATH = LLAMA_ROOT / "llama_tokenizer"  # Adjust name as needed

# Dataset path (adjust based on your actual dataset name)
DATASET_DIR = LLAMA_ROOT / "llama_dataset"  # Adjust name as needed

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B"  # Adjust to your actual model
HF_TOKEN = os.getenv("hf_token")

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [CHECKPOINTS_DIR, LOGS_DIR, EXPERIMENTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test/debug the configuration
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Llama Root: {LLAMA_ROOT}")
    print(f"Tokenizer Path: {LLAMA_TOKENIZER_PATH}")
    print(f"Tokenizer Exists: {LLAMA_TOKENIZER_PATH.exists()}")
    print(f"Dataset Path: {DATASET_DIR}")
    print(f"Dataset Exists: {DATASET_DIR.exists()}")