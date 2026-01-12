"""Configuration for Mistral 7B experiments - Paths and essentials only."""
from pathlib import Path
import os
from dotenv import load_dotenv
# Import the token and project root from your global package config
from sft_grpo.config import HF_TOKEN, PROJECT_ROOT


# Absolute path to .../src/sft_grpo/sft/mistral/
MISTRAL_SFT_ROOT = Path(__file__).parent.resolve()

# Mistral-specific directories
CHECKPOINTS_DIR = MISTRAL_SFT_ROOT / "checkpoints"
LOGS_DIR = MISTRAL_SFT_ROOT / "logs"
EXPERIMENTS_DIR = MISTRAL_SFT_ROOT / "experiments"



# Tokenizer paths
CUSTOM_TOKENIZER_V1_PATH = MISTRAL_SFT_ROOT / "mistral_7B_customized_tokenizer"
CUSTOM_TOKENIZER_V2_PATH = MISTRAL_SFT_ROOT / "mistral_7B_customized_tokenizer_v2"

# Dataset path
DATASET_DIR = MISTRAL_SFT_ROOT / "dataset_ultrachat"

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN = HF_TOKEN

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [CHECKPOINTS_DIR, LOGS_DIR, EXPERIMENTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_tokenizer_path(version: int = 2) -> Path:
    """Get the appropriate tokenizer path based on version."""
    if version == 1:
        return CUSTOM_TOKENIZER_V1_PATH
    elif version == 2:
        return CUSTOM_TOKENIZER_V2_PATH
    else:
        raise ValueError(f"Unknown tokenizer version: {version}")

if __name__ == "__main__":
    # Test/debug the configuration
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Mistral Root: {MISTRAL_SFT_ROOT}")
    print(f"Tokenizer V2: {CUSTOM_TOKENIZER_V2_PATH}")
    print(f"Tokenizer V2 Exists: {CUSTOM_TOKENIZER_V2_PATH.exists()}")
    print(f"Dataset Path: {DATASET_DIR}")
    print(f"Dataset Exists: {DATASET_DIR.exists()}")
    print(HF_TOKEN)