"""Configuration for Mistral 7B experiments - Paths and essentials only."""

#%%
from pathlib import Path
import os
from dotenv import load_dotenv
# Import the token and project root from your global package config
from sft_grpo.config import HF_TOKEN, PROJECT_ROOT
from sft_grpo.sft.mistral_sft.mistral_sft_config import MISTRAL_SFT_ROOT

# print(HF_TOKEN)
# print(PROJECT_ROOT)

#%%
# Absolute path to .../src/sft_grpo/sft/mistral/
MISTRAL_GRPO_ROOT = Path(__file__).parent.resolve()

# print(MISTRAL_GRPO_ROOT)

# path to sft LoRA adapters
LORA_SFT_ADAPTER_PATH = MISTRAL_SFT_ROOT/"mistral_sft_experiments/final_sft_model_v3_1epoch_svd_r"


# Mistral-specific directories
GRPO_CHECKPOINTS_DIR = MISTRAL_GRPO_ROOT / "mistral_grpo_checkpoints"
GRPO_LOGS_DIR = MISTRAL_GRPO_ROOT / "mistral_grpo_logs"
GRPO_EXPERIMENTS_DIR = MISTRAL_GRPO_ROOT / "mistral_grpo_experiments"

#%%
# note that tokenizer and sft trained model are in the sft --> mistral_sft directory
# Tokenizer paths

CUSTOM_TOKENIZER_V2_PATH = MISTRAL_SFT_ROOT / "mistral_7B_customized_tokenizer_v2"

# Dataset path
DATASET_DIR = MISTRAL_SFT_ROOT / "dataset_ultrachat"

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
HF_TOKEN = HF_TOKEN

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [GRPO_CHECKPOINTS_DIR, GRPO_LOGS_DIR, GRPO_EXPERIMENTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test/debug the configuration
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Mistral_sft Root: {MISTRAL_SFT_ROOT}")
    print(f"Mistral_grpo Root: {MISTRAL_GRPO_ROOT}")
    print(f"Tokenizer V2: {CUSTOM_TOKENIZER_V2_PATH}")
    print(f"Tokenizer V2 Exists: {CUSTOM_TOKENIZER_V2_PATH.exists()}")
    print(f"Dataset Path: {DATASET_DIR}")
    print(f"Dataset Exists: {DATASET_DIR.exists()}")
    print(HF_TOKEN)