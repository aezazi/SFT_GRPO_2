import os
from pathlib import Path
from dotenv import load_dotenv

# Points to src/sft_grpo/
PACKAGE_ROOT = Path(__file__).parent.resolve()

# Points to the absolute Project Root (SFT_GRPO_2/)
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Explicitly load the .env from the root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("hf_token")
WANDB_API_KEY = os.getenv("WANDB_API_KEY") # If you use it later