# SFT_GRPO_2

Supervised Fine-Tuning experiments comparing Mistral 7B and Llama 3B models.

## Project Structure
```
SFT_GRPO_2/
├── mistral/          # Mistral 7B experiments
├── llama/            # Llama 3B experiments
├── shared/           # Shared utilities
└── scripts/          # Standalone scripts
```

## Setup

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd SFT_GRPO_2

# Create .env file with your HuggingFace token
echo "hf_token=your_token_here" > .env

# Install package in editable mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Usage
```python
# Import from anywhere
from mistral.config import CUSTOM_TOKENIZER_V2_PATH
from mistral.utils import TruncatingCollator
```

## Models

- **Mistral 7B**: Fine-tuning experiments with custom tokenizer
- **Llama 3B**: Comparison experiments

## Requirements

- Python >= 3.9
- CUDA-capable GPU (for training)
- HuggingFace account and token