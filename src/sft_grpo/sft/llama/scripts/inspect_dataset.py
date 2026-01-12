

#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk, load_dataset
import torch
from dotenv import load_dotenv
import os

#%%
# Load environment variables
load_dotenv()
hf_token = os.getenv("hf_access_token")


# For 3B, use LLaMA 3.2 base:
model_name = "meta-llama/Llama-3.2-3B"  # This is the base model

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
# model.resize_token_embeddings(len(tokenizer))
# tokenizer.padding_side = 'right'

#%%
#Load dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")


# %%
print(dataset.column_names)
print(len(dataset['train']))

# %%
# Format for SFT (you'll need instruction and response columns)
def format_for_base_model(example):
    # Base models haven't learned instruction following yet
    # So we format more simply - just concatenate with clear separators
    text = f"### User:\n{example['instruction']}\n\n### Assistant:\n{example['response']}<|end_of_text|>"
    return {"text": text}

formatted_dataset = dataset.map(format_for_base_model, remove_columns=dataset["train"].column_names)
# %%
print(type(formatted_dataset))
print(formatted_dataset.column_names)
print(type(formatted_dataset['train'][0]))
formatted_dataset['train'][1]
# %%
