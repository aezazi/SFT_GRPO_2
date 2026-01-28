
#%%
import os
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

#%%
from SFT_GRPO_v2.utils import TruncatingCollator, format_tokenize_with_spans




# %%

# %%
 #load customized tokenizer and verify special tokens and chat template

tokenizer = AutoTokenizer.from_pretrained(
    "mistral/mistral_7B_customized_tokenizer_v2",
    local_files_only=True
)


print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.pad_token)

#verify pad token is set eos token
print(f'pad token set to eos token: {tokenizer.pad_token == tokenizer.eos_token}')
print(f'pad token id set to eos token id: {tokenizer.pad_token_id == tokenizer.eos_token_id}')

# check if the model tokenizer already has a chat template
tokenizer.chat_template
# %%
print(tok.all_special_ids)
print(tok.all_special_tokens)
print(tok.pad_token)
# %%
print(type(tok))
# %%
