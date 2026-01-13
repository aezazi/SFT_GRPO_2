

#%%
#imports
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import os
from dotenv import load_dotenv
import multiprocessing
from pathlib import Path

from sft_grpo.config import PACKAGE_ROOT, HF_TOKEN
from sft_grpo.sft.mistral.mistral_config import MISTRAL_SFT_ROOT, MODEL_NAME
from sft_grpo.sft.mistral.mistral_sft_utils import format_tokenize_with_spans

#%%
# Load environment variables
print(HF_TOKEN)

if torch.cuda.is_available():
    print('cuda available')

#%%
# Load BASE model

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
)

#%%
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# Check if pad token exists
if tokenizer.pad_token is None:
    print("⚠️  No pad token found, using EOS as pad")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = 'right'

#%%
# load huggingface ultrachat dataset
dataset_ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k")
train_data = dataset_ultrachat['train_sft']
test_data = dataset_ultrachat['test_sft']

# %%
# check if the model tokenizer already has a chat template
if tokenizer.chat_template is None:
    print('no chat template')
else:
    print('model already has a chat template')

#%%
# add special tokens to the tokenizer to define roles in the converstations.
# the tokenizer will not breakup a special token into parts. so for exmaple <|system|> is never broken up into '<', 'system', '>' so that its semantic meaning as a whole is preserved during sft training
ROLE_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

num_added = tokenizer.add_special_tokens(
    {"additional_special_tokens": ROLE_TOKENS}
)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

#%%
# A custom chat template that injects a bos token at the begining of each conversation, injects an eos token at the end of every turn, injects a system role and message role at begining of each conversation if none is present

aae_chat_temp_3 = """{% if messages[0]['role'] != 'system' %}{% set default_msg = 'You are a helpful assistant.' %}{% if default_system_message %}{% set default_msg = default_system_message %}{% endif %}{% set messages = [{'role': 'system', 'content': default_msg}] + messages %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}</s>{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}</s>{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}</s>{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"""

# set the tokenizer's chat template
tokenizer.chat_template = aae_chat_temp_3
print('set tokenizer chat template')

#%%
# THIS IS CRITICAL. ALWAYS RESIZE THE MODEL TO ACCOMMODATE THE EXTRA SPECIAL TOKENS
model.resize_token_embeddings(len(tokenizer))

#%%
# tokenize train data
from functools import partial
num_cores = multiprocessing.cpu_count()

processed_train = dataset_ultrachat["train_sft"].map(
    partial(format_tokenize_with_spans, tokenizer=tokenizer),
    remove_columns=dataset_ultrachat["train_sft"].column_names,
    num_proc=num_cores-2,
    desc="Preprocessing UltraChat train",
)

#%%
# tokenize test data
processed_eval = dataset_ultrachat["test_sft"].map(
    partial(format_tokenize_with_spans, tokenizer=tokenizer),
    remove_columns=dataset_ultrachat["test_sft"].column_names,
    num_proc=8,
    desc="Preprocessing UltraChat eval",
)


#%%
# print(type(str(MISTRAL_SFT_ROOT)))
# path_test = MISTRAL_SFT_ROOT / 'dataset_ultrachat/train_dataset_tokenized_v2'
# print(type(path_test))
# print(path_test)

from sft_grpo.sft.mistral import mistral_config as m_cfg
print(m_cfg.DATASET_DIR / "train_dataset_tokenized_v2")
#%%
from sft_grpo.sft.mistral import mistral_config as m_cfg
print(m_cfg.DATASET_DIR)


# save the unformatted and untokenized datasets
train_data.save_to_disk(m_cfg.DATASET_DIR / "train_dataset_orig")
test_data.save_to_disk(m_cfg.DATASET_DIR / "test_datase_orig")

# save the formatted and tokenized datasets
processed_train.save_to_disk(m_cfg.DATASET_DIR / "train_dataset_tokenized_v2")
processed_eval.save_to_disk(m_cfg.DATASET_DIR / "test_dataset_tokenized_v2")

# save the custom tokenizer
tokenizer.save_pretrained(m_cfg.MISTRAL_SFT_ROOT/ "mistral_7B_customized_tokenizer_v2")

# %%
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

ex = processed_train[1]
print(ex.keys())
print(ex['input_ids'])
print(ex['labels'])

#%%
print(tokenizer.decode(ex["input_ids"]))
print(sum(l != -100 for l in ex["labels"]), "supervised tokens")
# %%
# %%
# ============================ STATISTICS ==============================

# Get some statistics about token lengths
if 'input_ids' in train_data_tokenized.column_names:
    token_lengths = [len(ids) for ids in train_data_tokenized["input_ids"]]
    print(f"\nToken length statistics:")
    print(f"  Min: {min(token_lengths)}")
    print(f"  Max: {max(token_lengths)}")
    print(f"  Mean: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"  Median: {sorted(token_lengths)[len(token_lengths)//2]}")
    
    # Count examples that exceed common context lengths
    for threshold in [512, 1024, 2048, 4096]:
        count = sum(1 for length in token_lengths if length > threshold)
        print(f"  Examples > {threshold} tokens: {count} ({count/len(token_lengths)*100:.1f}%)")
