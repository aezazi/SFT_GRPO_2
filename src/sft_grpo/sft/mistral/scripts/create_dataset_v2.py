#%%
# verify current working directory is correct
from pathlib import Path

parent_dir_path = os.getenv("project_dir_path") + "sft_mistral_7B"
print(parent_dir_path)

print("Current working directory:", os.getcwd())
os.chdir(parent_dir_path)
print("Current working directory:", os.getcwd())

#%%
#imports
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import os
from dotenv import load_dotenv
import multiprocessing
from pathlib import Path
from utils import format_tokenize_with_spans

#%%
# Load environment variables
load_dotenv()
hf_token = os.getenv("hf_token")
print(hf_token)

if torch.cuda.is_available():
    print('cuda available')

#%%
# Load BASE model
model_name = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token
)

#%%
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

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
# save the unformatted and untokenized datasets
train_data.save_to_disk("./dataset_ultrachat/train_dataset_orig")
test_data.save_to_disk("./dataset_ultrachat/test_datase_orig")

# save the formatted and tokenized datasets
processed_train.save_to_disk("./dataset_ultrachat/train_dataset_tokenized_v2")
processed_eval.save_to_disk("./dataset_ultrachat/test_dataset_tokenized_v2")

# save the custom tokenizer
tokenizer.save_pretrained("./mistral_7B_customized tokenizer_v2")

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
