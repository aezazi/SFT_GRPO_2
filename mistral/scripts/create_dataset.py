#%%
#imports
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import os
from dotenv import load_dotenv
import multiprocessing
from pathlib import Path

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
aae_chat_temp =  """{% if bos_token %}{{ bos_token }}{% endif %}{% if messages[0]['role'] != 'system' %}{% set default_msg = 'You are a helpful assistant.' %}{% if default_system_message %}{% set default_msg = default_system_message %}{% endif %}{% set messages = [{'role': 'system', 'content': default_msg}] + messages %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}</s>{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}</s>{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}</s>{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"""

# set the tokenizer's chat template
tokenizer.chat_template = aae_chat_temp_2
print('set tokenizer chat template')


# %%
# code below is a helper function to find begining and end of last assistant respnse

def find_last_assistant_span(
    input_ids,
    tokenizer,
    max_eos_scan_back: int = 20,
):
    """
    Find the token span (start_idx, end_idx) of the LAST assistant message.

    The span:
      - starts immediately AFTER the final <|assistant|> token
      - ends AT the corresponding EOS token
      - ignores trailing whitespace/newline tokens after EOS

    Returns:
      (start_idx, end_idx) inclusive indices

    Raises:
      ValueError if assistant token or EOS cannot be found
    """
    # Get token IDs from tokenizer
    assistant_token_id = tokenizer.convert_tokens_to_ids('<|assistant|>')
    eos_token_id = tokenizer.eos_token_id

    # Find the last <|assistant|> token
    try:
        assistant_start = len(input_ids) - 1 - input_ids[::-1].index(assistant_token_id)
    except ValueError:
        raise ValueError("No <|assistant|> token found in input_ids")

    content_start = assistant_start + 1

    # 2. Scan backwards to find EOS (ignore trailing whitespace/newlines)
    eos_pos = None
    scan_limit = max(content_start, len(input_ids) - max_eos_scan_back)

    for i in range(len(input_ids) - 1, scan_limit - 1, -1):
        if input_ids[i] == eos_token_id:
            eos_pos = i
            break

    if eos_pos is None:
        raise ValueError("No EOS token found for last assistant message")

    if eos_pos < content_start:
        raise ValueError("EOS found before assistant content starts")

    return content_start, eos_pos

#%%
# function to format and tokenize data. get the span of last assistant message
# custom_system_msg = 'you are an online therapy chatbot'

def format_tokenize_with_spans(example):
    formatted_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        default_system_message='you are a helpful assistant',
    )

    tokenized = tokenizer(
        formatted_text,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    input_ids = tokenized["input_ids"]

    # if find_last_assistant_span() throws an error, mark the example as valid=false. The collator will then skip false examples
    try:
        assistant_start, assistant_end = find_last_assistant_span(
            input_ids, tokenizer
        )
        valid = True
    except ValueError:
        assistant_start = -1
        assistant_end = -1
        valid = False

    return {
        "formatted_text":formatted_text,
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,
        "valid": valid,
    }


#%%
# tokenize train data
num_cores = multiprocessing.cpu_count()
train_data_tokenized = train_data.map(
    format_tokenize_with_spans,
    desc="Tokenizing conversations",
    remove_columns=train_data.column_names,  # Remove original columns so object has three columns: input_ids, attention_mask, and formatted_text 
    num_proc=num_cores-2
)

#%%
# tokenize test data
test_data_tokenized = test_data.map(
    format_tokenize_with_spans,
    desc="Tokenizing conversations",
    remove_columns=test_data.column_names,  # Remove original columns so object has three columns: input_ids, attention_mask, and formatted_text 
    num_proc=num_cores-2
)

#%%
# verify current working directory is correct
from pathlib import Path

# make sure the current working parent directory is where this file is actually located. sometimes when opening vscode in interactve mode, vs code sets the working directory to what ever file it opens first. I have hard coded the path to to "SFT_GRPO" in .env file. here I just append "sft_mistral_7B" to that path to get the parent directory of this file and then set the working directory to that path. Note that "project_dir_path" in the .env file has to be manually coded depending on the platform (local, remote gpu, etc.)

parent_dir_path = os.getenv("project_dir_path") + "sft_mistral_7B"
print(parent_dir_path)

print("Current working directory:", os.getcwd())
os.chdir(parent_dir_path)
print("Current working directory:", os.getcwd())


#%%
# save the unformatted and untokenized datasets
train_data.save_to_disk("./dataset_ultrachat/train_dataset_orig")
test_data.save_to_disk("./dataset_ultrachat/test_datase_orig")

# save the formatted and tokenized datasets
train_data_tokenized.save_to_disk("./dataset_ultrachat/train_dataset_tokenized")
test_data_tokenized.save_to_disk("./dataset_ultrachat/test_dataset_tokenized")

# save the custom tokenizer
tokenizer.save_pretrained("./mistral_7B_customized tokenizer")



# %%
