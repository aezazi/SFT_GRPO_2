#%%
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import os
from dotenv import load_dotenv
import multiprocessing

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
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Check if pad token exists
if tokenizer.pad_token is None:
    print("⚠️  No pad token found, using EOS as pad")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = 'right'

#%%
# load the huggingface ultrachat dataset
dataset_ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k")
train_data = dataset_ultrachat['train_sft']


# %%
# explore the dataset structure and its elements
import pprint


print(f'- the type train_data is: {type(train_data)}\n') # a HF Dataset object. It is an iterable that can be indexed into. 

print(f'- Dataset object info:\n  {train_data.info}\n')# all the information about the Dataset object 

# The 'features' object within the Dataset object is like a database table schema which contains column name'
print(f'- Dataset "features" object type: {type(train_data.features)}\n ')
print(f'- Dataset "features" contains information about each column, similar to a databse table schema:\n  {train_data.features}\n')

print('=============================================================\n')

# when using the .select method of the HF Dataset class, another Dataset object is returned. Dataset objects have a bevy of methods (incluling .map) which python dictionaries do not. Each row returned is one conversation. The object is like a database table with column names which are stored in 'features'

print(f'- the type of a train_data row using .select is another Dataset object:\n  {type(train_data.select(range(1)))}\n')
print(f'- the features (columns) of each row returned are:\n {train_data.select(range(1))}\n')

# each column can be accessed by the column label. the type of each column is a arrow_dataset.column
print(f'- the type of each column name is: {type(train_data.select(range(1))['messages'])}\n')

# Each column behaves like list which contains lists for each row (example)
print(f'- for each row (example), the "messages" columns contains one list which contains a list that contains dictionari: {type(train_data.select(range(1))['messages'])}\n')

# here, I am indexing into the column first and then into the list for that row to sow the keys for the dictionary.
print(f'- the list inside the column list contains dictionaries where each\n dictionary is a turn in the chat for that row (example). the keys for the\n dictionary are "role" and "content":\n {train_data.select(range(1))['messages'][0][0].keys()}\n')


# the list that holds dictionaries holding the converasation for the first row (example)
print(f'the dictionaries holding th converasation for the first row (example):\n {train_data.select(range(1))['messages'][0]}')


print('=============================================================\n')

# when using slicing, Each row is a dictionary
print(f'- the type of a train_data row using slicing is: {type(train_data[1])}') 
print(f'- the keys of each row dictionary are: {train_data[1].keys()}')
print(f"- 'messages' holds the conversation. the value of this key is a list of dictionaries.")
print(f'- each dictionary is a turn in the conversation. \n- the keys of each converstaion turn dictionary are: {train_data[1]['messages'][0].keys()}\n')


print(f'- length of train_data: {len(train_data)}')


# %%
# check if the model tokenizer already has a chat template
if tokenizer.chat_template is None:
    print('no chat template')
else:
    print('model already has a chat template')

# %%
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

# THIS IS CRITICAL. ALWAYS RESIZE THE MODEL TO ACCOMMODATE THE EXTRA SPECIAL TOKENS
model.resize_token_embeddings(len(tokenizer))

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)

#%%
# A custom chat template that injects a bos token at the begining of each conversation, injects an eos token at the end of every turn, injects a system role and message role at begining of each conversation if none is present
aae_chat_temp =  """{% if bos_token %}{{ bos_token }}{% endif %}{% if messages[0]['role'] != 'system' %}{% set default_msg = 'You are a helpful assistant.' %}{% if default_system_message %}{% set default_msg = default_system_message %}{% endif %}{% set messages = [{'role': 'system', 'content': default_msg}] + messages %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}</s>{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}</s>{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}</s>{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"""

# set the tokenizer's chat template
tokenizer.chat_template = aae_chat_temp

#%%
#function to format dataset
# Note that since we will be using the map method of HF Datasets object, the result of what this function returns must be a dictionary that will be added to the Dataset object (train_data) we will pass to this function
custom_system_msg = 'you are a helpful customer service assisstant'

def format(conversation):
    formatted_chat = tokenizer.apply_chat_template(
                conversation["messages"], 
                tokenize=False, 
                add_generation_prompt=False,
                default_system_message=custom_system_msg
    )
    
    return {"formatted_chat": formatted_chat} 
    
#%%
custom_system_msg = 'you are an online therapy chatbot'

test_format = train_data.select(range(0,2)).map(
    format
)

print(f'applying the template using map returns anothe HF Dataset object: {type(test_format)}\n')

print(f'the Dataset object has the same feature (columns) as the original Dataset plus a "formatted_chat" column that was returned by the format function\n: {test_format.features}\n')

print(f'the "messageDataset: {test_format['formatted_chat'][0]}\n')
# %%
assert tokenizer.eos_token_id == tokenizer.pad_token_id

# %%
for c in train_data:
    print(c['messages'])
    break


# %%
# function to both format and tokenize
def format_tokenize(example):
    """Format and tokenize a single conversation"""
    formatted_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        default_system_message=custom_system_msg
    )
    
    # Tokenize the formatted text
    tokenized = tokenizer(
        formatted_text,
        truncation=False,
        padding=False,    # Don't pad here; do it in the trainer
        return_attention_mask=True # note tha the attention mask here will return all 1s. the point is to create an attention mask that we will modify later
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "formatted_text": formatted_text  # Keep for reference
    }
#%%
# Test the format_tokenize function
train_data_tokenized = train_data.select(range(1)).map(
    format_tokenize,
    desc="Tokenizing conversations",
    remove_columns=train_data.column_names  # Remove original columns so object has three columns: input_ids, attention_mask, and formatted_text 
)

print(train_data_tokenized.features)
print(train_data_tokenized['formatted_text'][0])
print(train_data_tokenized['input_ids'][0])
print(train_data_tokenized['attention_mask'][0])

#%%
# format_tokenize to entire train_data
num_cores = multiprocessing.cpu_count()

train_data_tokenized = train_data.map(
    format_tokenize,
    desc="Tokenizing conversations",
    remove_columns=train_data.column_names,  # Remove original columns so object has three columns: input_ids, attention_mask, and formatted_text 
    num_proc=num_cores-2
)

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

#%%



# %%
# code below is a helper function to find begining and end of last assistan respnse

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

    # 1. Find the last <|assistant|> token

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


# %%
num_cores = multiprocessing.cpu_count()
train_data_tokenized_2 = train_data.select(range(5000)).map(
    format_tokenize_with_spans,
    desc="Tokenizing conversations",
    remove_columns=train_data.column_names,  # Remove original columns so object has three columns: input_ids, attention_mask, and formatted_text 
    num_proc=num_cores-2
)
#%%
print(f'{train_data_tokenized_2.features}\n')
print(train_data_tokenized_2['formatted_text'][0])
print(train_data_tokenized_2['formatted_text'][1])
print(train_data_tokenized_2['input_ids'][0])
print(train_data_tokenized_2['input_ids'][1])
print(train_data_tokenized_2['attention_mask'][0])
print(train_data_tokenized_2['attention_mask'][1])
print(train_data_tokenized_2["assistant_start"][0])
print(train_data_tokenized_2["assistant_start"][1])
print(train_data_tokenized_2["assistant_end"][0])
print(train_data_tokenized_2["assistant_end"][1])

# %%
# print(tokenizer.all_special_tokens)
# print(tokenizer.all_special_ids)
# print(type(train_data_tokenized_2))

# print(f'sequence length: {len(train_data_tokenized_2['input_ids'][0])}')
# print(f'tokens at assistant content start-1 and start: {train_data_tokenized_2['input_ids'][0][537]}, {train_data_tokenized_2['input_ids'][0][538]}')
# print(f'last token in the sequence: {train_data_tokenized_2['input_ids'][0][707]}')

# %%
# custom data collator
from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch

@dataclass
class AssistantOnlyDataCollator:
    tokenizer: Any
    max_length: int = 4096

    assistant_token_id: int = field(init=False)
    eos_token_id: int = field(init=False)

    # Counters
    total_examples: int = field(init=False, default=0)
    context_truncated_examples: int = field(init=False, default=0)
    assistant_partially_truncated_examples: int = field(init=False, default=0)
    assistant_fully_truncated_examples: int = field(init=False, default=0)
    skipped_examples: int = field(init=False, default=0)

    def __post_init__(self):
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        self.eos_token_id = self.tokenizer.eos_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ex in features:
            self.total_examples += 1

            input_ids = ex["input_ids"]
            attention_mask = ex["attention_mask"]
            a_start = ex["assistant_start"]
            a_end = ex["assistant_end"]

            # check if example is valid as per find_last_assistant_span(), if so, skip
            if not ex["valid"]:
                self.skipped_examples += 1
                continue

            seq_len = len(input_ids)
            overflow = seq_len - self.max_length

            # Track truncation
            truncated = overflow > 0
            cut = 0

            if truncated:
                self.context_truncated_examples += 1
                cut = min(overflow, a_start)

                input_ids = input_ids[cut:]
                attention_mask = attention_mask[cut:]

                # Shift assistant span
                new_a_start = a_start - cut
                new_a_end = a_end - cut
            else:
                new_a_start = a_start
                new_a_end = a_end

            # 2. Classify assistant truncation
            assistant_fully_truncated = (
                new_a_end < 0 or new_a_start >= len(input_ids)
            )

            assistant_partially_truncated = (
                truncated
                and not assistant_fully_truncated
                and (new_a_start < 0 or new_a_end >= len(input_ids))
            )

            if assistant_fully_truncated:
                self.assistant_fully_truncated_examples += 1
            elif assistant_partially_truncated:
                self.assistant_partially_truncated_examples += 1

            # 3. Build labels (assistant-only)
            labels = [-100] * len(input_ids)

            if not assistant_fully_truncated:
                start = max(new_a_start, 0)
                end = min(new_a_end + 1, len(input_ids))
                for i in range(start, end):
                    labels[i] = input_ids[i]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        # 4. Pad batch
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
#%%
collator = AssistantOnlyDataCollator(tokenizer=tokenizer)
# %%
batch = collator(train_data_tokenized_2)

# %%
print(f"total: {collator.total_examples}")
print(f"context truncated: {collator.context_truncated_examples}")
print(f"assistant partially truncated: {collator.assistant_partially_truncated_examples}")
print(f"assistant fully truncated: {collator.assistant_fully_truncated_examples}")
print(f"skipped: {collator.skipped_examples}")

# %%
tokenizer.save_pretrained("./mistral_7B_customized tokenizer")
# %%
tok2 = AutoTokenizer.from_pretrained("./mistral_7B_customized tokenizer")
print(tok2.all_special_tokens)
print(tok2.all_special_ids)
print(tok2.chat_template)

# %%
