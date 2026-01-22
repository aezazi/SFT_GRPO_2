#%%
from transformers import AutoTokenizer
from sft_grpo.sft.mistral.mistral_config import CUSTOM_TOKENIZER_V2_PATH,  MODEL_NAME
from sft_grpo.sft.mistral.mistral_sft_utils import format_tokenize_with_spans
#%%
#%%
# load customized tokenizer and verify special tokens and chat template

tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_V2_PATH)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.pad_token)
# tokenizer.chat_template


#%%
example = {
    "messages": [
        {"role": "user", "content": "where is the cat?"},
        {"role": "assistant", "content": "cat is on an a mat"},
    ]
}
custom_system_msg=None

text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        default_system_message=None
    )
print(text)

enc = tokenizer(
    text,
    add_special_tokens=True,
    truncation=False,
    padding=False,
    return_attention_mask=False,
)
print(f'length tokenized inputs: {len(enc['input_ids'])}\n{enc}\n\n')

input_ids = enc["input_ids"]
labels = [-100] * len(input_ids)


# 3. Identify special tokens
assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
system_token_id = tokenizer.convert_tokens_to_ids("<|system|>")
eos_token_id = tokenizer.eos_token_id

# 4. Masking state
in_assistant = False

for i, tok in enumerate(input_ids):

    # Role boundaries
    if tok == assistant_token_id:
        in_assistant = True
        labels[i] = -100
        continue

    if tok == user_token_id or tok == system_token_id:
        in_assistant = False
        labels[i] = -100
        continue

    # Assistant span → supervise EVERYTHING (including EOS)
    if in_assistant:
        labels[i] = tok
    else:
        labels[i] = -100



#%%
# now lets day our max sequence length is 30 (the tokenized example above is 24 tokens long). so we need padding
# compute the starting and ending position of the eligible window for the entire sequence starting from the back, so the window end is the last position. window start is either at position 0 (if seq_len is <= max_length) or seq_len - self.max_length ( if seq_len > max_length)
seq_len = len(input_ids)
print(f'seq_len: {seq_len}')
window_end = seq_len
print(f'window_end: {window_end}')
window_start = max(0, seq_len - 30)
print(f'window_start: {window_start}')

assistant_positions = [i for i, label in enumerate(labels) if label != -100]
print(f'assistant_positions:\n{assistant_positions}')

# compute the positions of the assistant tokens in the window_start to window_end span.
kept_asst_positions = [i for i in assistant_positions if window_start <= i < window_end]
print(f'kept_asst_positions\n{kept_asst_positions}')
kept_asst_count = len(kept_asst_positions)

# create an attention mask that is 1s for the entire sequence length. This is because in the forward pass the model must see all system, user, and assistant messages. the labels (targets) will be all -100 except the assistant messages. this gets the model to train onl 
attention_mask = []
attention_mask.extend([1] * (window_end - window_start))
print(f'attentin_mask:\n{attention_mask}')

#%%
# padding
max_len = 30
pad_token_id = tokenizer.pad_token_id
label_pad_token_id = -100



padded_input_ids = input_ids + ([pad_token_id] * (max_len- len(input_ids)))
padded_labels = labels + ([label_pad_token_id] * (max_len- len(input_ids)))
padded_attention_mask = attention_mask + ([0] * (max_len - len(input_ids)))

print(f'\nformatted sequence:')
print(text)

print(f'\nmax_length: {max_len}')
print(f'tokenized sequence length: {len(input_ids)}')
print(f'eos token id: {2}')
print(f'pad_token_id: {pad_token_id}')
print(f'label_pad_token_id: {label_pad_token_id}')

print(f'\ninpt_ids, padded_ids:')
print(input_ids)
print(padded_input_ids)

print(f'\nlabels, padded_labels:')
print(labels)
print(padded_labels)

print(f'\nattention_mask, padded_attention_mask:')
print(attention_mask)
print(padded_attention_mask)



#%%
# how the model will use input_ids and labels
# inside the Mistral model, the tensors are sliced before the loss is calculated. If your batch has a sequence length of 3072:The model takes the first 3071 outputs (logits).The model takes the last 3071 labels.The 3072nd prediction is simply thrown away because there is no ground truth for it. Note that the attention mask does not need to be sliced. the attention mask is used only in the forward pass to excluse padding tokens from being included in the attetion computations

input_ids_slice = input_ids[:-1]
labels_slice = labels[1:]

print(input_ids)
print(labels)

# match input_id to target
print(f'\nmatching input_ids to targets')
for i, id in enumerate(input_ids_slice):
    print(f'{id} --> {labels_slice[i]}')


# %%
