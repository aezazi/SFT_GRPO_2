#%%
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


#%%
# this is how raw conversation data needs to be formatted. The chat template is then applied to this create what is fed to the model
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]


# %%
# note that some models have bos tokens, some do not. some have dedicated pad tokens some do not.
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

print("mistralai/Mistral-7B-Instruct-v0.1")
print(tokenizer.all_special_tokens)
print(f'bos token: {tokenizer.bos_token}')
print(f'eos token: {tokenizer.eos_token}')
print(f'unk token: {tokenizer.unk_token}')
print(f'pad token: {tokenizer.pad_token}')
print(f'cls token: {tokenizer.cls_token}') # In bert models for classification
print(f'mask token: {tokenizer.mask_token}\n')

print("meta-llama/Llama-3.2-3B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
print(tokenizer.all_special_tokens)
print(f'bos token: {tokenizer.bos_token}')
print(f'eos token: {tokenizer.eos_token}')
print(f'unk token: {tokenizer.unk_token}')
print(f'pad token: {tokenizer.pad_token}')
print(f'cls token: {tokenizer.cls_token}') # In bert models for classification
print(f'mask token: {tokenizer.mask_token}\n')

print("HuggingFaceH4/zephyr-7b-beta")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
print(tokenizer.all_special_tokens)
print(f'bos token: {tokenizer.bos_token}')
print(f'eos token: {tokenizer.eos_token}')
print(f'unk token: {tokenizer.unk_token}')
print(f'pad token: {tokenizer.pad_token}')
print(f'cls token: {tokenizer.cls_token}') # In bert models for classification
print(f'mask token: {tokenizer.mask_token}\n')


# %%
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto", dtype=torch.bfloat16)

#%%
messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))

# %%
print(tokenizer.apply_chat_template(chat, tokenize=False))
print(tokenizer.apply_chat_template(messages, tokenize=False))
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# %%
tokenizer.chat_template.bos_token


#%%
# Refactored chat template with dynamic system role and conditional BOS token
# The default_system_message variable will be passsed to the template automatically by setting the default_system_message parameter of tokenizer.apply_chat_template. see code below
# note that if the model tokenizer already has a eos and/or bos, it is best practice to use those for the chat template. In the case of the mistral model, the chat template does not use the models bos token. But according to my research its best to add it. 


aae_chat_temp =  """{% if bos_token %}{{ bos_token }}{% endif %}{% if messages[0]['role'] != 'system' %}{% set default_msg = 'You are a helpful assistant.' %}{% if default_system_message %}{% set default_msg = default_system_message %}{% endif %}{% set messages = [{'role': 'system', 'content': default_msg}] + messages %}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}</s>
{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}</s>
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}</s>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""

tokenizer.chat_template = aae_chat_temp


# %%
# Apply the new chat template to the tokenizer
tokenizer.chat_template = aae_chat_temp

# Test examples
test_conversations = [
    # Example 1: Conversation without system message
    {
        "messages": [
            {"role": "user", "content": "Hello! I need help with my order."},
            {"role": "assistant", "content": "Of course! I'd be happy to help. Could you provide your order number?"},
            {"role": "user", "content": "It's ORD-12345."}
        ]
    },
    # Example 2: Conversation with custom system message
    {
        "messages": [
            {"role": "system", "content": "You are a technical support specialist for software products."},
            {"role": "user", "content": "My application keeps crashing on startup."},
            {"role": "assistant", "content": "I'll help you troubleshoot this. What operating system are you using?"}
        ]
    },
    # Example 3: Single user message (for generation)
    {
        "messages": [
            {"role": "user", "content": "What are your business hours?"}
        ]
    }
]

# Tokenize and print examples
print("=" * 80)
print("REFACTORED ZEPHYR CHAT TEMPLATE FOR MISTRAL-7B SFT")
print("=" * 80)
print("\nChat Template:")
print(aae_chat_temp)
print("\n" + "=" * 80)

for i, conv in enumerate(test_conversations, 1):
    print(f"\n{'='*80}")
    print(f"EXAMPLE {i}")
    print(f"{'='*80}")
    
    # Determine if we should pass a custom system message
    custom_system = None
    if i == 1:
        custom_system = "You are an expert technical support agent specializing in software troubleshooting."
    
    # Apply chat template (without generation prompt)
    if custom_system:
        print(f"\nUsing custom system message: {repr(custom_system)}")
        formatted_chat = tokenizer.apply_chat_template(
            conv["messages"], 
            tokenize=False, 
            add_generation_prompt=False,
            default_system_message=custom_system
        )
    else:
        print("\nUsing default system message")
        formatted_chat = tokenizer.apply_chat_template(
            conv["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
    
    print("\nFormatted Chat (text):")
    print(repr(formatted_chat))
    
    # Tokenize
    if custom_system:
        tokens = tokenizer.apply_chat_template(
            conv["messages"], 
            tokenize=True, 
            add_generation_prompt=False,
            default_system_message=custom_system
        )
    else:
        tokens = tokenizer.apply_chat_template(
            conv["messages"], 
            tokenize=True, 
            add_generation_prompt=False
        )
    
    print(f"\nTokenized (first 50 tokens):")
    print(tokens[:50])
    print(f"\nTotal tokens: {len(tokens)}")
    
    # Decode to verify
    decoded = tokenizer.decode(tokens)
    print("\nDecoded text:")
    print((decoded))
    
    # With generation prompt (for inference)
    if custom_system:
        formatted_with_prompt = tokenizer.apply_chat_template(
            conv["messages"], 
            tokenize=False, 
            add_generation_prompt=True,
            default_system_message=custom_system
        )
    else:
        formatted_with_prompt = tokenizer.apply_chat_template(
            conv["messages"], 
            tokenize=False, 
            add_generation_prompt=True
        )
    print("\nWith generation prompt:")
    print((formatted_with_prompt))

print("\n" + "=" * 80)
print("TOKENIZER SPECIAL TOKENS")
print("=" * 80)
print(f"BOS token: {(tokenizer.bos_token)} (ID: {tokenizer.bos_token_id})")
print(f"EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
print(f"PAD token: {repr(tokenizer.pad_token)} (ID: {tokenizer.pad_token_id})")
print(f"UNK token: {repr(tokenizer.unk_token)} (ID: {tokenizer.unk_token_id})")
# %%
