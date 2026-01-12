# %%
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
hf_token = os.getenv("hf_access_token")

print("="*80)
print("LOADING LORA FINE-TUNED MODEL")
print("="*80)



# 1. Load base model
print("\n1. Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_token
)
print(f"   ✅ Base model loaded")

#%%
# 2. Load LoRA adapters

# Note that with PEFT LoRA trained model, only the adapters are saved. so we have to load the base model and then add the adapters. 
model_save_path = "/Users/solo_60/Desktop/SFT_GRPO/sft_using_llama_3B/llama3.2_base_cs_chat_v4/checkpoint-4536"

print("\n2. Loading LoRA adapters...")
model = PeftModel.from_pretrained(
    base_model,
    model_save_path,
    torch_dtype=torch.bfloat16,
)
print(f"   ✅ LoRA adapters loaded")


#%%
# 3. Setup tokenizer
print("\n3. Setting up tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_save_path, token=hf_token)
tokenizer.padding_side = 'left' #Note that for generation we should pad left

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
print(f"   ✅ Tokenizer configured")

# 4. Resize embeddings. 
print("\n4. Resizing embeddings...")
model.resize_token_embeddings(len(tokenizer))
print(f"   ✅ Resized to {len(tokenizer)}")

print(f'\nlen tokenizer: {len(tokenizer)}  embedding size: {model.get_input_embeddings().weight.shape[0]}')

# 5. Verify token IDs
print("\n5. Verifying token IDs...")
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

print(f"   <|eot_id|>: {eot_id}")
print(f"   eos_token_id: {eos_id}")
print(f"   pad_token_id: {pad_id}")

print("\n✅ MODEL READY!")
print("="*80)


#%%
# 6. Build bad words list
print("\nBuilding bad words list...")
bad_words_ids = []
for i in range(256):
    token_name = f"<|reserved_special_token_{i}|>"
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id != tokenizer.unk_token_id:
        bad_words_ids.append([token_id])
print(f"✅ Blocking {len(bad_words_ids)} reserved tokens")

#%%
# 7. Test generation with FIXED parameters
# 5. Get all stop token IDs

import re

print("4. Identifying stop tokens...")
stop_token_names = ["<|eot_id|>", "<|end_of_text|>"]
stop_token_ids = []

for token_name in stop_token_names:
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id != tokenizer.unk_token_id:
        stop_token_ids.append(token_id)
        print(f"   {token_name}: {token_id}")


# a reusable fuction for generation
def generate_response(query, system_message, model, tokenizer, stop_token_ids, bad_words_ids):
    """Generate response with robust stopping and validation."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with strict parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,  # Increased slightly
        min_new_tokens=10,   # Ensure minimum response length
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_token_ids,  # Multiple stop tokens
        bad_words_ids=bad_words_ids,
        repetition_penalty=1.15,  # Increased slightly
        no_repeat_ngram_size=3,  # Prevent 3-gram repetition
    )
    
    # Decode full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant response more robustly
    if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
        # Split and get last assistant section
        parts = full_response.split("<|start_header_id|>assistant<|end_header_id|>")
        assistant_part = parts[-1]
        
        # Find first occurrence of ANY stop token
        stop_positions = []
        for stop_token in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
            pos = assistant_part.find(stop_token)
            if pos != -1:
                stop_positions.append(pos)
        
        if stop_positions:
            # Stop at earliest token
            earliest_stop = min(stop_positions)
            assistant_response = assistant_part[:earliest_stop]
        else:
            # No stop token found - take everything
            assistant_response = assistant_part
        
        assistant_response = assistant_response.strip()
    else:
        assistant_response = full_response
    
    # Validation
    issues = []
    
    # Check for reserved/problematic tokens in FULL response
    if "<|reserved_special_token" in full_response:
        issues.append("Reserved tokens in full response")
    if "<|finetune_right_pad_id|>" in full_response:
        issues.append("Finetune padding token appeared")
    if "<|eom_id|>" in full_response:
        issues.append("EOM token appeared")
    
    # Check for multiple conversation turns
    assistant_count = full_response.count("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_count > 1:
        issues.append(f"Multiple assistant responses ({assistant_count})")
    
    user_count = full_response.count("<|start_header_id|>user<|end_header_id|>")
    if user_count > 1:
        issues.append(f"Multiple user messages ({user_count})")
    
    # Check for non-ASCII characters (potential garbled text)
    non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
    if non_ascii_pattern.search(assistant_response):
        non_ascii_chars = non_ascii_pattern.findall(assistant_response)
        issues.append(f"Non-ASCII characters: {non_ascii_chars[:3]}")  # Show first 3
    
    # Check for incomplete response (ends mid-sentence)
    if assistant_response and assistant_response[-1] not in '.!?':
        # Might be truncated
        if len(assistant_response) > 50:  # Only flag if substantial
            issues.append("Response appears truncated (no ending punctuation)")
    
    return assistant_response, issues, full_response


#%%
# test generation 

# 8. Test with various queries
print("\n" + "="*80)
print("COMPREHENSIVE TESTING")
print("="*80)

SYSTEM_MESSAGE = "You are a helpful and professional customer service assistant. Provide clear, accurate, and friendly responses to customer inquiries."

test_cases = [
    ("How do I track my order?", "tracking"),
    ("I want to return my product", "return"),
    ("What's your refund policy?", "policy"),
    ("your service is terrible, you're a bunch of crooks", "complaint"),
    ("Can I change my shipping address?", "address"),
]

for i, (query, category) in enumerate(test_cases, 1):
    print(f"\n[TEST {i}/{len(test_cases)}] Category: {category}")
    print("="*80)
    print(f"Query: {query}")
    print("Generating...")
    
    response, issues, full_response = generate_response(
        query,
        SYSTEM_MESSAGE,
        model,
        tokenizer,
        stop_token_ids,
        bad_words_ids
    )
    
    print(f"\nResponse:\n{response}")
    print(f"\nResponse length: {len(response)} characters")
    
    # Show issues
    if issues:
        print(f"\n⚠️  ISSUES DETECTED ({len(issues)}):")
        for issue in issues:
            print(f"   ❌ {issue}")
        
        # Show problematic portion of full response for debugging
        print(f"\n🔍 Debug - Full response snippet:")
        # Show last 200 chars where issues likely are
        if len(full_response) > 200:
            print(f"...{full_response[-200:]}")
        else:
            print(full_response)
    else:
        print(f"\n✅ CLEAN - No issues detected")
    
    print("="*80)

print("\n✅ ALL TESTS COMPLETE!")
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("If you still see issues, the model may need:")
print("1. Retraining with better stop token handling")
print("2. More aggressive max_new_tokens limit")
print("3. Post-processing to strip everything after first valid response")
print("="*80)


#%%
# save the tokenizer if satisfied with tests

# %%
# Merge LoRA weights into base model for standalone deployment
merged_model = model.merge_and_unload()

# Save merged model
merged_output = "llama3-customer-service-merged"
merged_model.save_pretrained(merged_output)
tokenizer.save_pretrained(merged_output)

# Now you can load without PeftModel:
# model = AutoModelForCausalLM.from_pretrained(merged_output)

# %%