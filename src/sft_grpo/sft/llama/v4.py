#%%
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import torch
import os
from dotenv import load_dotenv


#%%
# Load environment variables
load_dotenv()
hf_token = os.getenv("hf_access_token")
print(hf_token)

if torch.cuda.is_available():
    print('cuda available')

#%%
# Load BASE model
model_name = "meta-llama/Llama-3.2-3B"

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
# Set up chat template for base model (LLaMA 3 format)
if tokenizer.chat_template is None:
    print("Setting up LLaMA 3 chat template...")
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
    print("✅ Chat template configured!")

# Load dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# System message for customer service
SYSTEM_MESSAGE = "You are a helpful and professional customer service assistant. Provide clear, accurate, and friendly responses to customer inquiries."

# Format using chat template
def format_with_chat_template(example):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

formatted_dataset = dataset.map(format_with_chat_template, remove_columns=dataset["train"].column_names)

# Tokenization with label masking
def tokenize_function(examples):
    # Tokenize the full text
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,  # Don't pad here - let the data collator handle it
    )
    
    # Create labels - we want to mask everything except the assistant's response
    labels = []
    for input_ids in result["input_ids"]:
        label = input_ids.copy()
        
        # Find where the assistant response starts
        # Look for the pattern: <|start_header_id|>assistant<|end_header_id|>
        assistant_header_start = "<|start_header_id|>assistant<|end_header_id|>"
        
        # Tokenize just the assistant header to find it
        assistant_header_tokens = tokenizer.encode(
            assistant_header_start, 
            add_special_tokens=False
        )
        
        # Find the position where assistant content begins
        # We want to mask everything BEFORE the assistant's actual content
        mask_until_idx = 0
        for i in range(len(label) - len(assistant_header_tokens)):
            if label[i:i+len(assistant_header_tokens)] == assistant_header_tokens:
                # Found the assistant header, now skip past it and the newlines
                # The content starts after the header + "\n\n"
                mask_until_idx = i + len(assistant_header_tokens)
                
                # Skip the newline tokens that come after the header
                # This is typically 1-2 tokens depending on tokenization
                while mask_until_idx < len(label):
                    token_text = tokenizer.decode([label[mask_until_idx]])
                    if token_text.strip():  # Found actual content
                        break
                    mask_until_idx += 1
                break
        
        # Mask everything before the assistant's response content
        if mask_until_idx > 0:
            label[:mask_until_idx] = [-100] * mask_until_idx
        
        labels.append(label)
    
    result["labels"] = labels
    return result

tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Split dataset
train_test = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# Data collator with proper label handling
from transformers import DataCollatorForLanguageModeling

# Custom data collator that handles labels properly
class CustomDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        # Separate labels from examples for padding
        labels = [example.pop("labels") for example in examples if "labels" in example]
        
        # Pad input_ids and attention_mask
        batch = super().torch_call(examples)
        
        # Now pad labels to match input_ids length
        if labels:
            import torch
            max_length = batch["input_ids"].shape[1]
            
            padded_labels = []
            for label in labels:
                padding_length = max_length - len(label)
                # Pad labels with -100 (ignored in loss calculation)
                padded_label = label + [-100] * padding_length
                padded_labels.append(padded_label)
            
            batch["labels"] = torch.tensor(padded_labels)
        
        return batch

data_collator = CustomDataCollator(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# CRITICAL: Enable gradient checkpointing AFTER applying LoRA
# This must be done on the base model, not the PEFT wrapper
if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()
else:
    # Fallback for older versions
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

print("\n" + "="*50)
print("Trainable parameters:")
model.print_trainable_parameters()
print("="*50 + "\n")

# DIAGNOSTIC: Verify trainable parameters
print("🔍 Checking trainable parameters...")
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
if not trainable_params:
    print("❌ ERROR: No trainable parameters found!")
    print("This shouldn't happen with LoRA. Checking model structure...")
else:
    print(f"✅ Found {len(trainable_params)} trainable parameters")
    print("Sample trainable params:")
    for name in trainable_params[:5]:
        print(f"  - {name}")
print("="*50 + "\n")



# Training arguments
training_args = SFTConfig(
    output_dir="./llama3.2_base_cs_chat_v4",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # Logging
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=50,
    eval_accumulation_steps=1,
    
    # Saving
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    
    # Performance
    bf16=True,
    bf16_full_eval=True,
    
    neftune_noise_alpha=5, # Adds small random noise to embeddings during training. Acts as a regularization technique
    weight_decay=0.01, 
    max_grad_norm=1.0, 
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    
    # Reporting - disable default progress bars for cleaner output
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    disable_tqdm=False,
    log_level="warning",  # Suppress INFO logs that print the dict
)

# Custom callback for clean logging with CSV export
from transformers import TrainerCallback
import csv
import os

class CleanLoggingCallback(TrainerCallback):
    def __init__(self, train_csv="training_log.csv", eval_csv="eval_log.csv"):
        self.train_csv = train_csv
        self.eval_csv = eval_csv
        self.steps_per_epoch = None
        self.total_epochs = None
        
        # Initialize CSV files with headers
        with open(self.train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'learning_rate', 'grad_norm'])
        
        with open(self.eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'eval_loss', 'learning_rate'])
    
    def on_train_begin(self, args, state, control, **kwargs):
        # Calculate steps per epoch and total epochs
        self.total_epochs = int(args.num_train_epochs)
        self.steps_per_epoch = state.max_steps // self.total_epochs
    
    def _ensure_initialized(self, args, state):
        """Ensure steps_per_epoch and total_epochs are initialized"""
        if self.steps_per_epoch is None or self.total_epochs is None:
            self.total_epochs = int(args.num_train_epochs)
            # Calculate steps per epoch from training arguments
            # Use the actual dataset size and batch configuration
            if state.max_steps > 0:
                self.steps_per_epoch = state.max_steps // self.total_epochs
            else:
                # Fallback calculation for initial evaluation
                # This happens before training starts
                self.steps_per_epoch = 1  # Placeholder, will be updated on train begin
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Ensure initialized (in case on_train_begin wasn't called yet)
            self._ensure_initialized(args, state)
            
            # Skip epoch calculation if we don't have valid steps_per_epoch yet
            if self.steps_per_epoch is None or self.steps_per_epoch == 0:
                # Just log without epoch info for initial eval
                if 'eval_loss' in logs:
                    eval_loss = logs['eval_loss']
                    print(f"[INITIAL EVAL] Eval Loss: {eval_loss:.4f}")
                    # Save to CSV
                    with open(self.eval_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([state.global_step, eval_loss, 0])
                return
            
            # Training metrics
            if 'loss' in logs:
                loss = logs['loss']
                lr = logs.get('learning_rate', 0)
                grad_norm = logs.get('grad_norm', 0)
                
                # Calculate current epoch and step within epoch
                # Don't add 1 to current_epoch since we're 0-indexed until we complete an epoch
                current_epoch = (state.global_step // self.steps_per_epoch) + 1
                step_in_epoch = state.global_step % self.steps_per_epoch
                if step_in_epoch == 0:
                    step_in_epoch = self.steps_per_epoch
                
                # Print single clean line
                print(f"\nStep {step_in_epoch}/{self.steps_per_epoch} | Epoch {current_epoch}/{self.total_epochs} | "
                      f"Loss: {loss:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.4f}")
                
                # Save to CSV
                with open(self.train_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([state.global_step, loss, lr, grad_norm])
            
            # Evaluation metrics
            elif 'eval_loss' in logs:
                eval_loss = logs['eval_loss']
                lr = logs.get('learning_rate', 0)
                
                current_epoch = (state.global_step // self.steps_per_epoch) + 1
                step_in_epoch = state.global_step % self.steps_per_epoch
                if step_in_epoch == 0:
                    step_in_epoch = self.steps_per_epoch
                
                # Print single clean line for eval
                print(f"\n[EVAL] Step {step_in_epoch}/{self.steps_per_epoch} | Epoch {current_epoch}/{self.total_epochs} | "
                      f"Eval Loss: {eval_loss:.4f} | LR: {lr:.2e}")
                
                # Save to CSV
                with open(self.eval_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([state.global_step, eval_loss, lr])

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[CleanLoggingCallback(train_csv="training_log.csv", eval_csv="eval_log.csv")],
)

# Print sample
print("\n" + "="*50)
print("Sample training example:")
sample = train_dataset[0]
decoded = tokenizer.decode(sample['input_ids'])
print(f"Full text:\n{decoded}\n")
print(f"Input IDs length: {len(sample['input_ids'])}")
print(f"Number of non-masked tokens: {sum(1 for x in sample['labels'] if x != -100)}")

# Show what's masked vs not masked
print("\nLabel masking visualization (first 50 tokens):")
for i in range(min(50, len(sample['input_ids']))):
    token = tokenizer.decode([sample['input_ids'][i]])
    masked = "🔴 MASKED" if sample['labels'][i] == -100 else "✅ TRAINED"
    print(f"  [{i:3d}] {masked}: '{token}'")
print("="*50 + "\n")

# Initial evaluation
print("Running initial evaluation...")
initial_metrics = trainer.evaluate()
print(f"\n📈 Initial Eval Loss: {initial_metrics['eval_loss']:.4f}\n")

#%%
# ============================= start trainining ==============================================

print("Starting training...\n")
trainer.train()

#===============================================================================================

# Final evaluation
print("\n" + "="*50)
print("Running final evaluation...")
final_metrics = trainer.evaluate()
print(f"Final Eval Loss: {final_metrics['eval_loss']:.4f}")
print(f"Improvement: {initial_metrics['eval_loss'] - final_metrics['eval_loss']:.4f}")
print("="*50 + "\n")

# Save
trainer.save_model("./llama3.2_base_cs_chat_v2_final")
tokenizer.save_pretrained("./llama3.2_base_cs_chat_v4_final")
print("✅ Model and tokenizer saved!")

# Test generation
print("\n" + "="*50)
print("Testing generation:")

# Create a test prompt using the chat template
test_messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": "How do I track my order?"}
]

test_prompt = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True  # This adds the assistant header
)

print(f"Test prompt:\n{test_prompt}\n")

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"Full response:\n{response}")

# Extract just the assistant's response
assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
assistant_response = assistant_response.replace("<|eot_id|>", "").strip()
print(f"\nAssistant's response only:\n{assistant_response}")
print("="*50 + "\n")