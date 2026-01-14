#%%
# imports
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os


# NEW: Import from your package
from sft_grpo.config import HF_TOKEN
from sft_grpo.sft.mistral.mistral_config import CUSTOM_TOKENIZER_V2_PATH, DATASET_DIR, MODEL_NAME, MISTRAL_SFT_ROOT
from sft_grpo.sft.mistral.mistral_sft_utils import TruncatingCollator

#%%
# Load base model
print(MODEL_NAME)

# Define model base parameters in a dictionary
model_params = {
    "pretrained_model_name_or_path": MODEL_NAME,
    "dtype": torch.bfloat16,
    "device_map": "auto",
    "token": HF_TOKEN
}

# check if cuda is available. If so add flash attention to model parameters
if torch.cuda.is_available():
    model_params["attn_implementation"] = "flash_attention_2"
    print('cuda available')
    # print(MODEL_NAME.config._attn_implementation)
else:
    print('cuda not available')

# Load the model using dictionary unpacking (**)
model = AutoModelForCausalLM.from_pretrained(**model_params)
print(model.config._attn_implementation)


#%%
# load customized tokenizer and verify special tokens and chat template

tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_V2_PATH)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.pad_token)

#verify pad token is set eos token
print(f'pad token set to eos token: {tokenizer.pad_token == tokenizer.eos_token}')
print(f'pad token id set to eos token id: {tokenizer.pad_token_id == tokenizer.eos_token_id}')

# check if the model tokenizer already has a chat template
tokenizer.chat_template

#%%
# THIS IS CRITICAL. ALWAYS RESIZE THE MODEL TO ACCOMMODATE THE EXTRA SPECIAL TOKENS
model.resize_token_embeddings(len(tokenizer))

# %%
# load data sets
train_data = load_from_disk(DATASET_DIR / 'train_dataset_tokenized_v2')

eval_data = load_from_disk(DATASET_DIR / 'test_dataset_tokenized_v2')


# %%
# configure PEFT LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


#%%
# Updated Training Arguments optimized for single H200 GPU
training_args = TrainingArguments(
    output_dir=str(MISTRAL_SFT_ROOT / "checkpoints"),
    overwrite_output_dir=True,

    remove_unused_columns=False,
    
    # Batch sizes
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=4,

    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # Training duration
    num_train_epochs=1,
    
    # Logging and evaluation
    logging_steps=10,
    eval_steps=1000,
    save_steps=1000,
    eval_strategy="steps",
    save_strategy="steps",
    
    # Model saving
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Precision
    bf16=True,
    fp16=False,
    bf16_full_eval=True,

    # Optimizer
    optim="adamw_torch_fused",
    max_grad_norm=1.0,

    # Reporting
    report_to="none",
    run_name="sft-mistral-7b-ultrachat",

    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    group_by_length=False,
    
    # Memory optimization
    gradient_checkpointing=False,
    ddp_find_unused_parameters=False,
)

#%%
# Instantiate the data collator imported from utils
data_collator = TruncatingCollator(tokenizer, max_length=3072)

#%%
# Test the collator before training
print("\n" + "="*80)
print("TESTING COLLATOR")
print("="*80)

test_batch = [train_data[i] for i in range(4)]

print("Sample input types before collation:")
print(f"  input_ids type: {type(test_batch[0]['input_ids'])}")
print(f"  labels type: {type(test_batch[0]['labels'])}")
print(f"  input_ids[0] length: {len(test_batch[0]['input_ids'])}")
print(f"  input_ids[1] length: {len(test_batch[1]['input_ids'])}")

collated = data_collator(test_batch)

print("\nCollated batch keys:", collated.keys())
print("Input IDs shape:", collated["input_ids"].shape)
print("Labels shape:", collated["labels"].shape)
print("Attention mask shape:", collated["attention_mask"].shape)
print("\nSample attention mask (first sequence):")
print(collated["attention_mask"][0])
print("\nInitial collator stats:")
data_collator.print_stats()
print("="*80 + "\n")

#%%
# UPDATED CALLBACK WITH FIXED STAT NAMES
import csv
from transformers import TrainerCallback, TrainerControl, TrainerState
from datetime import datetime
import json


class SFTLoggingCallback(TrainerCallback):
    def __init__(self, log_file=None, collator=None):
        self.log_file = log_file
        self.collator = collator
        self.start_time = None
        self.best_eval_loss = float('inf')
        self.training_history = []
        
        # Check if log file exists to avoid overwriting 4000 steps of data
        file_exists = os.path.exists(self.log_file)
        
        # Open in append mode ('a') instead of write mode ('w')
        if not file_exists:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "step", "epoch", "train_loss", "eval_loss", 
                    "learning_rate", "examples_seen", "examples_per_sec", 
                    "no_truncation", "context_only_truncated", "assistant_partial_loss", 
                    "assistant_complete_loss", "skipped_no_labels"
                ])
        else:
            print(f"✅ Existing log found at {self.log_file}. Resuming in append mode.")

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        print("=" * 80)
        print(f"Training resumed at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        return control

    # on_log and on_train_end remain the same as your original script, 
    # as they already use 'a' (append) mode.


#%%
# Generation test callback
class GenerationTestCallback(TrainerCallback):
    """
    Callback to test model generation during training.
    """
    def __init__(self, tokenizer, test_prompts, generation_steps=500):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generation_steps = generation_steps
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Generate sample outputs during evaluation"""
        if state.global_step % self.generation_steps == 0 and state.global_step > 0:
            print(f"\n{'='*80}")
            print(f"GENERATION TEST AT STEP {state.global_step}")
            print(f"{'='*80}")
            
            model.eval()
            
            for i, prompt in enumerate(self.test_prompts):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    input_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=150,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    print(f"\n--- Test Prompt {i+1} ---")
                    print(f"Prompt: {prompt}")
                    print(f"Generated: {generated_text}")
                    print("-" * 80)
                
                except Exception as e:
                    print(f"\n--- Test Prompt {i+1} FAILED ---")
                    print(f"Prompt: {prompt}")
                    print(f"Error: {e}")
                    print("-" * 80)
            
            print(f"{'='*80}\n")
            model.train()
        
        return control


#%%
# Instantiate callbacks

# Ensure the logs directory exists before starting
(DATASET_DIR.parent / "logs").mkdir(parents=True, exist_ok=True)

logging_callback = SFTLoggingCallback(
    log_file=str(MISTRAL_SFT_ROOT / "logs" / "training_log.csv"),
    collator=data_collator
)

test_prompts = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "How do I make a good cup of coffee?"
]

generation_callback = GenerationTestCallback(
    tokenizer=tokenizer, 
    test_prompts=test_prompts,
    generation_steps=500
)


#%%
# DIAGNOSTIC: Check labels in a batch
print("\n" + "="*80)
print("DIAGNOSTIC: Checking batch labels")
print("="*80)

test_batch = [train_data[i] for i in range(4)]
collated = data_collator(test_batch)

print(f"Batch shape: {collated['input_ids'].shape}")
print(f"Labels shape: {collated['labels'].shape}")

for i in range(len(collated['labels'])):
    labels = collated['labels'][i]
    non_ignored = (labels != -100).sum().item()
    total = labels.shape[0]
    print(f"Example {i}: {non_ignored}/{total} tokens have labels ({100*non_ignored/total:.1f}%)")

total_tokens = collated['labels'].numel()
labeled_tokens = (collated['labels'] != -100).sum().item()
print(f"\nOverall: {labeled_tokens}/{total_tokens} tokens have labels ({100*labeled_tokens/total_tokens:.1f}%)")
print("="*80 + "\n")

#%%
# 🔥 FORCE FRESH MODEL RELOAD BEFORE TRAINING
# This ensures we're starting from a clean state, not contaminated by diagnostics
print("\n" + "="*80)
print("🔄 RELOADING MODEL TO GUARANTEE FRESH STATE")
print("="*80)

# Delete existing model and clear cache
del model
torch.cuda.empty_cache()

# Reload from scratch
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    token=HF_TOKEN
)

# Resize for special tokens
model.resize_token_embeddings(len(tokenizer))

# Apply fresh LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Verify fresh state
print("\nVerifying fresh model state...")
test_batch = data_collator([train_data[i] for i in range(4)])
test_batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in test_batch.items()}

model.eval()
with torch.no_grad():
    fresh_loss = model(**test_batch).loss.item()
    print(f"✓ Fresh model loss: {fresh_loss:.4f}")
    
    if fresh_loss < 0.8:
        print("⚠️  WARNING: Loss is still suspiciously low!")
        print("   Consider restarting the Python kernel entirely.")
    elif fresh_loss < 1.2:
        print("✅ Good! Normal for LoRA with strong base model.")
    else:
        print("✅ Model is fresh and ready!")

print("="*80 + "\n")

# Reset collator stats to start fresh
data_collator.reset_stats()

#%%
# Create trainer with fresh model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    callbacks=[logging_callback, generation_callback],
)

# Print training setup summary
print("\n" + "="*80)
print("TRAINING SETUP SUMMARY")
print("="*80)
print(f"Model: {MODEL_NAME}")
print(f"Training samples: {len(train_data):,}")
print(f"Eval samples: {len(eval_data):,}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size}")
print(f"Steps per epoch: {len(train_data) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)}")
print(f"Warmup steps: {int(0.03 * len(train_data) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size))}")
print(f"Logging every: {training_args.logging_steps} steps")
print(f"Evaluating every: {training_args.eval_steps} steps")
print(f"Saving checkpoints every: {training_args.save_steps} steps")
print(f"Output directory: {training_args.output_dir}")
print(f"Max sequence length: {data_collator.max_length}")
print("="*80 + "\n")


#%%
# Start training
def start_training():
    print("🚀 Starting training...")
    print("="*80 + "\n")
    
    trainer.train()

    # Define final model path relative to Mistral root
    final_model_path = MISTRAL_SFT_ROOT / "experiments" / "final_sft_model_v1"
    
    print(f"\nSaving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print("\n✅ Training complete!")
    print(f"Final model saved to: ./final_model")
    print(f"Best model saved to: {training_args.output_dir}")

#%%
start_training()