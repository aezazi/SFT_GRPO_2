#%%
# imports
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os
from dotenv import load_dotenv

# NEW: Import from your package
from mistral.config import CUSTOM_TOKENIZER_V2_PATH, DATASET_DIR
from mistral.utils import TruncatingCollator
from mistral.config import HF_TOKEN


#%%
# Load base model
model_name = "mistralai/Mistral-7B-v0.1"

# Define model base parameters in a dictionary
model_params = {
    "pretrained_model_name_or_path": model_name,
    "dtype": torch.bfloat16,
    "device_map": "auto",
    "token": HF_TOKEN
}

# check if cuda is available. If so add flash attention to model parameters
if torch.cuda.is_available():
    model_params["attn_implementation"] = "flash_attention_2"
    print('cuda available')
    print(model_name.config._attn_implementation)
else:
    print('cuda not available')

# Load the model using dictionary unpacking (**)
model = AutoModelForCausalLM.from_pretrained(**model_params)


#%%
# load customized tokenizer and verify special tokens and chat template

tokenizer = AutoTokenizer.from_pretrained(
    str(CUSTOM_TOKENIZER_V2_PATH),
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

#%%
# THIS IS CRITICAL. ALWAYS RESIZE THE MODEL TO ACCOMMODATE THE EXTRA SPECIAL TOKENS
model.resize_token_embeddings(len(tokenizer))

# %%
# load data sets
train_data = load_from_disk(str(DATASET_DIR / 'train_dataset_tokenized_v2'))

train_data = load_from_disk(str(DATASET_DIR / 'test_dataset_tokenized_v2'))


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
    output_dir="./checkpoints",
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
    eval_steps=50,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    
    # Model saving
    save_total_limit=3,
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
    """
    Custom callback for logging training metrics and collator statistics.
    """
    def __init__(self, log_file="training_log.csv", collator=None):
        self.log_file = log_file
        self.collator = collator
        self.start_time = None
        self.best_eval_loss = float('inf')
        self.training_history = []
        
        # Initialize CSV file with UPDATED headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "step",
                "epoch",
                "train_loss",
                "eval_loss",
                "learning_rate",
                "examples_seen",
                "examples_per_sec",
                "no_truncation",
                "context_only_truncated",
                "assistant_partial_loss",
                "assistant_complete_loss",
                "skipped_no_labels"
            ])
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        print("=" * 80)
        print(f"Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs (every logging_steps)"""
        if logs is None:
            return control
        
        # Get collator statistics
        collator_stats = {}
        if self.collator is not None:
            stats = self.collator.get_stats()
            collator_stats = {
                'no_truncation': stats['no_truncation'],
                'context_only_truncated': stats['context_only_truncated'],
                'assistant_partial_loss': stats['assistant_partial_loss'],
                'assistant_complete_loss': stats['assistant_complete_loss'],
                'skipped_no_labels': stats['skipped_no_labels']
            }
        
        # Prepare log entry
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        step = state.global_step
        epoch = round(state.epoch, 4) if state.epoch is not None else 0
        train_loss = logs.get('loss', None)
        eval_loss = logs.get('eval_loss', None)
        learning_rate = logs.get('learning_rate', None)
        
        # Calculate examples seen
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
        examples_seen = step * effective_batch
        examples_per_sec = logs.get('train_samples_per_second', None)
        
        # Console output
        print(f"\n{'='*80}")
        print(f"Step: {step} | Epoch: {epoch:.4f}")
        if train_loss is not None:
            print(f"Train Loss: {train_loss:.4f}")
        if eval_loss is not None:
            print(f"Eval Loss: {eval_loss:.4f}")
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                print(f"🎯 New best eval loss!")
        if learning_rate is not None:
            print(f"Learning Rate: {learning_rate:.2e}")
        
        # Print collator stats every 50 steps
        if (state.global_step % 50 == 0) and self.collator is not None:
            self.collator.print_stats()
        print(f"{'='*80}")
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                step,
                epoch,
                train_loss if train_loss is not None else "",
                eval_loss if eval_loss is not None else "",
                learning_rate if learning_rate is not None else "",
                examples_seen,
                examples_per_sec if examples_per_sec is not None else "",
                collator_stats.get('no_truncation', ''),
                collator_stats.get('context_only_truncated', ''),
                collator_stats.get('assistant_partial_loss', ''),
                collator_stats.get('assistant_complete_loss', ''),
                collator_stats.get('skipped_no_labels', '')
            ])
        
        # Store for summary
        self.training_history.append({
            'step': step,
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'learning_rate': learning_rate
        })
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Print summary statistics when training ends"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {duration}")
        print(f"Total Steps: {state.global_step}")
        print(f"Total Epochs: {state.epoch:.4f}")
        print(f"Best Eval Loss: {self.best_eval_loss:.4f}")
        
        # Calculate training statistics
        train_losses = [h['train_loss'] for h in self.training_history if h['train_loss'] is not None]
        if train_losses:
            print(f"\nTraining Loss Statistics:")
            print(f"  Initial: {train_losses[0]:.4f}")
            print(f"  Final: {train_losses[-1]:.4f}")
            print(f"  Min: {min(train_losses):.4f}")
            print(f"  Max: {max(train_losses):.4f}")
            print(f"  Improvement: {train_losses[0] - train_losses[-1]:.4f}")
        
        eval_losses = [h['eval_loss'] for h in self.training_history if h['eval_loss'] is not None]
        if eval_losses:
            print(f"\nEval Loss Statistics:")
            print(f"  Best: {min(eval_losses):.4f}")
            print(f"  Final: {eval_losses[-1]:.4f}")
        
        # Get final collator statistics
        if self.collator is not None:
            print("\nFinal Collator Statistics:")
            self.collator.print_stats()
        
        print("=" * 80)
        print(f"Training log saved to: {self.log_file}")
        print(f"Best model saved to: {args.output_dir}/best_model")
        print("=" * 80 + "\n")
        
        # Save summary to JSON
        summary = {
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration.total_seconds(),
            'total_steps': state.global_step,
            'total_epochs': state.epoch,
            'best_eval_loss': self.best_eval_loss,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_eval_loss': eval_losses[-1] if eval_losses else None,
        }
        
        if self.collator is not None:
            summary['collator_stats'] = self.collator.get_stats()
        
        with open(f"{args.output_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return control


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
logging_callback = SFTLoggingCallback(
    log_file="./training_log.csv",
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
# DIAGNOSTIC: VERIFY FRESH MODEL
print("\n" + "="*80)
print("🔍 VERIFYING MODEL IS FRESH (before any training)")
print("="*80)

model.eval()
losses = []
for i in range(5):
    test_examples = [train_data[j] for j in range(i*2, (i+1)*2)]
    test_batch = data_collator(test_examples)
    test_batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in test_batch.items()}
    
    with torch.no_grad():
        outputs = model(**test_batch)
        losses.append(outputs.loss.item())
        print(f"Batch {i}: loss = {outputs.loss.item():.4f}")

avg_loss = sum(losses) / len(losses)
print(f"\nAverage starting loss: {avg_loss:.4f}")
print(f"Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.2f}")

if avg_loss < 0.8:
    print("❌ ERROR: Loss is suspiciously low (<0.8) - model may be contaminated!")
    print("   Consider restarting the kernel.")
elif avg_loss < 1.2:
    print("✅ Normal for LoRA! Base model is capable, LoRA adapters initialized to zero.")
else:
    print("✅ Normal starting loss for base model.")

print("="*80 + "\n")

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
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    token=hf_token
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
print(f"Model: {model_name}")
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

    # Save final model
    print("\nSaving final model...")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

    print("\n✅ Training complete!")
    print(f"Final model saved to: ./final_model")
    print(f"Best model saved to: {training_args.output_dir}")

#%%
start_training()