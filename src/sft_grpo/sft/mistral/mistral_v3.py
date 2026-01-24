#%%
# imports
import sys
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, TrainerControl, TrainerState
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os
from datetime import datetime
import csv
import json

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


#%%
# configure PEFT LoRA 
# Basic
# lora_config = LoraConfig(
#     r=32,
#     lora_alpha=64,
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],
# )


# ==============================================================================
# 3. SURGICAL LORA CONFIG
# ==============================================================================

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    rank_pattern={
        "gate_proj": 256, "down_proj": 256, "up_proj": 128,
        "q_proj": 128, "k_proj": 32, "v_proj": 64,
    },
    alpha_pattern={
        "gate_proj": 512, "down_proj": 512, "up_proj": 256,
        "q_proj": 256, "k_proj": 64, "v_proj": 128,
    },
    use_rslora=True,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
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
    # learning_rate=2e-4,
    learning_rate=5e-5, # lr for surgical lora configs
    lr_scheduler_type="cosine",
    # warmup_ratio=0.03, 
    warmup_ratio=0.1, # for surgical lora configs

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
class SFTLoggingCallback(TrainerCallback):
    def __init__(self, log_file=None, collator=None):
        self.log_file = log_file
        self.collator = collator
        self.overwrite_csv_log = True # Set/modify via start_training()
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

        # Decide whether to wipe or append
        # We only wipe if overwrite is True AND we are at the very beginning (step 0)
        should_wipe = self.overwrite_csv_log and state.global_step == 0
        
        if should_wipe or not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "step", "epoch", "train_loss", "eval_loss", 
                    "learning_rate", "examples_seen", "examples_per_sec", 
                    "no_truncation", "context_only_truncated", "assistant_partial_loss", 
                    "assistant_complete_loss", "skipped_no_labels"
                ])
            mode_msg = "CLEARED and started fresh" if should_wipe else "CREATED new"
            print(f"📝 Log file {mode_msg} at: {self.log_file}")
        else:
            print(f"📈 Log file exists. Appending to: {self.log_file}")


        print("=" * 80)
        print(f"Training started/resumed at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        return control

    # on_log and on_train_end remain the same as your original script, 
    # as they already use 'a' (append) mode.

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
    Callback to test model generation and LOG results to a file.
    """
    def __init__(self, tokenizer, test_prompts, log_file, generation_steps=500):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generation_steps = generation_steps
        self.log_file = log_file
        self.overwrite_csv_log = False # Set/modify via start_training()
        
    def on_train_begin(self, args, state, control, **kwargs):
        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Determine if we should start a fresh log
        should_wipe = self.overwrite_csv_log and state.global_step == 0
        
        if should_wipe or not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write(f"Generation Test Log - Started: {datetime.now()}\n")
                f.write("="*80 + "\n")
            mode_msg = "CLEARED and started fresh" if should_wipe else "CREATED new"
            print(f"📝 Gen-log {mode_msg} at: {self.log_file}")
        else:
            print(f"📈 Gen-log exists. Appending to: {self.log_file}")
            
        return control

    def on_evaluate(self, args, state, control, model, **kwargs):
        if state.global_step % self.generation_steps == 0 and state.global_step > 0:
            # Prepare the log entry string
            output_header = f"\n\n{'='*80}\nGENERATION TEST AT STEP {state.global_step}\n{'='*80}\n"
            print(output_header) # Still print to console so you see it live
            
            model.eval()
            with open(self.log_file, "a") as f:
                f.write(output_header)
                
                for i, prompt in enumerate(self.test_prompts):
                    try:
                        messages = [{"role": "user", "content": prompt}]
                        input_text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
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
                        
                        # Format the result
                        result_str = (
                            f"\n--- Test Prompt {i+1} ---\n"
                            f"Prompt: {prompt}\n"
                            f"Generated: {generated_text}\n"
                            f"{'-' * 80}\n"
                        )
                        
                        print(result_str) # Live view
                        f.write(result_str) # Permanent record
                        
                    except Exception as e:
                        error_msg = f"\n--- Test Prompt {i+1} FAILED: {e} ---\n"
                        print(error_msg)
                        f.write(error_msg)
            
            model.train()
        return control


#%%
# Instantiate callbacks

# Ensure the logs directory exists before starting
(MISTRAL_SFT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

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
    log_file=str(MISTRAL_SFT_ROOT / "logs" / "generations.log"), # NEW
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
def start_training(checkpoint: str=None, overwrite_logs: bool=True):

    # 1. Update both callbacks with the overwrite preference
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, (SFTLoggingCallback, GenerationTestCallback)):
            callback.overwrite = overwrite_logs

    if checkpoint is not None:
        print(f"🚀 Resuming training from {checkpoint}")
        print("="*80 + "\n")
        # Define the path to your specific checkpoint folder
        checkpoint_path = str(MISTRAL_SFT_ROOT / "checkpoints" / checkpoint)
        # Pass the path to the resume_from_checkpoint argument
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
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

# comment to test goog push to github