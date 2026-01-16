#%%
# imports
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os
from dotenv import load_dotenv
import pprint


#%%
# Load environment variables
load_dotenv()
hf_token = os.getenv("hf_token")
print(hf_token)

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
# Load base model
model_name = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    token=hf_token
)

if torch.cuda.is_available():
    print('cuda available')
    # model = model.to("cuda")
    print(model.config._attn_implementation)


#%%
# load customized tokenizer and verify special tokens and chat template
tokenizer = AutoTokenizer.from_pretrained("./mistral_7B_customized tokenizer_v2")
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
train_data = load_from_disk('./dataset_ultrachat/train_dataset_tokenized_v2')
eval_data = load_from_disk('./dataset_ultrachat/test_dataset_tokenized_v2')



# %%
# configure PEFT LoRA
# since I am adding peft and lora to the model here, no need to set any arguments trainer object later
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

    # packing=False,
    # packing_strategy="bfd",

    remove_unused_columns=False,
    
    # Batch sizes - optimized for H200's 141GB HBM3e
    per_device_train_batch_size=2,  # Increased from 2 - H200 can handle larger batches
    gradient_accumulation_steps=16,  # Reduced - still effective batch = 16
    per_device_eval_batch_size=4,   # Match training batch size

    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # Training duration
    num_train_epochs=1,
    
    # Logging and evaluation
    logging_steps=10,           # Log every 50 steps (every ~800 examples)
    eval_steps=50,             # Evaluate every 500 steps (every ~8000 examples)
    save_steps=500,             # Save checkpoint every 500 steps
    eval_strategy="steps",
    save_strategy="steps",
    
    # Model saving
    save_total_limit=3,         # Keep only 3 most recent checkpoints
    load_best_model_at_end=True,  # Load best model based on eval_loss
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Precision - optimal for H200
    bf16=True,
    fp16=False,
    bf16_full_eval=True,        # Use bf16 for evaluation too

    # Optimizer - fused for better H200 performance
    optim="adamw_torch_fused",
    max_grad_norm=1.0,

    # Reporting
    report_to="none",  # change to "wandb" if you want to use weights & biases
    run_name="sft-mistral-7b-ultrachat",

    dataloader_pin_memory=True,
    
    # Performance - optimized for single GPU
    dataloader_num_workers=8,   # H200 systems typically have good CPU
    group_by_length=False,      # Can enable if you want to group similar lengths
    
    # Memory optimization
    gradient_checkpointing=False,  # H200 has enough memory, keep disabled for speed
    ddp_find_unused_parameters=False,  # Not using DDP
)

#%%
# create data collator
class TruncatingCollator:
    """
    Assistant-aware truncating collator for SFT.

    - Truncates from the LEFT (keeps recent context)
    - Allows partial assistant truncation if necessary
    - Tracks detailed truncation statistics
    - Generates attention masks
    """

    def __init__(self, tokenizer, max_length: int, min_assistant_tokens: int = 50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_assistant_tokens = min_assistant_tokens  # Minimum assistant tokens to keep
        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_token_id = -100

        # Statistics (cumulative)
        self.total_examples = 0
        self.context_truncated_examples = 0
        self.assistant_partially_truncated_examples = 0
        self.assistant_fully_truncated_examples = 0
        self.skipped_examples = 0

    def __call__(self, features):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            self.total_examples += 1

            # Ensure input_ids and labels are lists
            input_ids = f["input_ids"]
            if not isinstance(input_ids, list):
                input_ids = input_ids.tolist()
            
            labels = f["labels"]
            if not isinstance(labels, list):
                labels = labels.tolist()
            
            seq_len = len(input_ids)

            # If already fits, keep as-is
            if seq_len <= self.max_length:
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_attention_mask.append([1] * len(input_ids))
                continue

            # Identify assistant token positions
            assistant_positions = [
                i for i, label in enumerate(labels) if label != -100
            ]

            # No assistant tokens → skip
            if not assistant_positions:
                self.skipped_examples += 1
                continue

            first_asst = assistant_positions[0]
            last_asst = assistant_positions[-1]
            assistant_len = last_asst - first_asst + 1

            # NEW: If assistant is too long, truncate it but keep minimum tokens
            if assistant_len > self.max_length:
                # Keep at least min_assistant_tokens from the END of assistant response
                if assistant_len >= self.min_assistant_tokens:
                    # Truncate from the end to fit max_length
                    window_end = last_asst + 1
                    window_start = window_end - self.max_length
                    
                    self.assistant_fully_truncated_examples += 1
                    self.assistant_partially_truncated_examples += 1
                    self.context_truncated_examples += 1
                    
                    batch_input_ids.append(input_ids[window_start:window_end])
                    batch_labels.append(labels[window_start:window_end])
                    batch_attention_mask.append([1] * self.max_length)
                    continue
                else:
                    # Assistant is too long and we can't keep minimum → skip
                    self.assistant_fully_truncated_examples += 1
                    self.skipped_examples += 1
                    continue

            # Normal truncation: Keep entire assistant + as much context as fits
            window_end = seq_len
            window_start = max(0, last_asst + 1 - self.max_length)

            # Detect truncation types
            if window_start > 0:
                self.context_truncated_examples += 1
                if window_start > first_asst:
                    self.assistant_partially_truncated_examples += 1

            # Slice
            truncated_length = window_end - window_start
            batch_input_ids.append(input_ids[window_start:window_end])
            batch_labels.append(labels[window_start:window_end])
            batch_attention_mask.append([1] * truncated_length)

        # Handle empty batch - instead of raising error, log warning and return a dummy batch
        if not batch_input_ids:
            print(f"\n⚠️  WARNING: Entire batch was skipped (processed: {self.total_examples}, skipped: {self.skipped_examples})")
            # Create a minimal dummy batch to avoid crashing
            # This will have zero loss contribution
            batch_input_ids.append([self.pad_token_id] * self.min_assistant_tokens)
            batch_labels.append([self.label_pad_token_id] * self.min_assistant_tokens)
            batch_attention_mask.append([0] * self.min_assistant_tokens)

        # Manual padding
        max_len = max(len(seq) for seq in batch_input_ids)
        
        # Pad input_ids
        padded_input_ids = []
        for seq in batch_input_ids:
            padding_length = max_len - len(seq)
            padded_seq = seq + [self.pad_token_id] * padding_length
            padded_input_ids.append(padded_seq)
        
        # Pad labels
        padded_labels = []
        for seq in batch_labels:
            padding_length = max_len - len(seq)
            padded_seq = seq + [self.label_pad_token_id] * padding_length
            padded_labels.append(padded_seq)
        
        # Pad attention_mask
        padded_attention_mask = []
        for seq in batch_attention_mask:
            padding_length = max_len - len(seq)
            padded_seq = seq + [0] * padding_length
            padded_attention_mask.append(padded_seq)
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

    def get_stats(self):
        """Return current statistics as a dictionary"""
        if self.total_examples == 0:
            return {
                'total_examples': 0,
                'context_truncated': 0,
                'assistant_partially_truncated': 0,
                'assistant_fully_truncated': 0,
                'skipped_examples': 0,
                'context_truncated_pct': 0.0,
                'assistant_partially_truncated_pct': 0.0,
                'assistant_fully_truncated_pct': 0.0,
                'skipped_examples_pct': 0.0,
            }
        
        return {
            'total_examples': self.total_examples,
            'context_truncated': self.context_truncated_examples,
            'assistant_partially_truncated': self.assistant_partially_truncated_examples,
            'assistant_fully_truncated': self.assistant_fully_truncated_examples,
            'skipped_examples': self.skipped_examples,
            'context_truncated_pct': 100 * self.context_truncated_examples / self.total_examples,
            'assistant_partially_truncated_pct': 100 * self.assistant_partially_truncated_examples / self.total_examples,
            'assistant_fully_truncated_pct': 100 * self.assistant_fully_truncated_examples / self.total_examples,
            'skipped_examples_pct': 100 * self.skipped_examples / self.total_examples,
        }

    def reset_stats(self):
        """Reset all statistics counters"""
        self.total_examples = 0
        self.context_truncated_examples = 0
        self.assistant_partially_truncated_examples = 0
        self.assistant_fully_truncated_examples = 0
        self.skipped_examples = 0


data_collator = TruncatingCollator(tokenizer, max_length=3072)

#%%
# Test the collator before training
print("\n" + "="*80)
print("TESTING COLLATOR")
print("="*80)

# Test with a small batch
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
print("\nVerify all tensors are same length:")
print(f"  All input_ids same length: {collated['input_ids'].shape[1]}")
print(f"  All labels same length: {collated['labels'].shape[1]}")
print(f"  All attention_mask same length: {collated['attention_mask'].shape[1]}")
print("\nInitial collator stats:")
print(data_collator.get_stats())
print("="*80 + "\n")

#%%
# callback
# Training Callback Implementation
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
        self.collator = collator  # Store reference to collator
        self.start_time = None
        self.best_eval_loss = float('inf')
        self.training_history = []
        
        # Initialize CSV file with headers
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
                "context_truncated",
                "assistant_partially_truncated",
                "assistant_fully_truncated",
                "skipped_examples"
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
        
        # Get collator statistics directly from stored reference
        collator_stats = {}
        if self.collator is not None:
            stats = self.collator.get_stats()
            collator_stats = {
                'context_truncated': stats['context_truncated'],
                'assistant_partially_truncated': stats['assistant_partially_truncated'],
                'assistant_fully_truncated': stats['assistant_fully_truncated'],
                'skipped_examples': stats['skipped_examples']
            }
        
        # Prepare log entry
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        step = state.global_step
        epoch = round(state.epoch, 4) if state.epoch is not None else 0
        train_loss = logs.get('loss', None)
        eval_loss = logs.get('eval_loss', None)
        learning_rate = logs.get('learning_rate', None)
        
        # Calculate examples seen (step * effective_batch_size)
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
        
        # Print collator stats periodically
        if (state.global_step % 25 == 0) and collator_stats and self.collator is not None:
            stats = self.collator.get_stats()
            print(f"\nCollator Statistics (cumulative):")
            print(f"  Total Examples: {stats['total_examples']:,}")
            print(f"  Context Truncated: {stats['context_truncated']:,} ({stats['context_truncated_pct']:.2f}%)")
            print(f"  Assistant Partially Truncated: {stats['assistant_partially_truncated']:,} ({stats['assistant_partially_truncated_pct']:.2f}%)")
            print(f"  Assistant Fully Truncated: {stats['assistant_fully_truncated']:,} ({stats['assistant_fully_truncated_pct']:.2f}%)")
            print(f"  Skipped Examples: {stats['skipped_examples']:,} ({stats['skipped_examples_pct']:.2f}%)")
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
                collator_stats.get('context_truncated', ''),
                collator_stats.get('assistant_partially_truncated', ''),
                collator_stats.get('assistant_fully_truncated', ''),
                collator_stats.get('skipped_examples', '')
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
            stats = self.collator.get_stats()
            print(f"\nFinal Collator Statistics:")
            print(f"  Total Examples Processed: {stats['total_examples']:,}")
            print(f"  Context Truncated: {stats['context_truncated']:,} ({stats['context_truncated_pct']:.2f}%)")
            print(f"  Assistant Partially Truncated: {stats['assistant_partially_truncated']:,} ({stats['assistant_partially_truncated_pct']:.2f}%)")
            print(f"  Assistant Fully Truncated: {stats['assistant_fully_truncated']:,} ({stats['assistant_fully_truncated_pct']:.2f}%)")
            print(f"  Skipped Examples: {stats['skipped_examples']:,} ({stats['skipped_examples_pct']:.2f}%)")
        
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
        
        # Add collator stats to summary
        if self.collator is not None:
            summary['collator_stats'] = self.collator.get_stats()
        
        with open(f"{args.output_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return control
    

#%%
# create generation callback  

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
                    # Format as chat
                    messages = [{"role": "user", "content": prompt}]
                    input_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)
                    
                    # Generate
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
    collator=data_collator  # Pass the collator reference
)

# Optional: Test generation during training
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

# Updated check for LoRA
if avg_loss < 0.5:
    print("❌ ERROR: Loss is suspiciously low (<0.5)")
elif avg_loss < 1.2:
    print("✅ This is normal for LoRA! Base model is already capable.")
    print("   LoRA adapters are initialized to contribute zero.")
    print("   Training will fine-tune from this strong baseline.")
else:
    print("✅ Normal starting loss for base model.")

print("="*80 + "\n")
#%%
#%%
# DIAGNOSTIC: Check labels in a batch
print("\n" + "="*80)
print("DIAGNOSTIC: Checking batch labels")
print("="*80)

test_batch = [train_data[i] for i in range(4)]
collated = data_collator(test_batch)

print(f"Batch shape: {collated['input_ids'].shape}")
print(f"Labels shape: {collated['labels'].shape}")

# Count non-ignored labels
for i in range(len(collated['labels'])):
    labels = collated['labels'][i]
    non_ignored = (labels != -100).sum().item()
    total = labels.shape[0]
    print(f"Example {i}: {non_ignored}/{total} tokens have labels ({100*non_ignored/total:.1f}%)")
    
    # Show first few labels
    print(f"  First 20 labels: {labels[:20].tolist()}")
    print(f"  Last 20 labels: {labels[-20:].tolist()}")

# Calculate what percentage of the batch has real labels
total_tokens = collated['labels'].numel()
labeled_tokens = (collated['labels'] != -100).sum().item()
print(f"\nOverall: {labeled_tokens}/{total_tokens} tokens have labels ({100*labeled_tokens/total_tokens:.1f}%)")
print("="*80 + "\n")
#%%
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
print(f"Total optimization steps: {trainer.state.max_steps if hasattr(trainer.state, 'max_steps') else 'calculating...'}")
print(f"Logging every: {training_args.logging_steps} steps")
print(f"Evaluating every: {training_args.eval_steps} steps")
print(f"Saving checkpoints every: {training_args.save_steps} steps")
print(f"Output directory: {training_args.output_dir}")
print("="*80 + "\n")


#%%
# function to start training

def start_training():
    print("Starting training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

    # Save best model separately (already done by load_best_model_at_end, but explicit is good)
    print("Best model saved to: ./checkpoints/best_model")

    print("\n✅ Training complete!")

#%%
start_training()



#%%
# ====================  DIAGNOTISTICS TO BE RUN WITHOUT STARTING TRAINING ON GPU ========================
# ======================================================================================================
# import torch
# import os
# import subprocess

# print("="*80)
# print("SYSTEM DIAGNOSTICS")
# print("="*80)

# # GPU Info
# if torch.cuda.is_available():
#     print(f"\n🖥️  GPU Available: {torch.cuda.get_device_name(0)}")
#     print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#     print(f"Current Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
#     print(f"Current Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
# else:
#     print("❌ No GPU available")

# # CPU Info
# print(f"\n💻 CPU Cores: {os.cpu_count()}")

# # PyTorch version
# print(f"\n📦 PyTorch Version: {torch.__version__}")
# print(f"CUDA Version: {torch.version.cuda}")

# # Check nvidia-smi
# try:
#     result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
#     print("\n" + result.stdout)
# except:
#     print("Could not run nvidia-smi")

# print("="*80)
# # %%
# #%%
# print("\n" + "="*80)
# print("COLLATOR DIAGNOSTICS")
# print("="*80)

# # Reset collator stats
# data_collator.reset_stats()

# # Simulate what happens during training - process 100 batches
# import time

# batch_times = []
# for i in range(100):
#     # Get a batch like the trainer does
#     indices = list(range(i*2, (i+1)*2))  # batch_size=2
#     batch_examples = [train_data[idx % len(train_data)] for idx in indices]
    
#     start = time.time()
#     batch = data_collator(batch_examples)
#     batch_time = time.time() - start
#     batch_times.append(batch_time)
    
#     if i % 25 == 24:
#         print(f"\nAfter {i+1} batches:")
#         stats = data_collator.get_stats()
#         print(f"  Total examples processed: {stats['total_examples']}")
#         print(f"  Context truncated: {stats['context_truncated']} ({stats['context_truncated_pct']:.1f}%)")
#         print(f"  Assistant fully truncated: {stats['assistant_fully_truncated']} ({stats['assistant_fully_truncated_pct']:.1f}%)")
#         print(f"  Skipped: {stats['skipped_examples']} ({stats['skipped_examples_pct']:.1f}%)")
#         print(f"  Avg batch shape: {batch['input_ids'].shape}")
#         print(f"  Avg collation time: {sum(batch_times[-25:]) / 25 * 1000:.1f}ms")

# print(f"\n📊 Final Collator Stats:")
# final_stats = data_collator.get_stats()
# for key, val in final_stats.items():
#     print(f"  {key}: {val}")

# print(f"\n⏱️  Collation Performance:")
# print(f"  Mean time: {sum(batch_times) / len(batch_times) * 1000:.1f}ms")
# print(f"  Min time: {min(batch_times) * 1000:.1f}ms")
# print(f"  Max time: {max(batch_times) * 1000:.1f}ms")

# print("="*80)

# #%%
# #%%
# import time
# import torch

# print("\n" + "="*80)
# print("FORWARD PASS SPEED TEST")
# print("="*80)

# # Create a test batch
# test_batch_examples = [train_data[i] for i in range(2)]
# test_batch = data_collator(test_batch_examples)

# # Move to GPU
# test_batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in test_batch.items()}

# print(f"Batch shape: {test_batch['input_ids'].shape}")
# print(f"Sequence length: {test_batch['input_ids'].shape[1]}")

# # Warm up
# with torch.no_grad():
#     _ = model(**test_batch)

# torch.cuda.synchronize()

# # Time forward passes
# forward_times = []
# for i in range(10):
#     torch.cuda.synchronize()
#     start = time.time()
    
#     with torch.no_grad():
#         outputs = model(**test_batch)
    
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     forward_times.append(elapsed)
    
#     print(f"Forward pass {i+1}: {elapsed*1000:.1f}ms, loss: {outputs.loss.item():.4f}")

# print(f"\n⏱️  Forward Pass Performance:")
# print(f"  Mean: {sum(forward_times) / len(forward_times) * 1000:.1f}ms")
# print(f"  Min: {min(forward_times) * 1000:.1f}ms")
# print(f"  Max: {max(forward_times) * 1000:.1f}ms")

# print("="*80)

# #%%
# #%%
# import time
# import torch

# print("\n" + "="*80)
# print("TRAINING STEP SPEED TEST")
# print("="*80)

# # Put model in training mode
# model.train()

# # Create a test batch
# test_batch_examples = [train_data[i] for i in range(2)]
# test_batch = data_collator(test_batch_examples)
# test_batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in test_batch.items()}

# print(f"Testing with batch_size=2, sequence_length={test_batch['input_ids'].shape[1]}")

# # Create optimizer (simplified)
# from torch.optim import AdamW
# test_optimizer = AdamW(model.parameters(), lr=2e-4)

# # Time training steps
# step_times = []
# for i in range(10):
#     torch.cuda.synchronize()
#     start = time.time()
    
#     # Forward
#     outputs = model(**test_batch)
#     loss = outputs.loss
    
#     # Backward
#     loss.backward()
    
#     # Optimizer step
#     test_optimizer.step()
#     test_optimizer.zero_grad()
    
#     torch.cuda.synchronize()
#     elapsed = time.time() - start
#     step_times.append(elapsed)
    
#     print(f"Step {i+1}: {elapsed*1000:.0f}ms, loss: {loss.item():.4f}")

# print(f"\n⏱️  Training Step Performance:")
# print(f"  Mean: {sum(step_times) / len(step_times) * 1000:.0f}ms")
# print(f"  Min: {min(step_times) * 1000:.0f}ms")
# print(f"  Max: {max(step_times) * 1000:.0f}ms")

# print(f"\n🔍 With gradient_accumulation_steps=16:")
# print(f"  Expected time per optimization step: {sum(step_times) / len(step_times) * 16:.1f}s")
# print(f"  Your actual time: ~7s")
# print(f"  Ratio: {7 / (sum(step_times) / len(step_times) * 16):.2f}x")

# print("="*80)

# #%%
# #%%
# print("\n" + "="*80)
# print("ACTUAL STARTING LOSS CHECK")
# print("="*80)

# model.eval()

# losses = []
# for i in range(10):
#     test_batch_examples = [train_data[j] for j in range(i*2, (i+1)*2)]
#     batch = data_collator(test_batch_examples)
#     batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in batch.items()}
    
#     with torch.no_grad():
#         outputs = model(**batch)
#         losses.append(outputs.loss.item())
        
#         # Show labeled token percentage
#         labeled = (batch['labels'] != -100).sum().item()
#         total = batch['labels'].numel()
        
#         print(f"Batch {i}: loss={outputs.loss.item():.4f}, labeled tokens={labeled}/{total} ({100*labeled/total:.1f}%)")

# avg_loss = sum(losses) / len(losses)
# print(f"\n📊 Average starting loss: {avg_loss:.4f}")
# print(f"📊 Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.2f}")

# print("="*80)

# #%%
# #%%
# print("\n" + "="*80)
# print("DATASET STATISTICS")
# print("="*80)

# import numpy as np

# # Sample 1000 examples
# sample_sizes = []
# label_percentages = []

# for i in range(0, min(1000, len(train_data))):
#     example = train_data[i]
#     input_ids = example['input_ids']
#     labels = example['labels']
    
#     sample_sizes.append(len(input_ids))
#     labeled = sum(1 for l in labels if l != -100)
#     label_percentages.append(100 * labeled / len(labels))

# print(f"Sequence Length Statistics (1000 samples):")
# print(f"  Mean: {np.mean(sample_sizes):.0f}")
# print(f"  Median: {np.median(sample_sizes):.0f}")
# print(f"  Min: {np.min(sample_sizes):.0f}")
# print(f"  Max: {np.max(sample_sizes):.0f}")
# print(f"  90th percentile: {np.percentile(sample_sizes, 90):.0f}")
# print(f"  95th percentile: {np.percentile(sample_sizes, 95):.0f}")
# print(f"  99th percentile: {np.percentile(sample_sizes, 99):.0f}")

# print(f"\nLabel Percentage Statistics:")
# print(f"  Mean: {np.mean(label_percentages):.1f}%")
# print(f"  Median: {np.median(label_percentages):.1f}%")

# print(f"\n📊 Sequences longer than max_length (2048): {sum(1 for s in sample_sizes if s > 2048)}/1000 ({100*sum(1 for s in sample_sizes if s > 2048)/1000:.1f}%)")

# print("="*80)