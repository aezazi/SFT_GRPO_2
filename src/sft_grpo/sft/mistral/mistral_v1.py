#%%
# imports
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer, SFTConfig
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
    device_map=None,
    # attn_implementation="flash_attention_2",
    token=hf_token
)

if torch.cuda.is_available():
    print('cuda available')
    model = model.to("cuda")
    print(model.config._attn_implementation)


#%%
# load customozied tokenizer and verify special tokens and chat template
tokenizer = AutoTokenizer.from_pretrained("./mistral_7B_customized tokenizer")
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.pad_token)

#verify pad token is set eos token
print(f'pad token set to eos token: {tokenizer.pad_token == tokenizer.eos_token}')
print(f'pad token id set to eos token id: {tokenizer.pad_token_id == tokenizer.eos_token_id}')

# check if the model tokenizer already has a chat template
if tokenizer.chat_template is None:
    print('no chat template')
else:
    print(f'model already has a chat template:\n{(tokenizer.chat_template)}')

#%%
# THIS IS CRITICAL. ALWAYS RESIZE THE MODEL TO ACCOMMODATE THE EXTRA SPECIAL TOKENS
model.resize_token_embeddings(len(tokenizer))

# %%
# load data sets
train_data = load_from_disk('./dataset_ultrachat/train_dataset_tokenized')
eval_data = load_from_disk('./dataset_ultrachat/test_dataset_tokenized')

# %%
# custom data collator
from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch

@dataclass
class AssistantOnlyDataCollator:
    tokenizer: Any
    max_length: int = 2048

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

            # check if example is valid as per find_last_assistant_span(), if not, skip
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
# # test collator with subset of data
# test_data = train_data.select(range(5000))
# collator = AssistantOnlyDataCollator(tokenizer=tokenizer)
# test_batch = collator(test_data)

# # %%
# # Inspect collator output
# print(f"total: {collator.total_examples}")
# print(f"context truncated: {collator.context_truncated_examples}")
# print(f"assistant partially truncated: {collator.assistant_partially_truncated_examples}")
# print(f"assistant fully truncated: {collator.assistant_fully_truncated_examples}")
# print(f"skipped: {collator.skipped_examples}\n")

# print(f'type of input_ids: {type(test_batch["input_ids"][2])}')
# print(f'{test_batch["input_ids"][2].size()}\n')
# print(test_batch["input_ids"][2][:30])
# print(test_batch["input_ids"][2][-1000:])

# print(f'\ntype of attention_mask: {type(test_batch["attention_mask"][2])}')
# print(f'{test_batch["attention_mask"][2].size()}\n')
# print(test_batch["attention_mask"][2][:30])
# print(test_batch["attention_mask"][2][-1000:])

# print(f'\ntype of labels: {type(test_batch["labels"][2])}')
# print(f'{test_batch["labels"][2].size()}\n')
# print(test_batch["labels"][2][:30])
# print(test_batch["labels"][2][-1000:])

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
# Training Callback Implementation
import csv
from transformers import TrainerCallback, TrainerControl, TrainerState
from datetime import datetime
import json

class SFTLoggingCallback(TrainerCallback):
    """
    Custom callback for logging training metrics and collator statistics.
    """
    def __init__(self, log_file="training_log.csv"):
        self.log_file = log_file
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
        
        # Get collator statistics from the trainer's data_collator
        model = kwargs.get('model')
        train_dataloader = kwargs.get('train_dataloader')
        
        collator_stats = {}
        if train_dataloader is not None and hasattr(train_dataloader, 'collate_fn'):
            collator = train_dataloader.collate_fn
            if hasattr(collator, 'total_examples'):
                collator_stats = {
                    'context_truncated': collator.context_truncated_examples,
                    'assistant_partially_truncated': collator.assistant_partially_truncated_examples,
                    'assistant_fully_truncated': collator.assistant_fully_truncated_examples,
                    'skipped_examples': collator.skipped_examples
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
        # if train_loss is not None:
        #     print(f"Train Loss: {train_loss:.4f}")
        if eval_loss is not None:
            print(f"Eval Loss: {eval_loss:.4f}")
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                print(f"🎯 New best eval loss!")
        # if learning_rate is not None:
        #     print(f"Learning Rate: {learning_rate:.2e}")
        # print(f"Examples Seen: {examples_seen:,}")
        # if examples_per_sec is not None:
        #     print(f"Examples/sec: {examples_per_sec:.2f}")
        
        if (state.global_step % 25 ==0) and  collator_stats:
            print(f"\nCollator Statistics (cumulative):")
            print(f"  Context Truncated: {collator_stats['context_truncated']:,}")
            print(f"  Assistant Partially Truncated: {collator_stats['assistant_partially_truncated']:,}")
            print(f"  Assistant Fully Truncated: {collator_stats['assistant_fully_truncated']:,}")
            print(f"  Skipped Examples: {collator_stats['skipped_examples']:,}")
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
        train_dataloader = kwargs.get('train_dataloader')
        if train_dataloader is not None and hasattr(train_dataloader, 'collate_fn'):
            collator = train_dataloader.collate_fn
            if hasattr(collator, 'total_examples'):
                print(f"\nFinal Collator Statistics:")
                print(f"  Total Examples Processed: {collator.total_examples:,}")
                print(f"  Context Truncated: {collator.context_truncated_examples:,} ({100*collator.context_truncated_examples/collator.total_examples:.2f}%)")
                print(f"  Assistant Partially Truncated: {collator.assistant_partially_truncated_examples:,} ({100*collator.assistant_partially_truncated_examples/collator.total_examples:.2f}%)")
                print(f"  Assistant Fully Truncated: {collator.assistant_fully_truncated_examples:,} ({100*collator.assistant_fully_truncated_examples/collator.total_examples:.2f}%)")
                print(f"  Skipped Examples: {collator.skipped_examples:,} ({100*collator.skipped_examples/collator.total_examples:.2f}%)")
        
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
        
        with open(f"{args.output_dir}/training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return control


#%%
# Test Generation Callback (optional but recommended)
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
            
            print(f"{'='*80}\n")
            model.train()
        
        return control


#%%
# Updated Training Arguments optimized for single H200 GPU
training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,

    # Batch sizes - optimized for H200's 141GB HBM3e
    per_device_train_batch_size=4,  # Increased from 2 - H200 can handle larger batches
    gradient_accumulation_steps=8,  # Reduced - still effective batch = 16
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

    # Critical for custom collator
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    
    # Performance - optimized for single GPU
    dataloader_num_workers=8,   # H200 systems typically have good CPU
    group_by_length=False,      # Can enable if you want to group similar lengths
    
    # Memory optimization
    gradient_checkpointing=False,  # H200 has enough memory, keep disabled for speed
    ddp_find_unused_parameters=False,  # Not using DDP
)


#%%
# Instantiate callbacks
logging_callback = SFTLoggingCallback(log_file="./training_log.csv")

# Optional: Test generation during training
test_prompts = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "How do I make a good cup of coffee?"
]
generation_callback = GenerationTestCallback(
    tokenizer=tokenizer, 
    test_prompts=test_prompts,
    generation_steps=500  # Test generation every 500 steps
)


#%%
# Instantiate the Trainer
from transformers import Trainer

# Re-instantiate collator for training (to reset counters)
train_collator = AssistantOnlyDataCollator(tokenizer=tokenizer, max_length=2048)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=train_collator,
    callbacks=[logging_callback, generation_callback],  # Add both callbacks
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
# Start training
print("Starting training...")
trainer.train()

# Save final model
print("\nSaving final model...")
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# Save best model separately (already done by load_best_model_at_end, but explicit is good)
print("Best model saved to: ./checkpoints/best_model")

print("\n✅ Training complete!")

# %%
try:
    import flash_attn
    print(f"Flash Attention version: {flash_attn.__version__}")
    print("Flash Attention is installed")
except ImportError:
    print("Flash Attention is NOT installed")
# %%
