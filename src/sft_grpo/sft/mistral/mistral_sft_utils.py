#imports
import torch


# function to format and tokenize data. placed her ia separate utility file to facilitate
def format_tokenize_with_spans(example, tokenizer, custom_system_msg=None):
    """
    Returns:
        {
            "input_ids": List[int],
            "labels": List[int],
        }
    """

    # 1. Render chat template (NO bos injection here)
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        default_system_message=custom_system_msg,
    )

    # 2. Tokenize (no truncation, no padding)
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )

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

    return {
        "input_ids": input_ids,
        "labels": labels,
    }



# collator
class TruncatingCollator:
    """
    Assistant-aware truncating collator with proper statistics tracking.
    
    Statistics are MUTUALLY EXCLUSIVE categories:
    - no_truncation: Sequence fits within max_length
    - context_only_truncated: Lost context but kept all assistant tokens
    - assistant_partial_loss: Lost some assistant tokens
    - assistant_complete_loss: Lost all assistant tokens (skipped)
    - skipped_no_labels: No assistant tokens in original sequence
    """

    def __init__(self, tokenizer, max_length: int, min_assistant_tokens: int = 50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_assistant_tokens = min_assistant_tokens
        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_token_id = -100

        # Statistics - MUTUALLY EXCLUSIVE CATEGORIES
        self.total_examples = 0
        self.no_truncation = 0
        self.context_only_truncated = 0
        self.assistant_partial_loss = 0
        self.assistant_complete_loss = 0
        self.skipped_no_labels = 0

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

            # CASE 1: Sequence fits - no truncation needed
            if seq_len <= self.max_length:
                self.no_truncation += 1
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_attention_mask.append([1] * len(input_ids))
                continue

            # Find assistant token positions (labeled tokens)
            assistant_positions = [
                i for i, label in enumerate(labels) if label != -100
            ]

            # CASE 2: No assistant tokens - skip
            if not assistant_positions:
                self.skipped_no_labels += 1
                continue

            first_asst = assistant_positions[0]
            last_asst = assistant_positions[-1]
            original_asst_len = last_asst - first_asst + 1

            # Truncation strategy: keep the END (most recent context + assistant response)
            # This is better than keeping the beginning for conversational data
            window_end = seq_len
            window_start = max(0, seq_len - self.max_length)
            
            # Calculate how many assistant tokens we keep after truncation
            kept_asst_positions = [
                i for i in assistant_positions 
                if window_start <= i < window_end
            ]
            
            kept_asst_len = len(kept_asst_positions)
            
            # CASE 3: All assistant tokens lost
            if kept_asst_len == 0:
                self.assistant_complete_loss += 1
                continue  # Skip this example
            
            # CASE 4: Some assistant tokens lost
            if kept_asst_len < original_asst_len:
                # Check if we kept enough to be useful
                if kept_asst_len < self.min_assistant_tokens:
                    self.assistant_complete_loss += 1
                    continue  # Skip - not enough assistant tokens
                
                self.assistant_partial_loss += 1
                batch_input_ids.append(input_ids[window_start:window_end])
                batch_labels.append(labels[window_start:window_end])
                batch_attention_mask.append([1] * (window_end - window_start))
                continue
            
            # CASE 5: All assistant tokens kept, only context lost
            self.context_only_truncated += 1
            batch_input_ids.append(input_ids[window_start:window_end])
            batch_labels.append(labels[window_start:window_end])
            batch_attention_mask.append([1] * (window_end - window_start))

        # Handle empty batch
        if not batch_input_ids:
            raise ValueError(
                f"\n⚠️  ENTIRE BATCH WAS SKIPPED!\n"
                f"  Total processed: {self.total_examples}\n"
                f"  No labels: {self.skipped_no_labels}\n"
                f"  Assistant lost: {self.assistant_complete_loss}\n"
                f"  This suggests your max_length is too small or data has issues."
            )

        # Pad to max length in batch
        max_len = max(len(seq) for seq in batch_input_ids)
        
        padded_input_ids = [
            seq + [self.pad_token_id] * (max_len - len(seq))
            for seq in batch_input_ids
        ]
        
        padded_labels = [
            seq + [self.label_pad_token_id] * (max_len - len(seq))
            for seq in batch_labels
        ]
        
        padded_attention_mask = [
            seq + [0] * (max_len - len(seq))
            for seq in batch_attention_mask
        ]
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

    def get_stats(self):
        """Return statistics as a dictionary with clear categories"""
        if self.total_examples == 0:
            return {
                'total_examples': 0,
                'no_truncation': 0,
                'context_only_truncated': 0,
                'assistant_partial_loss': 0,
                'assistant_complete_loss': 0,
                'skipped_no_labels': 0,
                'no_truncation_pct': 0.0,
                'context_only_truncated_pct': 0.0,
                'assistant_partial_loss_pct': 0.0,
                'assistant_complete_loss_pct': 0.0,
                'skipped_no_labels_pct': 0.0,
            }
        
        return {
            'total_examples': self.total_examples,
            'no_truncation': self.no_truncation,
            'context_only_truncated': self.context_only_truncated,
            'assistant_partial_loss': self.assistant_partial_loss,
            'assistant_complete_loss': self.assistant_complete_loss,
            'skipped_no_labels': self.skipped_no_labels,
            'no_truncation_pct': 100 * self.no_truncation / self.total_examples,
            'context_only_truncated_pct': 100 * self.context_only_truncated / self.total_examples,
            'assistant_partial_loss_pct': 100 * self.assistant_partial_loss / self.total_examples,
            'assistant_complete_loss_pct': 100 * self.assistant_complete_loss / self.total_examples,
            'skipped_no_labels_pct': 100 * self.skipped_no_labels / self.total_examples,
        }

    def reset_stats(self):
        """Reset all statistics counters"""
        self.total_examples = 0
        self.no_truncation = 0
        self.context_only_truncated = 0
        self.assistant_partial_loss = 0
        self.assistant_complete_loss = 0
        self.skipped_no_labels = 0

    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("COLLATOR STATISTICS")
        print("="*60)
        print(f"Total examples processed: {stats['total_examples']:,}")
        print(f"\nCategory breakdown (mutually exclusive):")
        print(f"  ✓ No truncation needed:    {stats['no_truncation']:6,} ({stats['no_truncation_pct']:5.1f}%)")
        print(f"  ✂ Context only truncated:   {stats['context_only_truncated']:6,} ({stats['context_only_truncated_pct']:5.1f}%)")
        print(f"  ⚠ Assistant partial loss:  {stats['assistant_partial_loss']:6,} ({stats['assistant_partial_loss_pct']:5.1f}%)")
        print(f"  ✗ Assistant complete loss: {stats['assistant_complete_loss']:6,} ({stats['assistant_complete_loss_pct']:5.1f}%)")
        print(f"  ∅ No labels (skipped):     {stats['skipped_no_labels']:6,} ({stats['skipped_no_labels_pct']:5.1f}%)")
        
        # Verify sum equals total
        sum_categories = (stats['no_truncation'] + stats['context_only_truncated'] + 
                         stats['assistant_partial_loss'] + stats['assistant_complete_loss'] + 
                         stats['skipped_no_labels'])
        print(f"\nVerification: {sum_categories:,} = {stats['total_examples']:,} ✓")
        print("="*60)