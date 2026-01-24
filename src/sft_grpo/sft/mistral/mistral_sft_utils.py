#imports
import torch
import multiprocessing

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
import torch
import multiprocessing

class TruncatingCollator:
    def __init__(self, tokenizer, max_length: int, min_assistant_tokens: int = 50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_assistant_tokens = min_assistant_tokens
        self.pad_token_id = tokenizer.pad_token_id
        self.label_pad_token_id = -100

        # Statistics - Using multiprocessing Values for shared memory across workers
        self._total_examples = multiprocessing.Value('i', 0)
        self._no_truncation = multiprocessing.Value('i', 0)
        self._context_only_truncated = multiprocessing.Value('i', 0)
        self._assistant_partial_loss = multiprocessing.Value('i', 0)
        self._assistant_complete_loss = multiprocessing.Value('i', 0)
        self._skipped_no_labels = multiprocessing.Value('i', 0)

    def __call__(self, features):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        # Local counters for this specific batch in this worker
        l_total = 0
        l_no_trunc = 0
        l_ctx_trunc = 0
        l_asst_partial = 0
        l_asst_complete = 0
        l_skipped = 0

        for f in features:
            l_total += 1
            input_ids = f["input_ids"]
            labels = f["labels"]
            
            if not isinstance(input_ids, list): input_ids = input_ids.tolist()
            if not isinstance(labels, list): labels = labels.tolist()
            
            seq_len = len(input_ids)
            
            # if the sequence length is less than max_length, apppend all sequence tokens and labels to the batch. append 1s to the attention mask equal to the length of the sequence. increment the counter for no truncation. continue to the next example
            if seq_len <= self.max_length:
                l_no_trunc += 1
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_attention_mask.append([1] * len(input_ids))
                continue
            
            # we enter the blocks below if sequence length is > max_length, so we need to truncate. 
            # first identify the index positions for assistant responses. we do this using the labels we creates with the format_tokenize_with_span() function above
            assistant_positions = [i for i, label in enumerate(labels) if label != -100]

            # if there are no assistant responses in the sequence.increment the counter for skipped examples. skip the example and continue the next
            if not assistant_positions:
                l_skipped += 1
                continue
            
            # compute the count of all assistant responses in the sequence
            original_asst_count = len(assistant_positions)

            # compute the starting and ending position of the eligible window for the entire sequence starting from the back, so the window end is the last position. window start is either at position 0 (if seq_len is <= max_length) or seq_len - self.max_length ( if seq_len > max_length)
            window_end = seq_len
            window_start = max(0, seq_len - self.max_length)
            
            # compute the positions of the assistant tokens in the window_start to window_end span.
            kept_asst_positions = [i for i in assistant_positions if window_start <= i < window_end]
            kept_asst_count = len(kept_asst_positions)
            
            
            # check if kept_assisant_positions is less than all all assistant tokens in the sequnce (original_asst_count). if only some of the assistant response is kept, check if it falls within the minimum 50 tokens requirement. If kept assistant response is less than 50 tokens, skip the example and increment the counter for assistant response completely truncated. 
            if kept_asst_count < original_asst_count:
                if kept_asst_count < self.min_assistant_tokens:
                    l_asst_complete += 1
                    continue

                # If the assistant response is kept and its length is greater than 50 but less than max_length, increment the counter for assistant response partially truncated
                l_asst_partial += 1
            
            # kept_asst_count is not less than original_asst_count. we kept all assistant responses, no trunctaion. increment the no truncation counter
            else:
                l_ctx_trunc += 1
            
            batch_input_ids.append(input_ids[window_start:window_end])
            batch_labels.append(labels[window_start:window_end])
            batch_attention_mask.append([1] * (window_end - window_start))

        # Atomic updates to shared memory
        with self._total_examples.get_lock(): self._total_examples.value += l_total
        with self._no_truncation.get_lock(): self._no_truncation.value += l_no_trunc
        with self._context_only_truncated.get_lock(): self._context_only_truncated.value += l_ctx_trunc
        with self._assistant_partial_loss.get_lock(): self._assistant_partial_loss.value += l_asst_partial
        with self._assistant_complete_loss.get_lock(): self._assistant_complete_loss.value += l_asst_complete
        with self._skipped_no_labels.get_lock(): self._skipped_no_labels.value += l_skipped

        # (Rest of your padding logic remains the same...)
        if not batch_input_ids:
             # Custom error handling here
             pass

        max_len = max(len(seq) for seq in batch_input_ids)
        padded_input_ids = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in batch_input_ids]
        padded_labels = [seq + [self.label_pad_token_id] * (max_len - len(seq)) for seq in batch_labels]
        padded_attention_mask = [seq + [0] * (max_len - len(seq)) for seq in batch_attention_mask]
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

    def get_stats(self):
        """Return statistics as a dictionary using .value for shared memory"""
        total = self._total_examples.value
        if total == 0:
            return {
                'total_examples': 0, 'no_truncation': 0, 'context_only_truncated': 0,
                'assistant_partial_loss': 0, 'assistant_complete_loss': 0, 'skipped_no_labels': 0,
                'no_truncation_pct': 0.0, 'context_only_truncated_pct': 0.0,
                'assistant_partial_loss_pct': 0.0, 'assistant_complete_loss_pct': 0.0,
                'skipped_no_labels_pct': 0.0,
            }
        
        return {
            'total_examples': total,
            'no_truncation': self._no_truncation.value,
            'context_only_truncated': self._context_only_truncated.value,
            'assistant_partial_loss': self._assistant_partial_loss.value,
            'assistant_complete_loss': self._assistant_complete_loss.value,
            'skipped_no_labels': self._skipped_no_labels.value,
            'no_truncation_pct': 100 * self._no_truncation.value / total,
            'context_only_truncated_pct': 100 * self._context_only_truncated.value / total,
            'assistant_partial_loss_pct': 100 * self._assistant_partial_loss.value / total,
            'assistant_complete_loss_pct': 100 * self._assistant_complete_loss.value / total,
            'skipped_no_labels_pct': 100 * self._skipped_no_labels.value / total,
        }

    def reset_stats(self):
        """Reset all shared memory counters"""
        with self._total_examples.get_lock(): self._total_examples.value = 0
        with self._no_truncation.get_lock(): self._no_truncation.value = 0
        with self._context_only_truncated.get_lock(): self._context_only_truncated.value = 0
        with self._assistant_partial_loss.get_lock(): self._assistant_partial_loss.value = 0
        with self._assistant_complete_loss.get_lock(): self._assistant_complete_loss.value = 0
        with self._skipped_no_labels.get_lock(): self._skipped_no_labels.value = 0

    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("COLLATOR STATISTICS (SHARED MEMORY)")
        print("="*60)
        print(f"Total examples processed: {stats['total_examples']:,}")
        print(f"\nCategory breakdown (mutually exclusive):")
        print(f"  ✓ No truncation needed:    {stats['no_truncation']:6,} ({stats['no_truncation_pct']:5.1f}%)")
        print(f"  ✂ Context only truncated:   {stats['context_only_truncated']:6,} ({stats['context_only_truncated_pct']:5.1f}%)")
        print(f"  ⚠ Assistant partial loss:  {stats['assistant_partial_loss']:6,} ({stats['assistant_partial_loss_pct']:5.1f}%)")
        print(f"  ✗ Assistant complete loss: {stats['assistant_complete_loss']:6,} ({stats['assistant_complete_loss_pct']:5.1f}%)")
        print(f"  ∅ No labels (skipped):     {stats['skipped_no_labels']:6,} ({stats['skipped_no_labels_pct']:5.1f}%)")
        print("="*60)