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

            if seq_len <= self.max_length:
                l_no_trunc += 1
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_attention_mask.append([1] * len(input_ids))
                continue

            assistant_positions = [i for i, label in enumerate(labels) if label != -100]

            if not assistant_positions:
                l_skipped += 1
                continue

            first_asst = assistant_positions[0]
            last_asst = assistant_positions[-1]
            original_asst_len = last_asst - first_asst + 1

            window_end = seq_len
            window_start = max(0, seq_len - self.max_length)
            
            kept_asst_positions = [i for i in assistant_positions if window_start <= i < window_end]
            kept_asst_len = len(kept_asst_positions)
            
            if kept_asst_len == 0:
                l_asst_complete += 1
                continue 
            
            if kept_asst_len < original_asst_len:
                if kept_asst_len < self.min_assistant_tokens:
                    l_asst_complete += 1
                    continue
                
                l_asst_partial += 1
                batch_input_ids.append(input_ids[window_start:window_end])
                batch_labels.append(labels[window_start:window_end])
                batch_attention_mask.append([1] * (window_end - window_start))
                continue
            
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
        total = self._total_examples.value
        if total == 0: return {'total_examples': 0} # return empty dict or zeros
        
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