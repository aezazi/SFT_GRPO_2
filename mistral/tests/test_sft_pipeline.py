
import sys
from pathlib import Path



#%%
import pytest
import torch
from transformers import AutoTokenizer

# NEW: Import from your package
from mistral.utils import format_tokenize_with_spans, TruncatingCollator
from mistral.config import CUSTOM_TOKENIZER_V2_PATH

#%%
# --------------------
# Fixtures
# --------------------

@pytest.fixture(scope="session")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(
        str(CUSTOM_TOKENIZER_V2_PATH),
        local_files_only=True
    )
    return tok


@pytest.fixture
def raw_example():
    return {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 is 4."},
            {"role": "user", "content": "Say it again."},
            {"role": "assistant", "content": "It is 4."},
        ]
    }


@pytest.fixture
def collator(tokenizer):
    return TruncatingCollator(
        tokenizer=tokenizer,
        max_length=128,
        min_assistant_tokens=1,
    )

# --------------------
# Preprocessing tests
# --------------------

def test_preprocessing_supervises_eos(tokenizer, raw_example):
    out = format_tokenize_with_spans(raw_example, tokenizer)

    input_ids = out["input_ids"]
    labels = out["labels"]

    eos = tokenizer.eos_token_id

    eos_supervised = False
    for tok, lab in zip(input_ids, labels):
        if tok == eos and lab == eos:
            eos_supervised = True

    assert eos_supervised, "EOS inside assistant span must be supervised"


def test_preprocessing_masks_eos_outside_assistant(tokenizer):
    raw = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }

    out = format_tokenize_with_spans(raw, tokenizer)

    eos = tokenizer.eos_token_id

    for tok, lab in zip(out["input_ids"], out["labels"]):
        if tok == eos and lab == -100:
            break
    else:
        pytest.fail("Expected masked EOS outside assistant span")

# --------------------
# Collator tests
# --------------------

def test_collator_masks_padding_eos(collator):
    features = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4], "labels": [4]},
    ]

    batch = collator(features)

    pad_positions = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][pad_positions] == -100)


def test_collator_preserves_eos_labels(tokenizer, collator):
    eos = tokenizer.eos_token_id

    features = [{
        "input_ids": [10, 11, eos],
        "labels": [10, 11, eos],
    }]

    batch = collator(features)
    assert batch["labels"][0, -1].item() == eos

# --------------------
# End-to-end + golden
# --------------------

def test_e2e_only_assistant_tokens_supervised(
    tokenizer, collator, raw_example
):
    example = format_tokenize_with_spans(raw_example, tokenizer)
    batch = collator([example])

    labels = batch["labels"][0].tolist()

    supervised = [lab for lab in labels if lab != -100]
    assert supervised, "No supervised tokens found"


def test_golden_eos_is_last_supervised_token(tokenizer, collator):
    raw = {
        "messages": [
            {"role": "user", "content": "Finish the sentence"},
            {"role": "assistant", "content": "This is the end"},
        ]
    }

    example = format_tokenize_with_spans(raw, tokenizer)
    batch = collator([example])

    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()

    eos = tokenizer.eos_token_id

    supervised = [
        tok for tok, lab in zip(input_ids, labels)
        if lab != -100
    ]

    assert supervised[-1] == eos, (
        "GOLDEN FAILURE: EOS is not last supervised token"
    )
    