# imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------
# Configuration
# -----------------------
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
TOKENIZER_PATH = "./tokenizer"
LORA_PATH = None  # set to path if testing after training
DEVICE = "cuda"

PROMPTS = [
    "Explain why the sky is blue.",
    "Write a short polite customer service response to a refund request.",
    "What is the difference between supervised learning and reinforcement learning?",
]

MAX_NEW_TOKENS = 256

# -----------------------
# Load tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.pad_token = tokenizer.eos_token  # safety

# -----------------------
# Load model
# -----------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Resize embeddings (safe even if already correct)
model.resize_token_embeddings(len(tokenizer))

# Load LoRA if provided
if LORA_PATH is not None:
    model = PeftModel.from_pretrained(model, LORA_PATH)

model.eval()

# -----------------------
# Generation loop
# -----------------------
for i, user_prompt in enumerate(PROMPTS, 1):
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print("\n" + "=" * 60)
    print(f"[Prompt {i}] {user_prompt}")
    print("-" * 60)
    print(generated.strip())