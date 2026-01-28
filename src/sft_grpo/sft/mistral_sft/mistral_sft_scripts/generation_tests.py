import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sft_grpo.sft.mistral_sft.mistral_sft_config import MODEL_NAME, MISTRAL_SFT_ROOT, CUSTOM_TOKENIZER_V2_PATH

# Configuration
ADAPTER_PATH = MISTRAL_SFT_ROOT / "experiments" / "final_sft_model_v3_1epoch_svd_r"
OUTPUT_FILE = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

TEST_CASES = [
    {
        "name": "Negative Constraint (Shrimp Test)",
        "turns": [
            "I want to cook a traditional seafood paella for 4 people. Can you give me a recipe? Please remember: I have a severe allergy to shrimp, so do not include shrimp or prawns in any part of the recipe."
        ]
    },
    {
        "name": "Multi-Turn Reasoning",
        "turns": [
            "Explain the concept of 'Gradient Norm' in neural network training. Use a metaphor involving a mountain.",
            "If my gradient norm suddenly spikes to 10.0 after being stable at 2.0, what does that tell me about my learning rate or my data batch?"
        ]
    },
    {
        "name": "JSON Formatting",
        "turns": [
            "I need a summary of the benefits of Python for data science. Provide your answer as a JSON object with three keys: 'libraries', 'community', and 'ease_of_use'. Each key should contain a one-sentence string."
        ]
    }
]

def run_evaluation():
    print("🚀 Loading model for final evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_V2_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH)).eval()

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"SFT EVALUATION REPORT - {datetime.now()}\n")
        f.write("="*50 + "\n\n")

        for case in TEST_CASES:
            f.write(f"TEST CASE: {case['name']}\n" + "-"*30 + "\n")
            # Using the exact same separator logic as the training data
            history = f"<|system|>\nYou are a helpful assistant.</s>\n"
            
            for i, turn in enumerate(case['turns']):
                history += f"<|user|>\n{turn}</s>\n<|assistant|>\n"
                inputs = tokenizer(history, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=512, 
                        temperature=0.7, 
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                f.write(f"Turn {i+1} User: {turn}\n")
                f.write(f"Turn {i+1} Assistant: {response}\n\n")
                history += f"{response}</s>\n"
            
            f.write("\n" + "="*50 + "\n")

    print(f"✅ Evaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()