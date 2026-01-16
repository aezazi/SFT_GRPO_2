
#%%
# imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sft_grpo.sft.mistral.mistral_config import MODEL_NAME, MISTRAL_SFT_ROOT, CUSTOM_TOKENIZER_V2_PATH
#%%
# paths
print(f'mddel name: {MODEL_NAME}')
print(f'MISTRAL_SFT_ROOT: {MISTRAL_SFT_ROOT}')
print(f'CUSTOM_TOKENIZER_V2_PATH: {CUSTOM_TOKENIZER_V2_PATH}')

#%%
class MistralChat:
    def __init__(self, system_message="You are a helpful assistant."):
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOKENIZER_V2_PATH)
        
        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load LoRA Adapters
        adapter_path = MISTRAL_SFT_ROOT / "experiments" / "final_sft_model_v1"
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()

        # Initialize history with the system message
        # Format: <|system|>\n{msg}</s>\n
        self.history = f"<|system|>\n{system_message}</s>\n"
        print("Model loaded. Type 'exit' or 'quit' to stop.\n")

    def chat(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            # 1. Append user message to history
            # Format: <|user|>\n{msg}</s>\n<|assistant|>\n
            self.history += f"<|user|>\n{user_input}</s>\n<|assistant|>\n"

            # 2. Tokenize the entire history
            inputs = self.tokenizer(self.history, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]

            # 3. Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # 4. Decode ONLY the new tokens
            new_tokens = output_ids[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # 5. Append the assistant's response (with tokens) to history for next turn
            # We append the raw decoded response + the closing </s> token
            self.history += f"{response}</s>\n"

            print(f"\nAssistant: {response}\n" + "-"*40)

#%%
if __name__ == "__main__":
    # If you have already merged the model, change the logic to load from the merged path
    # Otherwise, this uses the Base + LoRA setup as discussed.
    session = MistralChat()
    session.chat()