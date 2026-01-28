#%%
import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import time

from sft_grpo.config import PACKAGE_ROOT, HF_TOKEN
from sft_grpo.sft.mistral_sft.mistral_sft_config import MISTRAL_SFT_ROOT, MODEL_NAME

#%%
# Load environment variables
print(HF_TOKEN)

if torch.cuda.is_available():
    print('cuda available')
    device = "auto"
elif torch.backends.mps.is_available():
    print('mps available')
    device = "mps"
else:
    print('cpu available')
    device = "cpu"


#%%
# Load BASE model

dtype = torch.float16 if device== "mps" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map=device,
    token=HF_TOKEN
)



# %%
import pandas as pd

# 3. Parameters
layer_indices = [0, 15, 31]
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
excel_data = {}

# 4. Analysis Loop
for i in layer_indices:
    print(f"\n--- Analyzing Layer {i} ---")
    layer_results = []
    
    # Define which projections to check in this layer
    # Attention: q_proj, k_proj, v_proj
    # MLP: gate_proj, up_proj, down_proj
    targets = {
        "Q (Attn)": model.model.layers[i].self_attn.q_proj,
        "K (Attn)": model.model.layers[i].self_attn.k_proj,
        "V (Attn)": model.model.layers[i].self_attn.v_proj,
        "Gate (MLP)": model.model.layers[i].mlp.gate_proj,
        "Up (MLP)": model.model.layers[i].mlp.up_proj,
        "Down (MLP)": model.model.layers[i].mlp.down_proj
    }

    for name, module in targets.items():
        start_time = time.time()
        W = module.weight.detach().to(torch.float32).cpu() # SVD always in float32
        
        S = torch.linalg.svdvals(W)
        s_sq = S**2
        cumulative_energy = torch.cumsum(s_sq, dim=0) / torch.sum(s_sq)
        
        row = {"Module": name, "Full Dim": f"{W.shape[0]}x{W.shape[1]}"}
        for t in thresholds:
            rank = torch.where(cumulative_energy >= t)[0][0].item() + 1
            row[f"{int(t*100)}%"] = rank
            
        layer_results.append(row)
        print(f"  {name} done ({time.time()-start_time:.1f}s)")

    df_layer = pd.DataFrame(layer_results)
    excel_data[f"Layer_{i}"] = df_layer
    print(df_layer.to_string(index=False))

# # 5. Export
# filename = "Mistral_Full_SVD.xlsx"
# with pd.ExcelWriter(filename) as writer:
#     for sheet, df in excel_data.items():
#         df.to_excel(writer, sheet_name=sheet, index=False)

# print(f"\nSaved to {filename}")
# %%
