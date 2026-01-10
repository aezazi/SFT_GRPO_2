
#%%
import os
import sys
from pathlib import Path

#%%
from utils import TruncatingCollator, format_tokenize_with_spans



# %%
project_root = os.getenv("project_dir_path")
if project_root:
    full_path = Path(project_root) / "sft_mistral_7B"

print(full_path)

# %%
project_root = os.getenv("project_dir_path")
if project_root:
    full_path = Path(project_root) / "sft_mistral_7B"
    sys.path.append(str(full_path.resolve()))

for i in sys.path:
    print(i)

# %%
