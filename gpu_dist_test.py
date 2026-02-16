import torch
import os
from transformers import AutoModelForCausalLM

# Hard-coded native Linux path
NATIVE_CACHE = "/home/brads/hf_cache"
os.makedirs(NATIVE_CACHE, exist_ok=True)

print(f"Inventory: {torch.cuda.device_count()} GPUs detected.")
print(f"Forcing download to native Linux path: {NATIVE_CACHE}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", 
        device_map="auto",
        cache_dir=NATIVE_CACHE  # This is the "sledgehammer" fix
    )
    print("\n[SUCCESS] Device Map created:")
    for k, v in model.hf_device_map.items():
        print(f"  Layer {k} -> GPU {v}")
except Exception as e:
    print(f"\n[ERROR] {e}")