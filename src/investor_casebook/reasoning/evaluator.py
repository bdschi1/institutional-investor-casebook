import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class InvestorEvaluator:
    def __init__(self, model_id="gpustack/llama-3-8b-instruct"):
        self.device_count = torch.cuda.device_count()
        print(f"Detected {self.device_count} GPUs. Distributing model...")
        
        # The 'auto' device map is key for your dual 2080 setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
