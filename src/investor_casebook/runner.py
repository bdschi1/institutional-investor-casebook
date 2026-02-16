import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from investor_casebook.data.loader import CasebookLoader

# Load environment variables (HF_TOKEN, HF_HOME, etc.)
load_dotenv()

class CasebookRunner:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_id = model_id
        # Safety valve for the 11GB VRAM limit
        self.offload_dir = "offload_temp"
        os.makedirs(self.offload_dir, exist_ok=True)
        
        print(f"--- Initializing Institutional Runner (Llama-3-8B) ---")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # We cap GPU usage at ~8.5GB to allow room for 'overhead' and system tasks
        # Total 11GB - 8.5GB = 2.5GB buffer per card
        max_memory = {0: "8.5GiB", 1: "9GiB", "cpu": "20GiB"}
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=self.offload_dir,
            offload_state_dict=True, # Moves data to RAM if VRAM spikes
            low_cpu_mem_usage=True
        )
        
        print(f"Device Map: {self.model.hf_device_map}")
        
        # FINAL ATTEMPT CONFIG: Forced offloading to prevent driver crash
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            quantization_config=bnb_config,
            device_map="auto",             # Let Accelerate handle the math
            max_memory=max_memory,
            offload_folder=self.offload_dir,
            offload_state_dict=True,      # Crucial for 11GB VRAM stability
            low_cpu_mem_usage=True
        )
        
        print(f"Final Device Map: {self.model.hf_device_map}")

    def run_case(self, case_dict):
        """
        Takes a single case from the Golden Answer dataset and runs inference.
        Matches schema: {"id": "...", "prompt": "...", "golden_answer": "..."}
        """
        case_id = case_dict.get("id", "Unknown")
        prompt_text = case_dict.get("prompt", "")
        
        # Llama-3 Instruct Prompt Template
        messages = [
            {
                "role": "system", 
                "content": "You are a Senior Portfolio Manager at a global multi-strategy hedge fund. Provide rigorous, institutional-grade financial reasoning."
            },
            {
                "role": "user", 
                "content": f"ANALYST CASE {case_id}: {prompt_text}"
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        print(f"\nEvaluating Case {case_id}...")
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for analytical consistency
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    # 1. Initialize Loader (Point to your data directory)
    loader = CasebookLoader("src/investor_casebook/data/")
    
    # 2. Load the cases (Specify your filename)
    cases = loader.load_cases("sample_cases.json") # Updated to match your filename
    
    if not cases:
        print("!! No cases found. Check path: src/investor_casebook/data/sample_cases.json")
    else:
        # 3. Initialize Runner and Evaluate First Case
        runner = CasebookRunner()
        analysis = runner.run_case(cases[0])
        
        print("\n" + "="*60)
        print(f"CASE {cases[0].get('id')} - PM ANALYSIS:")
        print("="*60)
        print(analysis)
        print("="*60)
        print(f"GOLDEN ANSWER: {cases[0].get('golden_answer')}")