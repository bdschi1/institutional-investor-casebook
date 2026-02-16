"""CasebookRunner — quantized LLM inference for PM case evaluation.

Loads Llama 3 8B Instruct at 4-bit NF4 precision, distributes across
available GPUs via Accelerate, and runs inference under a Senior PM
system prompt.

Pass ``mock=True`` to skip model loading for CI/testing without GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

from investor_casebook.data.loader import CasebookLoader


class CasebookRunner:
    """Run quantized LLM inference on institutional PM cases."""

    SYSTEM_PROMPT = (
        "You are a Senior Portfolio Manager at a global multi-strategy "
        "hedge fund. Provide rigorous, institutional-grade financial "
        "reasoning. Include specific numbers, calculations, and "
        "quantitative justifications in your analysis."
    )

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        mock: bool = False,
    ):
        self.model_id = model_id
        self.mock = mock
        self.model = None
        self.tokenizer = None

        if mock:
            print("--- CasebookRunner initialized in MOCK mode (no GPU) ---")
            return

        # Lazy imports — only needed when actually loading a model
        import torch
        from dotenv import load_dotenv
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        load_dotenv()

        self.offload_dir = "offload_temp"
        os.makedirs(self.offload_dir, exist_ok=True)

        print(f"--- Initializing CasebookRunner ({model_id}) ---")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Cap GPU usage per card — leave 2.5 GB buffer for system overhead
        max_memory = {0: "8.5GiB", 1: "9GiB", "cpu": "20GiB"}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=self.offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )

        print(f"Device Map: {self.model.hf_device_map}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_case(self, case_dict: dict) -> str:
        """Run inference on a single case.

        Parameters
        ----------
        case_dict : dict
            Must contain ``id`` and ``prompt`` keys.

        Returns
        -------
        str — model-generated analysis text.
        """
        case_id = case_dict.get("id", "Unknown")
        prompt_text = case_dict.get("prompt", "")

        if self.mock:
            return (
                f"[MOCK] Analysis for {case_id}: This is a placeholder "
                f"response. In production, the quantized Llama 3 model "
                f"would generate a detailed PM-grade analysis for: "
                f"{prompt_text[:200]}..."
            )

        import torch

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"ANALYST CASE {case_id}: {prompt_text}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move to the device of the first model parameter (respects device_map)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"\nEvaluating Case {case_id}...")
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_tokens[0][len(inputs["input_ids"][0]) :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def run_all_cases(self, cases: list[dict]) -> list[dict]:
        """Run inference on all cases and return enriched results.

        Returns
        -------
        list[dict] — each dict has the original case fields plus
        ``model_output`` with the generated text.
        """
        results = []
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] Running case {case.get('id', '?')}...")
            output = self.run_case(case)
            results.append(
                {
                    "id": case.get("id", "Unknown"),
                    "category": case.get("category", ""),
                    "prompt": case.get("prompt", ""),
                    "golden_answer": case.get("golden_answer", ""),
                    "model_output": output,
                }
            )
        return results


if __name__ == "__main__":
    loader = CasebookLoader(Path("src/investor_casebook/data/"))
    cases = loader.load_cases("sample_cases.jsonl")

    if not cases:
        print("!! No cases found.")
    else:
        runner = CasebookRunner()
        analysis = runner.run_case(cases[0])

        print("\n" + "=" * 60)
        print(f"CASE {cases[0].get('id')} - PM ANALYSIS:")
        print("=" * 60)
        print(analysis)
        print("=" * 60)
        print(f"GOLDEN ANSWER: {cases[0].get('golden_answer')}")
