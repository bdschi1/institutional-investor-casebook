import os
from investor_casebook.data.loader import CasebookLoader
from investor_casebook.reasoning.evaluator import InvestorEvaluator

def main():
    # 1. Load data
    loader = CasebookLoader("src/investor_casebook/data")
    cases = loader.load_cases("sample_cases.jsonl")
    print(f"Loaded {len(cases)} cases.")

    # 2. Initialize Evaluator (This will distribute across GPUs)
    # Defaulting to a small model for testing
    evaluator = InvestorEvaluator("gpustack/llama-3-8b-instruct")
    
    # 3. Future: Add scoring logic here
    print("Benchmark system ready for production-grade testing.")

if __name__ == "__main__":
    main()
