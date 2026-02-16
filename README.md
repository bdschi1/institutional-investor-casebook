# Institutional Investor Casebook

Evaluate LLM reasoning quality on institutional portfolio management cases using quantized local models. Runs Llama 3 8B at 4-bit precision on consumer GPUs — no API keys, no cloud, no data leaving the machine.

---

## What It Does

1. **Loads evaluation cases** from JSONL files — each case has a PM-grade prompt and a golden answer written by a domain expert.
2. **Runs local LLM inference** using 4-bit quantized Llama 3 with a Senior PM system prompt.
3. **Distributes across GPUs** automatically via Accelerate with memory offloading to prevent OOM crashes.
4. **Benchmarks output quality** against golden answers for portfolio construction, risk decomposition, and trade sizing.

---

## Key Features

- **4-bit NF4 quantization** — Runs Llama 3 8B Instruct at ~8.5 GB VRAM per card using bitsandbytes with double quantization and bfloat16 compute.
- **Multi-GPU distribution** — `device_map="auto"` splits the model across available GPUs with explicit per-device memory caps and CPU/disk offloading.
- **Institutional system prompt** — Every inference runs under a Senior PM persona: rigorous, institutional-grade financial reasoning.
- **Low-temperature generation** — `temperature=0.1` for analytical consistency across runs.
- **Offline execution** — Model downloads once from HuggingFace, then runs fully offline. No API calls, no data leakage.

---

## Case Format

Each case is a JSONL entry:

```json
{
  "id": "PM-RISK-001",
  "category": "Portfolio Construction",
  "prompt": "A multi-strat portfolio is Long $500M US Tech (Beta 1.4)...",
  "golden_answer": "Tech Sleeve: -$21M (1.4 Beta * -3% * $500M)..."
}
```

Cases cover portfolio construction, risk decomposition, P&L attribution, hedge effectiveness, and factor analysis.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 11 GB (single GPU) | 2x 11 GB (dual GPU) |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB (model cache) | 20 GB |

Tested on dual NVIDIA RTX 2080 (11 GB each). The runner caps GPU usage at 8.5 GB per card with 2.5 GB buffer, offloading overflow to RAM/disk.

---

## Setup

```bash
git clone https://github.com/bdschi1/institutional-investor-casebook.git
cd institutional-investor-casebook
pip install -e "."
```

### HuggingFace Login

The model requires a HuggingFace account with Llama 3 access:

```bash
pip install huggingface_hub
huggingface-cli login
```

Or set `HF_TOKEN` in `.env`.

## Run

```bash
python run_benchmark.py
```

Loads cases from `src/investor_casebook/data/`, initializes the quantized model, and runs inference.

To run inference directly:

```python
from investor_casebook.runner import CasebookRunner
from investor_casebook.data.loader import CasebookLoader

loader = CasebookLoader("src/investor_casebook/data/")
cases = loader.load_cases("sample_cases.json")

runner = CasebookRunner()
analysis = runner.run_case(cases[0])
print(analysis)
```

---

## Project Structure

```
institutional-investor-casebook/
├── run_benchmark.py                    Main entry point
├── gpu_dist_test.py                    GPU distribution testing
├── hf_login_script.py                  HuggingFace authentication
├── src/
│   └── investor_casebook/
│       ├── runner.py                   CasebookRunner — quantized model loading + inference
│       ├── reasoning/
│       │   └── evaluator.py            InvestorEvaluator — multi-GPU distribution
│       └── data/
│           ├── loader.py               CasebookLoader — JSONL case loading
│           └── sample_cases.json       Sample PM analysis cases
├── tests/
│   └── test_loader.py                  Case loading + error handling tests
└── pyproject.toml
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
black .
```

---

## Stack

- **LLM**: [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (4-bit quantized)
- **Inference**: [Transformers](https://huggingface.co/docs/transformers) + [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (NF4)
- **Distribution**: [Accelerate](https://huggingface.co/docs/accelerate) (auto device map, memory offloading)
- **Data**: JSONL cases with golden answers

## License

MIT

---

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Llama 3](https://img.shields.io/badge/Llama_3-0467DF?style=flat&logo=meta&logoColor=white)
