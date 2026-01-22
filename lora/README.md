# LoRA Fine-Tuning Examples

Low-Rank Adaptation (LoRA) for efficient LLM fine-tuning.

## Files

| File | Purpose | Run Command |
|------|---------|-------------|
| `lora_financial_finetuning.py` | Production training script | `python lora_financial_finetuning.py` |
| `lora_demo_auto.py` | Quick automated demo | `python lora_demo_auto.py` |
| `lora_demo_explained.py` | Interactive tutorial | `python lora_demo_explained.py` |
| `lora_minimal_example.py` | Math visualization | `python lora_minimal_example.py` |

## Quick Start

```bash
# Run the automated demo (uses GPT-2, works on CPU)
python lora_demo_auto.py

# For production (requires GPU):
# Edit model_name in lora_financial_finetuning.py to use Mistral/Llama
python lora_financial_finetuning.py
```

## Why LoRA?

```
Full Fine-Tuning:     Train 7,000,000,000 parameters → Need 112GB+ VRAM
LoRA Fine-Tuning:     Train     4,000,000 parameters → Need 16GB VRAM

Memory Saved: ~99%
Quality: Nearly identical
```

## Key Configuration

```python
LoraConfig(
    r=16,              # Rank: adapter capacity (8-64 typical)
    lora_alpha=32,     # Scaling: effective scale = alpha/r
    target_modules=[   # Which layers to adapt
        "q_proj",      # Query projection
        "v_proj",      # Value projection
        "k_proj",      # Key projection
        "o_proj",      # Output projection
    ],
)
```
