# Quantization Examples

Model quantization for efficient deployment (INT8, INT4, AWQ).

## Files

| File | Purpose | Run Command |
|------|---------|-------------|
| `quantization_benchmark.py` | Benchmark FP32/FP16/INT8/INT4 | `python quantization_benchmark.py` |
| `awq_quantization_demo.py` | AWQ demonstration with explanations | `python awq_quantization_demo.py` |

## Quick Start

```bash
# Run AWQ demo (works on CPU, shows concepts)
python awq_quantization_demo.py

# Run benchmark (full INT8/INT4 requires CUDA)
python quantization_benchmark.py --model gpt2 --runs 5
```

## Quantization Trade-offs

```
┌──────────┬─────────────┬───────────────┬────────────────────────┐
│ Format   │ Memory Use  │ Quality Loss  │ Best For               │
├──────────┼─────────────┼───────────────┼────────────────────────┤
│ FP32     │ 100% (base) │ None          │ Training               │
│ FP16     │ ~50%        │ Negligible    │ GPU inference          │
│ INT8     │ ~25%        │ <1%           │ Production serving     │
│ INT4     │ ~12.5%      │ 1-5%          │ Memory-constrained     │
│ AWQ INT4 │ ~12.5%      │ <1%           │ Best quality at 4-bit  │
└──────────┴─────────────┴───────────────┴────────────────────────┘
```

## Why AWQ?

AWQ (Activation-Aware Quantization) protects important weights:

```
Naive Quantization: Compress all weights equally → Some important ones ruined
AWQ Quantization:   Find important weights via calibration → Protect them

Result: INT4 models with FP16-like quality!
```
