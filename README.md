# LLM Engineering Toolkit

A comprehensive collection of production-ready code examples for Large Language Model engineering, covering fine-tuning, quantization, attention mechanisms, and inference optimization.

## Overview

```
llm-engineering-toolkit/
├── lora/                    # LoRA fine-tuning examples
├── quantization/            # Model quantization (INT8, INT4, AWQ)
├── attention/               # Attention mechanisms from scratch
├── inference/               # KV caching & vLLM server deployment
└── docs/                    # Additional documentation
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/omsaisudarshan108/llm-engineering-toolkit.git
cd llm-engineering-toolkit

# Install dependencies
pip install -r requirements.txt

# Run any demo
python attention/attention_from_scratch.py
python inference/kv_cache_demo.py
python lora/lora_demo_auto.py
```

## Components

### 1. LoRA Fine-Tuning (`lora/`)

Fine-tune large language models efficiently using Low-Rank Adaptation.

| File | Description |
|------|-------------|
| `lora_financial_finetuning.py` | Production-ready LoRA training for financial domain |
| `lora_demo_auto.py` | Automated demo with training results |
| `lora_demo_explained.py` | Interactive step-by-step tutorial |
| `lora_minimal_example.py` | Core concepts and math visualization |

**Key Features:**
- Configurable LoRA rank and target modules
- 4-bit quantization support (QLoRA)
- Before/after loss comparison
- Financial sentiment analysis task

**Example Results:**
```
╔════════════════════════════════════════════════╗
║  🧊 FROZEN:     124,439,808 params (untouched) ║
║  🔥 TRAINABLE:    1,622,016 params (LoRA)      ║
║  📊 Training only 1.29% of the model!          ║
╚════════════════════════════════════════════════╝
```

### 2. Quantization (`quantization/`)

Benchmark and implement model quantization for efficient deployment.

| File | Description |
|------|-------------|
| `quantization_benchmark.py` | FP32/FP16/INT8/INT4 comparison benchmark |
| `awq_quantization_demo.py` | Activation-Aware Quantization demonstration |

**Key Features:**
- Latency, throughput, and memory measurements
- AWQ calibration and implementation
- Perplexity comparison before/after
- Quality degradation analysis

**AWQ Results:**
```
┌────────────────────┬─────────────┬────────────────┐
│ Method             │ Perplexity  │ vs Original    │
├────────────────────┼─────────────┼────────────────┤
│ Original (FP32)    │   1211.75   │     baseline   │
│ AWQ INT4           │   1212.17   │       +0.0%    │
│ Naive INT4         │   1213.37   │       +0.1%    │
└────────────────────┴─────────────┴────────────────┘
```

### 3. Attention Mechanisms (`attention/`)

Understand transformer attention from first principles.

| File | Description |
|------|-------------|
| `attention_from_scratch.py` | Complete attention implementation with math-to-code mapping |

**Key Features:**
- Scaled dot-product attention with temperature scaling
- Multi-head attention with configurable heads
- Causal masking for autoregressive models
- Visualization of attention patterns

**Math-to-Code Mapping:**
```
EQUATION                         CODE
────────────────────────────────────────────────────────
scores = Q × K^T                 torch.matmul(Q, K.transpose(-2,-1))
scaled = scores / √d_k           scores / math.sqrt(d_k)
weights = softmax(scaled)        F.softmax(scaled_scores, dim=-1)
output = weights × V             torch.matmul(attention_weights, V)
```

### 4. Inference Optimization (`inference/`)

Optimize LLM inference for production deployment.

| File | Description |
|------|-------------|
| `kv_cache_demo.py` | KV caching with timing comparison |
| `vllm_server_setup.py` | vLLM setup guide with PagedAttention |
| `launch_vllm_server.py` | Production vLLM server launcher |
| `test_vllm_client.py` | Client for concurrent request testing |

**KV Cache Results:**
```
┌─────────────────────────┬────────────────┬────────────────┐
│ Metric                  │  Without Cache │   With Cache   │
├─────────────────────────┼────────────────┼────────────────┤
│ Time (ms)               │        251.6   │         60.7   │
│ Speedup                 │          1.0x  │          4.1x  │
└─────────────────────────┴────────────────┴────────────────┘
```

## Requirements

### Core Dependencies
```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
```

### Optional (GPU/Quantization)
```
bitsandbytes>=0.41.0    # INT8/INT4 quantization (CUDA required)
vllm>=0.2.0              # High-performance inference server
autoawq>=0.1.0           # AWQ quantization
```

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Attention demos | CPU | Any |
| LoRA fine-tuning (GPT-2) | CPU, 8GB RAM | GPU, 8GB VRAM |
| LoRA fine-tuning (7B) | GPU, 16GB VRAM | GPU, 24GB VRAM |
| Quantization benchmarks | GPU, 8GB VRAM | GPU, 16GB VRAM |
| vLLM serving | GPU, 16GB VRAM | GPU, 24GB+ VRAM |

## Key Concepts Explained

### Why LoRA?
- Train <2% of parameters while achieving similar results to full fine-tuning
- Memory efficient: 7B model trainable on 16GB GPU
- No catastrophic forgetting of base knowledge
- Modular adapters can be swapped without reloading base model

### Why Quantization?
- Reduce model size by 2-4x (FP16 → INT4)
- Enable larger models on consumer hardware
- AWQ preserves quality by protecting important weights

### Why KV Caching?
- Avoid recomputing attention for previous tokens
- O(n²) → O(n) complexity improvement
- Essential for practical autoregressive generation

### Why vLLM?
- PagedAttention: 10-20x more concurrent requests
- Continuous batching: Always-high GPU utilization
- OpenAI-compatible API for easy integration

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR.

## Author

Created with Claude Code as a comprehensive LLM engineering reference.
