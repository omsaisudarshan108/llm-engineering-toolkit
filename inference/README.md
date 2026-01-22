# Inference Optimization

KV caching and vLLM server deployment for production inference.

## Files

| File | Purpose | Run Command |
|------|---------|-------------|
| `kv_cache_demo.py` | KV caching demonstration | `python kv_cache_demo.py` |
| `vllm_server_setup.py` | Setup guide with PagedAttention | `python vllm_server_setup.py` |
| `launch_vllm_server.py` | Production server launcher | `python launch_vllm_server.py` |
| `test_vllm_client.py` | Concurrent request testing | `python test_vllm_client.py` |

## Quick Start

```bash
# Run KV cache demo (works on CPU)
python kv_cache_demo.py

# Set up vLLM (generates server scripts)
python vllm_server_setup.py

# Launch vLLM server (requires CUDA)
python launch_vllm_server.py --model microsoft/phi-2 --port 8000

# Test with concurrent requests
python test_vllm_client.py --concurrent 20
```

## KV Cache Results

```
┌─────────────────────────┬────────────────┬────────────────┐
│ Metric                  │  Without Cache │   With Cache   │
├─────────────────────────┼────────────────┼────────────────┤
│ Time (ms)               │        251.6   │         60.7   │
│ Speedup                 │          1.0x  │          4.1x  │
└─────────────────────────┴────────────────┴────────────────┘

Speedup INCREASES with sequence length (O(n²) → O(n))
```

## PagedAttention (vLLM)

Traditional KV cache wastes memory with pre-allocation:
```
Request 1 (100 tokens, allocated 2048): [████░░░░░░░░░░░░░░░░]  ~5% used
```

PagedAttention allocates in blocks:
```
Request 1 (100 tokens, 7 blocks):       [B1][B2][B3][B4][B5][B6][B7]  100% used
```

Result: **10-20x more concurrent requests!**
