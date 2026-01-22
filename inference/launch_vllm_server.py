#!/usr/bin/env python3
"""
vLLM Server Launch Script
=========================

This script launches a vLLM inference server with optimal settings.

Usage:
    python launch_vllm_server.py [--model MODEL] [--port PORT]

Example:
    python launch_vllm_server.py --model mistralai/Mistral-7B-v0.1 --port 8000
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM inference server")
    parser.add_argument("--model", default="microsoft/phi-2",
                       help="Model to serve (default: microsoft/phi-2)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to serve on (default: 8000)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory fraction to use (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                       help="Maximum sequence length (default: 4096)")
    args = parser.parse_args()

    # =========================================================================
    # PAGEDATTENTION CONFIGURATION
    # =========================================================================
    # block-size: Number of tokens per KV cache block
    #   - Smaller blocks = less memory waste, more overhead
    #   - Larger blocks = more memory waste, less overhead
    #   - Default 16 is a good balance
    #
    # swap-space: CPU RAM (GB) for swapping KV cache when GPU full
    #   - Allows serving more requests than GPU memory permits
    #   - Adds latency when swapping occurs

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),

        # PagedAttention settings
        "--block-size", "16",        # Tokens per KV cache block
        "--swap-space", "4",         # GB of CPU swap space

        # Continuous batching settings
        "--max-num-seqs", "256",     # Max concurrent sequences
        "--max-num-batched-tokens", "8192",  # Max tokens per forward pass

        # Optimization
        "--enforce-eager",           # Disable CUDA graphs (more flexible)
        # "--enable-prefix-caching",   # Enable for shared prefixes (uncomment if supported)
    ]

    print("="*70)
    print("  LAUNCHING VLLM INFERENCE SERVER")
    print("="*70)
    print(f"""
    Model: {args.model}
    Port: {args.port}
    GPU Memory: {args.gpu_memory_utilization * 100:.0f}%

    PagedAttention Settings:
    - Block size: 16 tokens (KV cache granularity)
    - Swap space: 4 GB (CPU overflow)

    Continuous Batching:
    - Max concurrent sequences: 256
    - Max tokens per iteration: 8192

    API Endpoints (OpenAI-compatible):
    - POST http://localhost:{args.port}/v1/completions
    - POST http://localhost:{args.port}/v1/chat/completions
    - GET  http://localhost:{args.port}/health
    """)
    print("="*70)
    print("Starting server (this may take a minute to load the model)...")
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
