"""
vLLM Local Inference Server Setup
==================================

This script demonstrates deploying a local LLM inference server using vLLM
with continuous batching and PagedAttention.

VLLM KEY INNOVATIONS:
---------------------
1. PagedAttention: Efficient KV cache memory management
2. Continuous Batching: Dynamic request scheduling
3. Optimized CUDA kernels: Faster attention computation

WHY VLLM?
---------
Traditional serving (e.g., HuggingFace):
- Fixed batch size
- Pre-allocated KV cache (wastes memory)
- Sequential request processing

vLLM:
- Dynamic batching (mix short/long requests)
- Paged KV cache (use only what you need)
- 10-24x higher throughput!

INSTALLATION:
-------------
pip install vllm

REQUIREMENTS:
- NVIDIA GPU with CUDA support
- Sufficient VRAM for model + KV cache
- Linux (vLLM has limited Windows/Mac support)
"""

import os
import sys
import time
import json
import asyncio
import subprocess
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading


# =============================================================================
# PAGEDATTENTION EXPLANATION
# =============================================================================

PAGED_ATTENTION_EXPLANATION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PAGEDATTENTION EXPLAINED                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THE PROBLEM WITH TRADITIONAL KV CACHE:                                      ║
║  ─────────────────────────────────────                                       ║
║  Traditional systems pre-allocate memory for maximum sequence length:        ║
║                                                                              ║
║    Request 1 (actual: 100 tokens, allocated: 2048)  [████░░░░░░░░░░░░░░░░]  ║
║    Request 2 (actual: 500 tokens, allocated: 2048)  [██████████░░░░░░░░░░]  ║
║    Request 3 (actual: 50 tokens, allocated: 2048)   [██░░░░░░░░░░░░░░░░░░]  ║
║                                                                              ║
║    → Massive memory waste! Only ~10% utilized.                              ║
║    → Can only serve ~3 concurrent requests.                                 ║
║                                                                              ║
║  PAGEDATTENTION SOLUTION:                                                    ║
║  ────────────────────────                                                    ║
║  Inspired by OS virtual memory: allocate in fixed-size "pages" (blocks)     ║
║                                                                              ║
║    Block size: 16 tokens                                                     ║
║    Request 1 (100 tokens): [B1][B2][B3][B4][B5][B6][B7]  (7 blocks)        ║
║    Request 2 (500 tokens): [B8][B9]...[B39]              (32 blocks)        ║
║    Request 3 (50 tokens):  [B40][B41][B42][B43]          (4 blocks)         ║
║                                                                              ║
║    → No wasted space! Memory = actual tokens used.                          ║
║    → Can serve 10-20x more concurrent requests!                             ║
║                                                                              ║
║  HOW IT WORKS:                                                               ║
║  ─────────────                                                               ║
║  1. KV cache divided into fixed-size blocks (e.g., 16 tokens each)          ║
║  2. Block table maps logical positions → physical GPU memory                 ║
║  3. Blocks allocated on-demand as sequence grows                             ║
║  4. Blocks freed when request completes (reused by new requests)            ║
║  5. Non-contiguous storage → custom attention kernel handles it             ║
║                                                                              ║
║  MEMORY SHARING:                                                             ║
║  ───────────────                                                             ║
║  Identical prefixes can SHARE blocks (copy-on-write):                       ║
║                                                                              ║
║    System prompt: "You are a helpful assistant..."                          ║
║    Request 1: [Shared prefix blocks] → [Request 1 specific]                 ║
║    Request 2: [Shared prefix blocks] → [Request 2 specific]                 ║
║                                                                              ║
║    → Multiple requests share the same cached system prompt!                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# CONTINUOUS BATCHING EXPLANATION
# =============================================================================

CONTINUOUS_BATCHING_EXPLANATION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CONTINUOUS BATCHING EXPLAINED                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STATIC BATCHING (Traditional):                                              ║
║  ──────────────────────────────                                              ║
║  Wait for batch to fill, then process together:                             ║
║                                                                              ║
║    Time→  ████████████████████████████████████████                          ║
║    Req 1: [=========PROCESSING=========][DONE]                              ║
║    Req 2: [=========PROCESSING=========][DONE]     ← Short request waits!  ║
║    Req 3: [=========PROCESSING=========][DONE]                              ║
║    Req 4:                                    [=====NEXT BATCH=====]         ║
║                                                                              ║
║    Problems:                                                                 ║
║    - Short requests wait for long ones to finish                            ║
║    - New requests must wait for entire batch                                ║
║    - GPU utilization drops as requests finish                               ║
║                                                                              ║
║  CONTINUOUS BATCHING (vLLM):                                                 ║
║  ──────────────────────────────                                              ║
║  Add/remove requests dynamically at each iteration:                         ║
║                                                                              ║
║    Time→  ████████████████████████████████████████                          ║
║    Req 1: [=====][DONE]                                                     ║
║    Req 2: [===============][DONE]                                           ║
║    Req 3: [=========================][DONE]                                 ║
║    Req 4:       [===========][DONE]     ← Joins immediately!               ║
║    Req 5:              [======][DONE]   ← Joins when space available       ║
║                                                                              ║
║    Benefits:                                                                 ║
║    - Short requests complete quickly (no waiting)                           ║
║    - New requests join immediately                                          ║
║    - GPU always fully utilized                                              ║
║    - Much higher throughput!                                                ║
║                                                                              ║
║  ITERATION-LEVEL SCHEDULING:                                                 ║
║  ───────────────────────────                                                 ║
║  Each forward pass (iteration):                                              ║
║  1. Check for completed requests → remove from batch                        ║
║  2. Check for new requests → add to batch                                   ║
║  3. Run forward pass for all active requests                                ║
║  4. Decode one token per request                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# VLLM SERVER CONFIGURATION
# =============================================================================

@dataclass
class VLLMServerConfig:
    """Configuration for vLLM server deployment."""

    # Model settings
    model: str = "microsoft/phi-2"  # Small model for demo; use "mistralai/Mistral-7B-v0.1" for production
    dtype: str = "auto"              # auto, float16, bfloat16, float32
    max_model_len: int = 2048        # Maximum sequence length

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None    # Optional API key for authentication

    # Performance settings
    tensor_parallel_size: int = 1    # Number of GPUs for tensor parallelism
    gpu_memory_utilization: float = 0.9  # Fraction of GPU memory to use
    max_num_seqs: int = 256          # Maximum concurrent sequences

    # PagedAttention settings
    block_size: int = 16             # KV cache block size (tokens per block)
    swap_space: int = 4              # CPU swap space in GB

    # Continuous batching settings
    max_num_batched_tokens: int = 8192  # Max tokens per iteration

    def to_args(self) -> List[str]:
        """Convert config to command line arguments."""
        args = [
            "--model", self.model,
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-seqs", str(self.max_num_seqs),
            "--block-size", str(self.block_size),
            "--swap-space", str(self.swap_space),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
        ]

        if self.api_key:
            args.extend(["--api-key", self.api_key])

        return args


# =============================================================================
# SERVER LAUNCHER
# =============================================================================

def launch_vllm_server(config: VLLMServerConfig) -> subprocess.Popen:
    """
    Launch vLLM server as a subprocess.

    The server provides an OpenAI-compatible API at:
    - POST /v1/completions (text completion)
    - POST /v1/chat/completions (chat completion)
    """
    cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"] + config.to_args()

    print("Launching vLLM server with command:")
    print(f"  {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return process


def wait_for_server(host: str, port: int, timeout: int = 120) -> bool:
    """Wait for server to become ready."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/health"
    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError):
            pass
        time.sleep(1)

    return False


# =============================================================================
# CLIENT FOR TESTING
# =============================================================================

def send_completion_request(
    prompt: str,
    host: str = "localhost",
    port: int = 8000,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> Dict:
    """Send a completion request to the vLLM server."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}

    data = json.dumps({
        "model": "default",  # vLLM uses the loaded model
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    start_time = time.time()
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode())
    elapsed = time.time() - start_time

    result["elapsed_time"] = elapsed
    return result


def send_concurrent_requests(
    prompts: List[str],
    host: str = "localhost",
    port: int = 8000,
    max_tokens: int = 50,
) -> List[Dict]:
    """Send multiple concurrent requests to demonstrate continuous batching."""

    results = []
    lock = threading.Lock()

    def make_request(prompt: str, request_id: int):
        start = time.time()
        result = send_completion_request(prompt, host, port, max_tokens)
        result["request_id"] = request_id
        result["start_time"] = start
        result["end_time"] = time.time()
        with lock:
            results.append(result)

    # Launch all requests concurrently
    threads = []
    for i, prompt in enumerate(prompts):
        t = threading.Thread(target=make_request, args=(prompt, i))
        threads.append(t)

    print(f"Launching {len(prompts)} concurrent requests...")
    launch_time = time.time()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    total_time = time.time() - launch_time
    print(f"All requests completed in {total_time:.2f}s")

    return sorted(results, key=lambda x: x["request_id"])


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_throughput(
    host: str = "localhost",
    port: int = 8000,
    num_requests: int = 10,
    prompt_tokens: int = 50,
    max_tokens: int = 100,
) -> Dict:
    """
    Benchmark server throughput with concurrent requests.

    Demonstrates continuous batching benefits:
    - All requests processed concurrently
    - Short requests complete quickly
    - GPU fully utilized
    """
    # Create prompts of similar length
    base_prompt = "Explain the concept of " + "artificial intelligence " * (prompt_tokens // 5)
    prompts = [base_prompt for _ in range(num_requests)]

    start_time = time.time()
    results = send_concurrent_requests(prompts, host, port, max_tokens)
    total_time = time.time() - start_time

    # Calculate metrics
    total_prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    total_tokens = total_prompt_tokens + total_completion_tokens

    throughput = total_completion_tokens / total_time

    return {
        "num_requests": num_requests,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "completion_tokens": total_completion_tokens,
        "throughput_tokens_per_sec": throughput,
        "avg_latency": sum(r["elapsed_time"] for r in results) / len(results),
    }


# =============================================================================
# MAIN DEMONSTRATION SCRIPT
# =============================================================================

def create_server_script():
    """Create a standalone server launch script."""

    script = '''#!/usr/bin/env python3
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
        print("\\nServer stopped.")

if __name__ == "__main__":
    main()
'''
    return script


def create_client_script():
    """Create a client script for testing the server."""

    script = '''#!/usr/bin/env python3
"""
vLLM Client Test Script
=======================

Tests the vLLM server with single and concurrent requests.

Usage:
    python test_vllm_client.py [--port PORT] [--concurrent N]
"""

import argparse
import json
import time
import threading
import urllib.request
from typing import List, Dict

def send_request(
    prompt: str,
    host: str = "localhost",
    port: int = 8000,
    max_tokens: int = 100,
) -> Dict:
    """Send a completion request."""
    url = f"http://{host}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}

    data = json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    start = time.time()
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode())
    result["latency"] = time.time() - start

    return result


def concurrent_benchmark(
    num_requests: int,
    host: str,
    port: int,
    max_tokens: int = 50,
) -> Dict:
    """Benchmark with concurrent requests."""

    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a haiku about programming:",
        "What is the meaning of life?",
        "Describe the water cycle:",
        "How does a neural network work?",
    ] * (num_requests // 5 + 1)
    prompts = prompts[:num_requests]

    results = []
    lock = threading.Lock()

    def worker(prompt: str, req_id: int):
        result = send_request(prompt, host, port, max_tokens)
        result["request_id"] = req_id
        with lock:
            results.append(result)

    threads = [
        threading.Thread(target=worker, args=(p, i))
        for i, p in enumerate(prompts)
    ]

    print(f"\\nLaunching {num_requests} concurrent requests...")
    start = time.time()

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_time = time.time() - start

    # Calculate metrics
    total_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    return {
        "num_requests": num_requests,
        "total_time_sec": total_time,
        "total_tokens": total_tokens,
        "throughput_tok_per_sec": total_tokens / total_time,
        "avg_latency_sec": avg_latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Test vLLM server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrent", type=int, default=10,
                       help="Number of concurrent requests for benchmark")
    args = parser.parse_args()

    print("="*60)
    print("  VLLM SERVER TEST")
    print("="*60)

    # Test single request
    print("\\n1. Single Request Test:")
    print("-"*40)
    result = send_request(
        "Explain artificial intelligence in one paragraph:",
        args.host, args.port, max_tokens=100
    )
    print(f"   Latency: {result['latency']:.2f}s")
    print(f"   Tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
    print(f"   Response: {result['choices'][0]['text'][:100]}...")

    # Test concurrent requests
    print(f"\\n2. Concurrent Requests Test ({args.concurrent} requests):")
    print("-"*40)
    metrics = concurrent_benchmark(args.concurrent, args.host, args.port)

    print(f"""
   Results:
   ┌────────────────────────┬─────────────────┐
   │ Metric                 │ Value           │
   ├────────────────────────┼─────────────────┤
   │ Total time             │ {metrics['total_time_sec']:>10.2f} sec  │
   │ Total tokens generated │ {metrics['total_tokens']:>10}      │
   │ Throughput             │ {metrics['throughput_tok_per_sec']:>10.1f} tok/s│
   │ Avg latency per request│ {metrics['avg_latency_sec']:>10.2f} sec  │
   └────────────────────────┴─────────────────┘
    """)

    print("\\nNote: High throughput with concurrent requests demonstrates")
    print("continuous batching effectiveness!")

if __name__ == "__main__":
    main()
'''
    return script


def main():
    """Main function - prints documentation and creates scripts."""

    print("="*80)
    print("  VLLM INFERENCE SERVER SETUP GUIDE")
    print("="*80)

    # Print explanations
    print(PAGED_ATTENTION_EXPLANATION)
    print(CONTINUOUS_BATCHING_EXPLANATION)

    # Create launch script
    print("\n" + "="*80)
    print("  CREATING SERVER LAUNCH SCRIPT")
    print("="*80)

    launch_script = create_server_script()
    launch_path = "/Users/Shannanigans/launch_vllm_server.py"
    with open(launch_path, "w") as f:
        f.write(launch_script)
    print(f"\n✓ Created: {launch_path}")

    # Create client script
    client_script = create_client_script()
    client_path = "/Users/Shannanigans/test_vllm_client.py"
    with open(client_path, "w") as f:
        f.write(client_script)
    print(f"✓ Created: {client_path}")

    # Print usage instructions
    print("\n" + "="*80)
    print("  USAGE INSTRUCTIONS")
    print("="*80)

    print("""
    STEP 1: Install vLLM
    ────────────────────
    pip install vllm

    Note: Requires NVIDIA GPU with CUDA support


    STEP 2: Launch the Server
    ─────────────────────────
    # Using the created script:
    python launch_vllm_server.py --model microsoft/phi-2 --port 8000

    # Or directly with vLLM CLI:
    python -m vllm.entrypoints.openai.api_server \\
        --model mistralai/Mistral-7B-v0.1 \\
        --port 8000 \\
        --gpu-memory-utilization 0.9


    STEP 3: Test the Server
    ───────────────────────
    # Using the test client:
    python test_vllm_client.py --port 8000 --concurrent 20

    # Or with curl:
    curl http://localhost:8000/v1/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "default",
            "prompt": "Hello, how are you?",
            "max_tokens": 50
        }'


    STEP 4: Monitor GPU Utilization
    ────────────────────────────────
    # In another terminal:
    watch -n 0.5 nvidia-smi

    # You should see:
    # - High GPU utilization during inference
    # - Memory usage stable (PagedAttention efficiency)
    # - Utilization stays high with concurrent requests (continuous batching)


    KEY CONFIGURATION OPTIONS:
    ──────────────────────────
    --gpu-memory-utilization 0.9   # Use 90% of GPU memory for KV cache
    --max-num-seqs 256             # Max concurrent requests
    --block-size 16                # PagedAttention block size
    --tensor-parallel-size N       # Split model across N GPUs


    EXPECTED PERFORMANCE:
    ─────────────────────
    Model           │ GPU      │ Throughput (tokens/sec)
    ────────────────┼──────────┼────────────────────────
    Phi-2 (2.7B)    │ RTX 3090 │ ~500-1000
    Mistral-7B      │ RTX 3090 │ ~200-400
    Llama-13B       │ A100     │ ~300-600
    Llama-70B       │ 2xA100   │ ~100-200
    """)

    print("="*80)
    print("  GPU UTILIZATION IMPROVEMENTS")
    print("="*80)
    print("""
    WITHOUT VLLM (Traditional):
    ───────────────────────────
    - Sequential request processing
    - Pre-allocated KV cache (memory waste)
    - GPU idle between batches

    GPU Utilization: ▓▓▓░░░░░░░▓▓▓░░░░░░░▓▓▓░░░░░░░  (~30%)


    WITH VLLM:
    ──────────
    - Continuous batching (always have work)
    - PagedAttention (efficient memory)
    - Optimal batch sizes

    GPU Utilization: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (~90%+)


    RESULT: 10-24x higher throughput!
    """)

    print("="*80)
    print("  SETUP COMPLETE")
    print("="*80)
    print("""
    Files created:
    - launch_vllm_server.py  : Server launch script with optimal settings
    - test_vllm_client.py    : Client for testing single/concurrent requests

    To get started:
    1. pip install vllm
    2. python launch_vllm_server.py
    3. python test_vllm_client.py --concurrent 20
    """)


if __name__ == "__main__":
    main()
