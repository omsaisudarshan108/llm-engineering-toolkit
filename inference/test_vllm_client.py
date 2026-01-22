#!/usr/bin/env python3
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

    print(f"\nLaunching {num_requests} concurrent requests...")
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
    print("\n1. Single Request Test:")
    print("-"*40)
    result = send_request(
        "Explain artificial intelligence in one paragraph:",
        args.host, args.port, max_tokens=100
    )
    print(f"   Latency: {result['latency']:.2f}s")
    print(f"   Tokens: {result.get('usage', {}).get('completion_tokens', 'N/A')}")
    print(f"   Response: {result['choices'][0]['text'][:100]}...")

    # Test concurrent requests
    print(f"\n2. Concurrent Requests Test ({args.concurrent} requests):")
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

    print("\nNote: High throughput with concurrent requests demonstrates")
    print("continuous batching effectiveness!")

if __name__ == "__main__":
    main()
