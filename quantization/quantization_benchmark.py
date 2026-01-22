"""
Quantization Inference Benchmark: FP16 vs INT8 vs INT4
=======================================================

This script benchmarks inference performance across different quantization levels,
measuring latency, throughput, memory usage, and output quality.

QUANTIZATION OVERVIEW:
----------------------
- FP16 (16-bit float): Full precision, best quality, highest memory
- INT8 (8-bit integer): 2x memory reduction, minimal quality loss
- INT4 (4-bit integer): 4x memory reduction, some quality degradation

Usage: python quantization_benchmark.py [--model MODEL_NAME] [--device cuda/cpu]
"""

import torch
import gc
import time
import argparse
import statistics
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

warnings.filterwarnings("ignore")

# Check for required packages
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install transformers accelerate psutil")
    exit(1)

# Optional: bitsandbytes for quantization (CUDA only)
BNB_AVAILABLE = False
BitsAndBytesConfig = None
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes
    BNB_AVAILABLE = True
except ImportError:
    pass  # Will run FP16/FP32 only


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration parameters."""
    # Model settings
    model_name: str = "gpt2"  # Use small model for demo; change to larger for real benchmarks
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Benchmark settings
    num_warmup_runs: int = 3       # Warmup iterations (not counted)
    num_benchmark_runs: int = 10   # Actual benchmark iterations
    max_new_tokens: int = 50       # Tokens to generate per prompt
    batch_size: int = 1            # Batch size for throughput test

    # Output settings
    verbose: bool = True
    save_outputs: bool = True      # Save generated text for quality comparison


# Fixed prompt set for consistent benchmarking
BENCHMARK_PROMPTS = [
    # Short prompts (test startup latency)
    "The capital of France is",
    "In machine learning, a neural network",
    "The stock market today showed",

    # Medium prompts (typical use case)
    "Explain the concept of inflation in simple terms:",
    "Write a Python function that calculates the factorial of a number:",
    "The key differences between supervised and unsupervised learning are:",

    # Longer context prompts (test context handling)
    """Summarize the following: Artificial intelligence has transformed numerous industries
    over the past decade. From healthcare diagnostics to autonomous vehicles, AI systems
    are becoming increasingly capable. However, concerns about bias, transparency, and
    job displacement remain significant challenges that researchers and policymakers must address.""",

    # Technical/domain-specific (test knowledge retention after quantization)
    "The transformer architecture uses self-attention mechanisms to",
    "In quantitative finance, the Black-Scholes model is used to",
    "The principles of object-oriented programming include",
]


# =============================================================================
# MEMORY TRACKING UTILITIES
# =============================================================================

def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    """Get reserved GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 2)
    return 0.0


def get_cpu_memory_mb() -> float:
    """Get current process CPU memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def clear_memory():
    """Clear GPU and CPU memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def track_memory(device: str):
    """Context manager to track memory usage during operation."""
    clear_memory()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_mem = get_gpu_memory_mb()
    else:
        start_mem = get_cpu_memory_mb()

    yield

    if device == "cuda":
        torch.cuda.synchronize()
        end_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        end_mem = get_cpu_memory_mb()

    return end_mem - start_mem


# =============================================================================
# MODEL LOADING WITH DIFFERENT QUANTIZATION
# =============================================================================

@dataclass
class QuantizationConfig:
    """Configuration for a specific quantization level."""
    name: str
    bits: int
    description: str
    bnb_config: Optional[BitsAndBytesConfig] = None
    torch_dtype: torch.dtype = torch.float16


def get_quantization_configs(include_quantized: bool = True) -> Dict[str, QuantizationConfig]:
    """Define quantization configurations for benchmarking."""

    configs = {}

    # FP32 - Full precision (CPU baseline)
    configs["FP32"] = QuantizationConfig(
        name="FP32",
        bits=32,
        description="32-bit floating point (full precision)",
        bnb_config=None,
        torch_dtype=torch.float32,
    )

    # FP16 - Half precision
    configs["FP16"] = QuantizationConfig(
        name="FP16",
        bits=16,
        description="16-bit floating point (half precision)",
        bnb_config=None,
        torch_dtype=torch.float16,
    )

    # Only add quantized configs if bitsandbytes is available and requested
    if include_quantized and BNB_AVAILABLE and BitsAndBytesConfig is not None:
        # INT8 - 8-bit quantization
        configs["INT8"] = QuantizationConfig(
            name="INT8",
            bits=8,
            description="8-bit integer quantization (LLM.int8())",
            bnb_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,  # Outlier threshold
            ),
            torch_dtype=torch.float16,
        )

        # INT4 - 4-bit quantization (NF4)
        configs["INT4"] = QuantizationConfig(
            name="INT4",
            bits=4,
            description="4-bit NormalFloat quantization (QLoRA-style)",
            bnb_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",           # NormalFloat 4-bit
                bnb_4bit_compute_dtype=torch.float16, # Compute in FP16
                bnb_4bit_use_double_quant=True,       # Nested quantization
            ),
            torch_dtype=torch.float16,
        )

    return configs


def load_model_with_quantization(
    model_name: str,
    quant_config: QuantizationConfig,
    device: str,
) -> Tuple[AutoModelForCausalLM, float]:
    """
    Load model with specified quantization and measure memory.

    Returns: (model, memory_used_mb)
    """
    clear_memory()

    print(f"    Loading {quant_config.name}...", end=" ", flush=True)
    start_time = time.time()

    # Track memory before loading
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = get_gpu_memory_mb()
    else:
        mem_before = get_cpu_memory_mb()

    # Load model with appropriate settings
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if quant_config.bnb_config is not None:
        # Quantized loading (requires CUDA)
        if device != "cuda":
            print("SKIPPED (requires CUDA)")
            return None, 0.0
        load_kwargs["quantization_config"] = quant_config.bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        # FP16 loading
        load_kwargs["torch_dtype"] = quant_config.torch_dtype
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Move to device if not using device_map
    if device != "cuda" or quant_config.bnb_config is None:
        if device == "cpu":
            model = model.to(device)

    model.eval()

    # Measure memory after loading
    if device == "cuda":
        torch.cuda.synchronize()
        mem_after = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        mem_after = get_cpu_memory_mb()

    memory_used = mem_after - mem_before
    load_time = time.time() - start_time

    print(f"Done ({load_time:.1f}s, {memory_used:.0f} MB)")

    return model, memory_used


# =============================================================================
# BENCHMARKING FUNCTIONS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from benchmarking a single quantization configuration."""
    quant_name: str
    bits: int

    # Memory metrics
    model_memory_mb: float
    peak_memory_mb: float

    # Latency metrics (in milliseconds)
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    time_to_first_token_ms: float

    # Throughput metrics
    tokens_per_second: float

    # Quality metrics
    sample_outputs: List[str] = field(default_factory=list)

    # Derived metrics (filled in later)
    memory_reduction: float = 1.0      # vs FP16
    speedup: float = 1.0               # vs FP16


def benchmark_inference(
    model,
    tokenizer,
    prompts: List[str],
    config: BenchmarkConfig,
    quant_name: str,
) -> BenchmarkResult:
    """
    Run inference benchmark on a model.

    Measures:
    - Latency: Time per generation
    - Time to first token: Initial response latency
    - Throughput: Tokens generated per second
    - Memory: Peak memory during inference
    """
    device = config.device
    latencies = []
    ttft_times = []  # Time to first token
    total_tokens = 0
    sample_outputs = []

    # Warmup runs (not counted)
    print(f"    Warming up ({config.num_warmup_runs} runs)...", end=" ", flush=True)
    for prompt in prompts[:config.num_warmup_runs]:
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id,
            )
    print("Done")

    # Reset memory stats for actual benchmark
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Actual benchmark runs
    print(f"    Benchmarking ({config.num_benchmark_runs} runs)...", end=" ", flush=True)

    for run_idx in range(config.num_benchmark_runs):
        for prompt_idx, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt")
            input_length = inputs["input_ids"].shape[1]

            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
                torch.cuda.synchronize()

            # Measure time to first token
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,  # Need this for TTFT measurement
                )

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Calculate metrics
            generation_time = (end_time - start_time) * 1000  # Convert to ms
            output_length = outputs.sequences.shape[1]
            new_tokens = output_length - input_length

            latencies.append(generation_time)
            total_tokens += new_tokens

            # Estimate TTFT (approximate - first token time)
            # In practice, use streaming for accurate TTFT
            ttft_times.append(generation_time / max(new_tokens, 1))

            # Save sample output (first run only)
            if run_idx == 0 and config.save_outputs:
                decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                sample_outputs.append(decoded)

    print("Done")

    # Get peak memory
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory = get_cpu_memory_mb()

    # Calculate statistics
    latency_mean = statistics.mean(latencies)
    latency_std = statistics.stdev(latencies) if len(latencies) > 1 else 0
    sorted_latencies = sorted(latencies)
    latency_p50 = sorted_latencies[len(sorted_latencies) // 2]
    latency_p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]

    total_time_s = sum(latencies) / 1000
    tokens_per_second = total_tokens / total_time_s if total_time_s > 0 else 0

    return BenchmarkResult(
        quant_name=quant_name,
        bits=16 if quant_name == "FP16" else (8 if quant_name == "INT8" else 4),
        model_memory_mb=0,  # Filled in by caller
        peak_memory_mb=peak_memory,
        latency_mean_ms=latency_mean,
        latency_std_ms=latency_std,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        time_to_first_token_ms=statistics.mean(ttft_times),
        tokens_per_second=tokens_per_second,
        sample_outputs=sample_outputs,
    )


# =============================================================================
# QUALITY ANALYSIS
# =============================================================================

def analyze_quality_degradation(results: Dict[str, BenchmarkResult]) -> str:
    """
    Analyze and comment on quality differences between quantization levels.

    This is a heuristic analysis based on output characteristics.
    For rigorous evaluation, use proper benchmarks (MMLU, HellaSwag, etc.)
    """

    commentary = []
    commentary.append("\n" + "="*70)
    commentary.append("QUALITY DEGRADATION ANALYSIS")
    commentary.append("="*70)

    if "FP16" not in results:
        commentary.append("Cannot analyze quality without FP16 baseline.")
        return "\n".join(commentary)

    fp16_outputs = results["FP16"].sample_outputs

    commentary.append("""
IMPORTANT NOTES ON QUANTIZATION QUALITY:
----------------------------------------
1. INT8 (8-bit): Generally preserves 99%+ of model quality
   - Uses mixed-precision for outlier features (LLM.int8())
   - Minimal perplexity increase (~0.1-0.5 points)
   - Safe for most production use cases

2. INT4 (4-bit): Quality depends on quantization method
   - NF4 (NormalFloat4): Optimized for normally distributed weights
   - GPTQ: Post-training quantization, good for inference
   - AWQ: Activation-aware, often best quality at 4-bit
   - Typical perplexity increase: 0.5-2.0 points
   - May struggle with: precise numerical reasoning, rare tokens,
     long-context coherence, and domain-specific terminology

3. Quality Degradation Patterns:
   - Factual accuracy: Usually preserved at INT8, may degrade at INT4
   - Reasoning chains: INT4 may produce shorter or less coherent chains
   - Code generation: Syntax usually fine, logic may have subtle errors
   - Creative writing: Often indistinguishable across quantization levels
   - Math/numbers: Most sensitive to quantization artifacts
""")

    # Compare outputs if available
    if len(fp16_outputs) > 0 and len(results.get("INT8", BenchmarkResult("", 0, 0, 0, 0, 0, 0, 0, 0, 0)).sample_outputs) > 0:
        commentary.append("\nSAMPLE OUTPUT COMPARISON:")
        commentary.append("-" * 40)

        for i, prompt in enumerate(BENCHMARK_PROMPTS[:3]):  # Show first 3
            commentary.append(f"\nPrompt: \"{prompt[:50]}...\"")
            commentary.append(f"  FP16: {fp16_outputs[i][len(prompt):len(prompt)+100]}...")

            for quant_name in ["INT8", "INT4"]:
                if quant_name in results and len(results[quant_name].sample_outputs) > i:
                    output = results[quant_name].sample_outputs[i]
                    commentary.append(f"  {quant_name}: {output[len(prompt):len(prompt)+100]}...")

    commentary.append("""
RECOMMENDATIONS:
----------------
- Production serving: INT8 offers best quality/efficiency trade-off
- Memory-constrained: INT4 with NF4 or AWQ for acceptable quality
- Quality-critical: Use FP16 or FP32, consider model distillation
- Always benchmark on YOUR specific use case before deploying
""")

    return "\n".join(commentary)


# =============================================================================
# RESULTS FORMATTING
# =============================================================================

def format_results_table(results: Dict[str, BenchmarkResult]) -> str:
    """Format benchmark results as a comparison table."""

    if not results:
        return "No results to display."

    # Determine which columns we have (in order of precision)
    all_columns = ["FP32", "FP16", "INT8", "INT4"]
    columns = [c for c in all_columns if c in results]

    if not columns:
        return "No valid results to display."

    # Calculate relative metrics (vs highest precision baseline)
    baseline_name = columns[0]  # FP32 or FP16
    baseline = results[baseline_name]
    for name, result in results.items():
        if baseline.model_memory_mb > 0:
            result.memory_reduction = baseline.model_memory_mb / max(result.model_memory_mb, 1)
        if result.latency_mean_ms > 0:
            result.speedup = baseline.latency_mean_ms / result.latency_mean_ms

    lines = []
    col_width = 14
    table_width = 32 + (col_width + 3) * len(columns)

    lines.append("\n" + "="*table_width)
    lines.append(" " * 15 + "QUANTIZATION BENCHMARK RESULTS")
    lines.append("="*table_width)

    # Header
    header = f"{'Metric':<30} |"
    for col in columns:
        header += f" {col:>{col_width}} |"
    lines.append(f"\n{header}")
    lines.append("-"*30 + "-+" + ("-"*col_width + "-+") * len(columns))

    def get_val(result, attr, fmt=".0f"):
        if result is None:
            return "-"
        val = getattr(result, attr, 0)
        if fmt == ".0f":
            return f"{val:,.0f}"
        elif fmt == ".1f":
            return f"{val:,.1f}"
        elif fmt == ".2fx":
            return f"{val:.2f}x"
        return str(val)

    def add_row(metric, attr, fmt=".0f"):
        row = f"{metric:<30} |"
        for col in columns:
            result = results.get(col)
            val = get_val(result, attr, fmt)
            row += f" {val:>{col_width}} |"
        lines.append(row)

    # Memory section
    lines.append(f"\n{'MEMORY':^{table_width}}")
    lines.append("-"*table_width)
    add_row("Model Size (MB)", "model_memory_mb", ".0f")
    add_row("Peak Memory (MB)", "peak_memory_mb", ".0f")
    add_row("Memory Reduction", "memory_reduction", ".2fx")

    # Latency section
    lines.append(f"\n{'LATENCY':^{table_width}}")
    lines.append("-"*table_width)
    add_row("Mean Latency (ms)", "latency_mean_ms", ".1f")
    add_row("Std Dev (ms)", "latency_std_ms", ".1f")
    add_row("P50 Latency (ms)", "latency_p50_ms", ".1f")
    add_row("P95 Latency (ms)", "latency_p95_ms", ".1f")
    add_row("Time to First Token (ms)", "time_to_first_token_ms", ".1f")

    # Throughput section
    lines.append(f"\n{'THROUGHPUT':^{table_width}}")
    lines.append("-"*table_width)
    add_row("Tokens/Second", "tokens_per_second", ".1f")
    add_row("Relative Speed", "speedup", ".2fx")

    lines.append("\n" + "="*table_width)

    # Summary
    lines.append("\nSUMMARY:")
    lines.append("-"*50)

    for i, col in enumerate(columns[1:], 1):  # Skip baseline
        result = results[col]
        lines.append(f"{col} vs {baseline_name}: "
                    f"{result.memory_reduction:.2f}x memory reduction, "
                    f"{result.speedup:.2f}x speed")

    return "\n".join(lines)


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_benchmark(config: BenchmarkConfig) -> Dict[str, BenchmarkResult]:
    """Run the complete benchmark suite."""

    print("\n" + "="*70)
    print(" QUANTIZATION INFERENCE BENCHMARK")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Warmup runs: {config.num_warmup_runs}")
    print(f"  Benchmark runs: {config.num_benchmark_runs}")
    print(f"  Max new tokens: {config.max_new_tokens}")
    print(f"  Number of prompts: {len(BENCHMARK_PROMPTS)}")

    # Load tokenizer (shared across all quantization levels)
    print("\nLoading tokenizer...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Done")

    # Get quantization configurations
    include_quantized = config.device == "cuda" and BNB_AVAILABLE

    if not include_quantized:
        if config.device == "cpu":
            print("\nNOTE: Running on CPU - comparing FP32 vs FP16 (INT8/INT4 require CUDA)")
        elif not BNB_AVAILABLE:
            print("\nNOTE: bitsandbytes not installed - comparing FP32 vs FP16")
            print("      Install with: pip install bitsandbytes (CUDA required)")

    quant_configs = get_quantization_configs(include_quantized=include_quantized)

    results = {}

    # Benchmark each quantization level
    for quant_name, quant_config in quant_configs.items():
        print(f"\n{'='*70}")
        print(f"Benchmarking: {quant_name} ({quant_config.description})")
        print(f"{'='*70}")

        # Load model
        model, model_memory = load_model_with_quantization(
            config.model_name,
            quant_config,
            config.device,
        )

        if model is None:
            print(f"  Skipping {quant_name} (failed to load)")
            continue

        # Run benchmark
        result = benchmark_inference(
            model=model,
            tokenizer=tokenizer,
            prompts=BENCHMARK_PROMPTS,
            config=config,
            quant_name=quant_name,
        )
        result.model_memory_mb = model_memory

        results[quant_name] = result

        # Clean up to free memory for next model
        del model
        clear_memory()

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quantization Inference Benchmark")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name or path (default: gpt2)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run on (default: auto)")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of benchmark runs (default: 10)")
    parser.add_argument("--tokens", type=int, default=50,
                       help="Max new tokens to generate (default: 50)")

    args = parser.parse_args()

    # Configure
    config = BenchmarkConfig(
        model_name=args.model,
        device=args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
        num_benchmark_runs=args.runs,
        max_new_tokens=args.tokens,
    )

    # Run benchmark
    results = run_benchmark(config)

    # Print results table
    print(format_results_table(results))

    # Print quality analysis
    print(analyze_quality_degradation(results))

    # Print conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The benchmark demonstrates the classic trade-off:

  MEMORY ←――――――――――――――――――――――――――――――――→ QUALITY
    ↑                                          ↑
   INT4        INT8                          FP16
  (4x smaller) (2x smaller)                (baseline)

Choose based on your constraints:
- Memory-limited deployment → INT4 with quality monitoring
- Balanced production use  → INT8 (best trade-off)
- Quality-critical tasks   → FP16/FP32

NOTE: Actual speedup varies by hardware. On modern GPUs:
- INT8 may be SLOWER due to dequantization overhead
- INT4 with specialized kernels (GPTQ, AWQ) can be faster
- Batch size significantly affects throughput metrics
""")

    return results


if __name__ == "__main__":
    results = main()
