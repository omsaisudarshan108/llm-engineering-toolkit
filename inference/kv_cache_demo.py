"""
KV Cache Demonstration: Accelerating Autoregressive Decoding
=============================================================

This demo shows how KV caching dramatically reduces inference time
in token-by-token generation.

THE PROBLEM WITH NAIVE AUTOREGRESSIVE GENERATION:
-------------------------------------------------
To generate token N, we need attention over ALL previous tokens.
Without caching, we recompute Q, K, V for tokens 1..N-1 EVERY time.

Example: Generating 100 tokens
  - Token 1: Compute attention for 1 token
  - Token 2: Recompute for 2 tokens (redundant: token 1 again)
  - Token 3: Recompute for 3 tokens (redundant: tokens 1,2 again)
  - ...
  - Token 100: Recompute for 100 tokens (redundant: 99 tokens again!)

Total compute: 1 + 2 + 3 + ... + 100 = 5,050 token computations
But we only generated 100 NEW tokens!

THE SOLUTION - KV CACHING:
--------------------------
Cache the K and V values for all previous tokens.
When generating token N, only compute Q, K, V for the NEW token.

  - Token 1: Compute Q1, K1, V1. Cache K1, V1
  - Token 2: Compute Q2, K2, V2. Attend to cached [K1,K2], [V1,V2]
  - Token 3: Compute Q3, K3, V3. Attend to cached [K1,K2,K3], [V1,V2,V3]
  - ...

Total compute: 100 new token computations (100x reduction!)

MEMORY vs LATENCY TRADEOFF:
---------------------------
KV Cache requires storing K, V for every layer and every past token.
Memory = num_layers × seq_len × 2 × d_model × sizeof(dtype)

For a 7B model generating 2048 tokens:
  - Memory: 32 layers × 2048 × 2 × 4096 × 2 bytes ≈ 1 GB per sequence!

This is why:
  - Long context = high memory (KV cache grows linearly)
  - Batch inference = memory × batch_size
  - vLLM's PagedAttention optimizes this (more later)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


# =============================================================================
# SIMPLIFIED TRANSFORMER COMPONENTS
# =============================================================================

class CachedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with KV caching support.

    KV CACHE EXPLANATION:
    ---------------------
    During autoregressive generation:
    1. First token: Compute full attention, cache K, V
    2. Subsequent tokens: Only compute Q for new token,
       use cached K, V for context

    The cache stores:
    - past_key: All previous K values (batch, num_heads, past_len, d_k)
    - past_value: All previous V values (batch, num_heads, past_len, d_v)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,                        # Current input (batch, seq_len, d_model)
        past_key_value: Optional[Tuple] = None,  # Cached K, V from previous steps
        use_cache: bool = False,                 # Whether to return updated cache
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass with optional KV caching.

        Args:
            x: Input tensor. For cached inference, this is just the NEW token(s)
            past_key_value: Tuple of (past_key, past_value) from previous steps
            use_cache: If True, return updated cache for next step

        Returns:
            output: Attention output
            present_key_value: Updated cache (if use_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # =====================================================================
        # Compute Q, K, V for current input
        # =====================================================================
        # NOTE: During cached generation, x is only the NEW token(s)
        # So we only compute projections for new positions, not the full context
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)  # (batch, seq_len, d_model)
        V = self.W_V(x)  # (batch, seq_len, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, d_k)

        # =====================================================================
        # KV Cache: Concatenate with past keys and values
        # =====================================================================
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Concatenate past and current along sequence dimension
            # past_key: (batch, num_heads, past_len, d_k)
            # K: (batch, num_heads, 1, d_k)  <- just the new token
            # Result: (batch, num_heads, past_len + 1, d_k)
            K = torch.cat([past_key, K], dim=2)
            V = torch.cat([past_value, V], dim=2)

        # =====================================================================
        # Compute attention
        # =====================================================================
        # Q: (batch, num_heads, new_len, d_k) - usually new_len=1 during generation
        # K: (batch, num_heads, total_len, d_k) - includes all past tokens
        # Scores: (batch, num_heads, new_len, total_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask: new tokens can only attend to previous positions
        total_len = K.size(2)
        new_len = Q.size(2)

        # Create causal mask for the query positions
        # For generation (new_len=1), this allows attending to all past tokens
        if new_len > 1:
            causal_mask = torch.triu(
                torch.ones(new_len, total_len, device=x.device),
                diagonal=total_len - new_len + 1
            )
            causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
            scores = scores + causal_mask

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, new_len, self.d_model)
        output = self.W_O(output)

        # =====================================================================
        # Return cache for next step
        # =====================================================================
        if use_cache:
            present_key_value = (K, V)  # Store full K, V including new token
        else:
            present_key_value = None

        return output, present_key_value


class CachedTransformerBlock(nn.Module):
    """Transformer block with KV caching support."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attention = CachedMultiHeadAttention(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:

        # Attention with residual
        normed = self.norm1(x)
        attn_output, present = self.attention(normed, past_key_value, use_cache)
        x = x + attn_output

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x, present


class CachedTransformer(nn.Module):
    """Simple transformer with KV caching for demonstration."""

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CachedTransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """
        Forward pass with optional KV caching.

        Args:
            input_ids: Token IDs (batch, seq_len)
            past_key_values: List of (past_key, past_value) for each layer
            use_cache: Whether to return updated caches

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            present_key_values: Updated caches for each layer
        """
        x = self.embedding(input_ids)

        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, present = layer(x, past_kv, use_cache)

            if use_cache:
                present_key_values.append(present)

        x = self.norm(x)
        logits = self.head(x)

        return logits, present_key_values


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_without_cache(
    model: CachedTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, float]:
    """
    Generate tokens WITHOUT KV caching (inefficient baseline).

    This recomputes attention over ALL previous tokens for each new token.
    Computational complexity: O(n²) where n = sequence length
    """
    model.eval()
    generated = input_ids.clone()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Feed ENTIRE sequence through model (inefficient!)
            # Every token recomputes Q, K, V for ALL previous tokens
            logits, _ = model(generated, use_cache=False)

            # Get next token (greedy decoding)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    elapsed = time.perf_counter() - start_time

    return generated, elapsed


def generate_with_cache(
    model: CachedTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, float]:
    """
    Generate tokens WITH KV caching (efficient).

    Only computes Q, K, V for the NEW token each step.
    Uses cached K, V from previous steps for context.
    Computational complexity: O(n) where n = sequence length
    """
    model.eval()
    generated = input_ids.clone()
    past_key_values = None

    start_time = time.perf_counter()

    with torch.no_grad():
        # First pass: process entire prompt, build initial cache
        logits, past_key_values = model(generated, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        # Subsequent passes: only process NEW token, use cache for context
        for _ in range(max_new_tokens - 1):
            # Only feed the LAST token (new one)
            # past_key_values contains K, V for all previous tokens
            logits, past_key_values = model(
                next_token,  # Just the new token!
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    elapsed = time.perf_counter() - start_time

    return generated, elapsed


# =============================================================================
# DEMONSTRATION
# =============================================================================

def calculate_kv_cache_memory(
    num_layers: int,
    seq_len: int,
    d_model: int,
    batch_size: int = 1,
    dtype_bytes: int = 4,  # float32 = 4, float16 = 2
) -> float:
    """Calculate KV cache memory usage in MB."""
    # Each layer stores K and V, each of shape (batch, num_heads, seq_len, d_k)
    # Total: num_layers × 2 × batch × seq_len × d_model × dtype_bytes
    bytes_used = num_layers * 2 * batch_size * seq_len * d_model * dtype_bytes
    return bytes_used / (1024 * 1024)


def main():
    print("="*70)
    print("  KV CACHE DEMONSTRATION: MEMORY vs LATENCY TRADEOFF")
    print("="*70)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("MODEL CONFIGURATION")
    print("-"*70)

    vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 4
    prompt_length = 32
    max_new_tokens = 64

    model = CachedTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"""
    Vocabulary size: {vocab_size}
    Model dimension: {d_model}
    Number of heads: {num_heads}
    Number of layers: {num_layers}
    Total parameters: {total_params:,}

    Prompt length: {prompt_length} tokens
    Tokens to generate: {max_new_tokens}
    """)

    # Create random prompt
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, prompt_length))

    # -------------------------------------------------------------------------
    # Generate WITHOUT cache (baseline)
    # -------------------------------------------------------------------------
    print("-"*70)
    print("GENERATION WITHOUT KV CACHE (Inefficient)")
    print("-"*70)

    print("""
    Without caching, each step recomputes attention for ALL tokens:
      Step 1: Attend to 32 tokens (prompt)
      Step 2: Attend to 33 tokens (prompt + 1 new)
      Step 3: Attend to 34 tokens (prompt + 2 new)
      ...
      Step 64: Attend to 95 tokens (prompt + 63 new)

    Total attention computations: 32 + 33 + 34 + ... + 95 = {sum(range(prompt_length, prompt_length + max_new_tokens)):,}
    """)

    output_no_cache, time_no_cache = generate_without_cache(
        model, input_ids, max_new_tokens
    )

    print(f"    Time elapsed: {time_no_cache*1000:.1f} ms")
    print(f"    Tokens generated: {max_new_tokens}")
    print(f"    Speed: {max_new_tokens/time_no_cache:.1f} tokens/sec")

    # -------------------------------------------------------------------------
    # Generate WITH cache (optimized)
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("GENERATION WITH KV CACHE (Efficient)")
    print("-"*70)

    print("""
    With caching, each step only computes the NEW token:
      Step 1: Compute Q,K,V for 32 tokens, cache K,V
      Step 2: Compute Q,K,V for 1 NEW token, use cached K,V
      Step 3: Compute Q,K,V for 1 NEW token, use cached K,V
      ...

    Total NEW token computations: 32 + 1 + 1 + ... = {32 + max_new_tokens - 1}
    (But we DO need attention over full context each step)
    """)

    output_with_cache, time_with_cache = generate_with_cache(
        model, input_ids, max_new_tokens
    )

    print(f"    Time elapsed: {time_with_cache*1000:.1f} ms")
    print(f"    Tokens generated: {max_new_tokens}")
    print(f"    Speed: {max_new_tokens/time_with_cache:.1f} tokens/sec")

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("TIMING COMPARISON")
    print("-"*70)

    speedup = time_no_cache / time_with_cache

    print(f"""
    ┌─────────────────────────┬────────────────┬────────────────┐
    │ Metric                  │  Without Cache │   With Cache   │
    ├─────────────────────────┼────────────────┼────────────────┤
    │ Time (ms)               │ {time_no_cache*1000:>12.1f}   │ {time_with_cache*1000:>12.1f}   │
    │ Tokens/second           │ {max_new_tokens/time_no_cache:>12.1f}   │ {max_new_tokens/time_with_cache:>12.1f}   │
    │ Speedup                 │          1.0x  │ {speedup:>12.1f}x  │
    └─────────────────────────┴────────────────┴────────────────┘
    """)

    # -------------------------------------------------------------------------
    # Memory analysis
    # -------------------------------------------------------------------------
    print("-"*70)
    print("MEMORY vs LATENCY TRADEOFF")
    print("-"*70)

    final_seq_len = prompt_length + max_new_tokens
    cache_memory = calculate_kv_cache_memory(
        num_layers, final_seq_len, d_model, batch_size=1, dtype_bytes=4
    )

    print(f"""
    KV Cache Memory Usage:
      Formula: num_layers × 2 × batch × seq_len × d_model × dtype_bytes
      = {num_layers} × 2 × 1 × {final_seq_len} × {d_model} × 4 bytes
      = {cache_memory:.3f} MB

    For real models (this is why memory matters):
    ┌────────────────────┬─────────────┬────────────────────────────┐
    │ Model              │ Context Len │ KV Cache Memory (FP16)     │
    ├────────────────────┼─────────────┼────────────────────────────┤
    │ GPT-2 (124M)       │ 1024        │ ~50 MB per sequence        │
    │ Llama-7B           │ 4096        │ ~1 GB per sequence         │
    │ Llama-70B          │ 4096        │ ~10 GB per sequence        │
    │ GPT-4 (estimated)  │ 128K        │ ~100+ GB per sequence!     │
    └────────────────────┴─────────────┴────────────────────────────┘

    THE TRADEOFF:
    ─────────────────────────────────────────────────────────────────
    Memory:   KV cache grows linearly with sequence length
              Cache size = O(layers × seq_len × d_model)

    Latency:  Without cache: O(n²) attention per token
              With cache:    O(n) attention per token

    This is why:
    1. Long context models need LOTS of memory
    2. Batch inference is memory-bound (cache × batch_size)
    3. vLLM's PagedAttention was invented (allocate cache dynamically)
    """)

    # -------------------------------------------------------------------------
    # Verify outputs match
    # -------------------------------------------------------------------------
    print("-"*70)
    print("VERIFICATION")
    print("-"*70)

    outputs_match = torch.equal(output_no_cache, output_with_cache)
    print(f"    Outputs match: {outputs_match}")
    if outputs_match:
        print("    ✓ KV caching produces identical results (just faster!)")
    else:
        print("    ✗ WARNING: Outputs differ (check implementation)")

    # -------------------------------------------------------------------------
    # Scaling analysis
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("SCALING ANALYSIS: How speedup grows with sequence length")
    print("-"*70)

    print("\n    Generating different lengths to show O(n²) vs O(n) scaling:\n")

    for new_tokens in [16, 32, 64, 128]:
        input_ids = torch.randint(0, vocab_size, (1, prompt_length))

        _, t_no_cache = generate_without_cache(model, input_ids, new_tokens)
        _, t_with_cache = generate_with_cache(model, input_ids, new_tokens)

        speedup = t_no_cache / t_with_cache
        print(f"    {new_tokens} tokens: Without={t_no_cache*1000:6.1f}ms, "
              f"With={t_with_cache*1000:6.1f}ms, Speedup={speedup:.1f}x")

    print("""
    Notice: Speedup INCREASES with sequence length!
    This is because:
      - Without cache: O(n²) → grows quadratically
      - With cache: O(n) → grows linearly
    """)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("="*70)
    print("  SUMMARY: KV CACHING")
    print("="*70)
    print("""
    WHAT KV CACHING DOES:
    • Stores Key (K) and Value (V) tensors from previous tokens
    • Reuses them instead of recomputing
    • Only computes Q, K, V for NEW tokens

    BENEFITS:
    • Dramatically faster inference (especially for long sequences)
    • Speedup grows with sequence length (O(n²) → O(n))
    • Essential for practical LLM deployment

    COSTS:
    • Memory usage scales with: layers × seq_len × d_model × batch_size
    • Long contexts require substantial memory
    • Batch inference is often memory-bound

    SOLUTIONS FOR MEMORY:
    • PagedAttention (vLLM): Allocate cache in pages, share across requests
    • Sliding window attention: Only cache recent tokens (Mistral)
    • Multi-Query Attention: Share K, V across heads (reduces cache size)
    • Quantized KV cache: Store cache in INT8 instead of FP16
    """)


if __name__ == "__main__":
    main()
