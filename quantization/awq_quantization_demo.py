"""
Activation-Aware Quantization (AWQ) Demonstration
==================================================

This script demonstrates AWQ quantization on transformer models with
layman-friendly explanations of each step and what problems they solve.

WHAT IS AWQ? (Layman Explanation)
---------------------------------
Imagine you're packing for a trip with a tiny suitcase (limited memory).
You can't take everything, so you need to compress your clothes.

NAIVE APPROACH: Squish everything equally → Some delicate items get ruined
AWQ APPROACH: Figure out which items are "delicate" (important weights) and
              protect them while compressing the rest more aggressively

AWQ looks at how the model actually behaves on real data (activations) to
find which weights are most important, then preserves those more carefully.

Author: AI Engineer Example
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# For actual AWQ, you'd use: from awq import AutoAWQForCausalLM
# We'll implement a simplified version to demonstrate the concepts


# =============================================================================
# LAYMAN EXPLANATION HELPER
# =============================================================================

def explain(title: str, explanation: str, problem_solved: str = ""):
    """Print a formatted explanation for each step."""
    print("\n" + "="*70)
    print(f"📚 {title}")
    print("="*70)
    print(f"\n{explanation}")
    if problem_solved:
        print(f"\n💡 PROBLEM SOLVED: {problem_solved}")
    print()


# =============================================================================
# STEP 1: UNDERSTANDING THE PROBLEM
# =============================================================================

explain(
    "STEP 1: WHY DO WE NEED QUANTIZATION?",
    """
    Large language models (LLMs) are like massive dictionaries with billions
    of numbers (weights). Each number is typically stored as a 16-bit or
    32-bit floating point value.

    Example: A 7B parameter model
    - FP16 (16-bit): 7B × 2 bytes = 14 GB of memory
    - FP32 (32-bit): 7B × 4 bytes = 28 GB of memory

    Most consumer GPUs have 8-16 GB of memory. We can't fit these models!

    QUANTIZATION = Reducing the precision of these numbers
    - INT8 (8-bit):  7B × 1 byte = 7 GB  (2x smaller)
    - INT4 (4-bit):  7B × 0.5 bytes = 3.5 GB (4x smaller)

    But there's a catch: Lower precision = potential quality loss.
    AWQ solves this by being SMART about which weights to protect.
    """,
    "Fitting large models on consumer hardware without destroying quality"
)


# =============================================================================
# STEP 2: THE CALIBRATION DATASET
# =============================================================================

explain(
    "STEP 2: THE CALIBRATION DATASET - Teaching AWQ What's Important",
    """
    Before quantizing, AWQ needs to "watch" the model process some text.
    This is called CALIBRATION. It's like a teacher observing students
    to understand which concepts they rely on most.

    WHY WE CHOOSE SPECIFIC CALIBRATION DATA:
    ----------------------------------------
    1. Representative: Should match how the model will be used
       - General model → Wikipedia, books, diverse web text
       - Code model → Programming samples
       - Chat model → Conversation examples

    2. Diverse: Cover different topics, lengths, and styles
       - Short prompts + long documents
       - Technical + casual language
       - Questions + statements

    3. Size: Usually 128-512 samples is enough
       - Too few: AWQ can't identify important patterns
       - Too many: Diminishing returns, slower calibration

    COMMON CALIBRATION DATASETS:
    - Pile (general): Diverse internet text
    - C4 (general): Cleaned Common Crawl
    - Wikitext (knowledge): Wikipedia articles
    - CodeParrot (code): GitHub code samples
    """,
    "AWQ learns which weights matter by watching the model think"
)

# Example calibration data (in practice, use hundreds of samples)
CALIBRATION_DATA = [
    # Diverse topics
    "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
    "In financial markets, volatility is measured using standard deviation of returns over a specific period.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight energy.",
    "The Python programming language emphasizes code readability with its use of significant whitespace.",
    "Quantum computers use qubits that can exist in superposition, enabling parallel computation.",
    "The French Revolution of 1789 fundamentally transformed European political structures.",
    "Machine learning models learn patterns from data without being explicitly programmed.",
    "Climate change is driven primarily by greenhouse gas emissions from human activities.",
    # Different lengths and styles
    "What is the capital of France?",
    "Explain the concept of neural networks in simple terms that a beginner could understand.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The patient presented with symptoms consistent with acute respiratory infection.",
]

print(f"Calibration dataset: {len(CALIBRATION_DATA)} samples")
print(f"Sample: \"{CALIBRATION_DATA[0][:50]}...\"")


# =============================================================================
# STEP 3: SIMPLIFIED TRANSFORMER FOR DEMONSTRATION
# =============================================================================

explain(
    "STEP 3: THE MODEL ARCHITECTURE - What Gets Quantized",
    """
    A transformer has two main types of layers that we quantize:

    1. ATTENTION LAYERS (q_proj, k_proj, v_proj, o_proj)
       - These decide "what to pay attention to" in the input
       - Like a spotlight that highlights relevant words
       - VERY sensitive to quantization errors

    2. MLP/FFN LAYERS (gate_proj, up_proj, down_proj)
       - These transform the information
       - Like a processing factory that refines the data
       - Slightly more robust to quantization

    AWQ treats these differently because they have different importance!

    ARCHITECTURE DIAGRAM:
    ┌─────────────────────────────────────────┐
    │              Input Tokens               │
    └─────────────────┬───────────────────────┘
                      ▼
    ┌─────────────────────────────────────────┐
    │         ATTENTION BLOCK                 │
    │  ┌───────┐ ┌───────┐ ┌───────┐         │
    │  │Q_proj │ │K_proj │ │V_proj │ ← Quantize│
    │  └───┬───┘ └───┬───┘ └───┬───┘         │
    │      └────────┼────────┘               │
    │               ▼                         │
    │      Self-Attention                     │
    │               ▼                         │
    │         ┌─────────┐                     │
    │         │ O_proj  │ ← Quantize          │
    │         └────┬────┘                     │
    └──────────────┼──────────────────────────┘
                   ▼
    ┌─────────────────────────────────────────┐
    │            MLP BLOCK                    │
    │  ┌──────────┐  ┌─────────┐              │
    │  │Gate_proj │  │Up_proj  │ ← Quantize   │
    │  └────┬─────┘  └────┬────┘              │
    │       └──────┬──────┘                   │
    │              ▼                          │
    │       ┌───────────┐                     │
    │       │Down_proj  │ ← Quantize          │
    │       └─────┬─────┘                     │
    └─────────────┼───────────────────────────┘
                  ▼
              Next Layer...
    """,
    "Understanding which parts of the model can be compressed"
)


class SimplifiedAttention(nn.Module):
    """Simplified attention layer for demonstration."""
    def __init__(self, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # These are the weight matrices we'll quantize
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Simplified attention (no proper multi-head for demo)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        return self.o_proj(out)


class SimplifiedMLP(nn.Module):
    """Simplified MLP layer for demonstration."""
    def __init__(self, hidden_size: int = 256, intermediate_size: int = 1024):
        super().__init__()
        # These are the weight matrices we'll quantize
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU activation (used in Llama/Mistral)
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SimplifiedTransformerBlock(nn.Module):
    """One transformer block with attention and MLP."""
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.attention = SimplifiedAttention(hidden_size)
        self.mlp = SimplifiedMLP(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimplifiedTransformer(nn.Module):
    """Simplified transformer for AWQ demonstration."""
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimplifiedTransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


print("Created simplified transformer model for demonstration")
model = SimplifiedTransformer()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


# =============================================================================
# STEP 4: AWQ - FINDING IMPORTANT WEIGHTS
# =============================================================================

explain(
    "STEP 4: AWQ'S SECRET SAUCE - Activation-Aware Scaling",
    """
    Here's where AWQ gets clever. Instead of treating all weights equally,
    it finds which weights are "salient" (important) by looking at ACTIVATIONS.

    WHAT ARE ACTIVATIONS?
    ---------------------
    When data flows through the model, intermediate values are created.
    These are "activations" - they show how the model is "thinking."

    input → [weight × activation = output] → next layer
                    ↑
            AWQ measures these!

    THE AWQ INSIGHT:
    ----------------
    Some weights always produce LARGE activations → Very important!
    Some weights always produce SMALL activations → Less critical

    AWQ protects the important weights by:
    1. Measuring activation magnitudes during calibration
    2. Finding weights that consistently cause large activations
    3. Scaling these weights UP before quantization
    4. Scaling them back DOWN after quantization

    This is like putting padding around fragile items before squishing them!

    MATHEMATICAL INTUITION:
    -----------------------
    If a weight w is multiplied by large activation a:
    - Quantization error in w gets AMPLIFIED by a
    - So we scale: w' = w × s, then quantize, then x' = x / s
    - Error is now divided by a, minimizing impact!
    """,
    "Identifying and protecting the most important weights in the model"
)


def compute_activation_scales(
    model: nn.Module,
    calibration_inputs: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute activation scales for AWQ.

    This measures how "important" each weight channel is by
    observing activation magnitudes during calibration.
    """
    activation_scales = {}
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            # Track input activation magnitudes
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            if inp is not None and inp.numel() > 0:
                # Average magnitude per channel
                scales = inp.abs().mean(dim=(0, 1)) if inp.dim() == 3 else inp.abs().mean(dim=0)
                if name in activation_scales:
                    activation_scales[name] = activation_scales[name] + scales
                else:
                    activation_scales[name] = scales.clone()
        return fn

    # Register hooks on linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_forward_hook(hook_fn(name))
            hooks.append(h)

    # Run calibration data through model
    model.eval()
    with torch.no_grad():
        for inp in calibration_inputs:
            _ = model(inp)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Normalize scales
    for name in activation_scales:
        activation_scales[name] = activation_scales[name] / len(calibration_inputs)

    return activation_scales


# Create dummy calibration inputs
print("\nRunning calibration to find activation scales...")
dummy_inputs = [torch.randint(0, 1000, (1, 32)) for _ in range(len(CALIBRATION_DATA))]
activation_scales = compute_activation_scales(model, dummy_inputs)

print(f"Computed activation scales for {len(activation_scales)} layers")
sample_layer = list(activation_scales.keys())[0]
print(f"Sample ({sample_layer}): mean={activation_scales[sample_layer].mean():.4f}, "
      f"max={activation_scales[sample_layer].max():.4f}")


# =============================================================================
# STEP 5: QUANTIZATION WITH AWQ PROTECTION
# =============================================================================

explain(
    "STEP 5: PERFORMING AWQ QUANTIZATION",
    """
    Now we apply quantization while protecting important weights.

    QUANTIZATION BASICS:
    --------------------
    Original weight (FP16):  0.0234375 (many decimal places)
    Quantized weight (INT4): 2 (just 0-15 for 4-bit)

    The mapping:
    1. Find min/max of weights in a group
    2. Divide range into 16 buckets (for 4-bit)
    3. Round each weight to nearest bucket

    AWQ ENHANCEMENT:
    ----------------
    Before step 1, scale important weights:
    - Important weight × scale_factor → larger value
    - Quantize (error is relatively smaller now)
    - After loading: divide by scale_factor

    GROUP-WISE QUANTIZATION:
    ------------------------
    Instead of one scale per entire matrix, use one scale per GROUP
    (e.g., every 128 weights share a scale). This improves accuracy.

    TYPICAL AWQ CONFIGURATION:
    - Bits: 4 (INT4)
    - Group size: 128
    - Zero-point: Yes (asymmetric quantization)
    """,
    "Compressing the model while minimizing quality loss"
)


@dataclass
class QuantConfig:
    """Configuration for AWQ-style quantization."""
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True


def quantize_weight_awq(
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    config: QuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight matrix using AWQ-style scaling.

    Returns: (quantized_weight, scales, zero_points)
    """
    # Step 1: Apply activation-aware scaling
    # Protect channels with high activation by scaling them up
    protection_factor = activation_scale / activation_scale.mean()
    protection_factor = protection_factor.clamp(0.5, 2.0)  # Reasonable bounds

    # Scale weights (will be unscaled during inference)
    scaled_weight = weight * protection_factor.unsqueeze(0)

    # Step 2: Group-wise quantization
    out_features, in_features = scaled_weight.shape
    num_groups = (in_features + config.group_size - 1) // config.group_size

    # Pad if necessary
    if in_features % config.group_size != 0:
        pad_size = config.group_size - (in_features % config.group_size)
        scaled_weight = torch.nn.functional.pad(scaled_weight, (0, pad_size))

    # Reshape for group quantization
    scaled_weight = scaled_weight.reshape(out_features, num_groups, config.group_size)

    # Compute scales and zero points per group
    w_min = scaled_weight.min(dim=2, keepdim=True)[0]
    w_max = scaled_weight.max(dim=2, keepdim=True)[0]

    q_max = 2 ** config.bits - 1
    scales = (w_max - w_min) / q_max
    scales = scales.clamp(min=1e-8)

    if config.zero_point:
        zero_points = (-w_min / scales).round().clamp(0, q_max)
    else:
        zero_points = torch.zeros_like(scales)

    # Quantize
    quantized = ((scaled_weight - w_min) / scales).round().clamp(0, q_max)

    return quantized.to(torch.int8), scales.squeeze(-1), zero_points.squeeze(-1), protection_factor


def dequantize_weight(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    protection_factor: torch.Tensor,
    original_in_features: int,
) -> torch.Tensor:
    """Dequantize weights back to floating point."""
    out_features, num_groups, group_size = quantized.shape

    # Dequantize
    dequantized = quantized.float() * scales.unsqueeze(-1)
    if zero_points is not None:
        dequantized = dequantized - zero_points.unsqueeze(-1) * scales.unsqueeze(-1)

    # Reshape back
    dequantized = dequantized.reshape(out_features, -1)[:, :original_in_features]

    # Remove AWQ scaling
    dequantized = dequantized / protection_factor.unsqueeze(0)

    return dequantized


# Demonstrate on one layer
print("\nDemonstrating AWQ quantization on a single layer:")
print("-"*50)

sample_weight = model.layers[0].attention.q_proj.weight.data.clone()
sample_scale = activation_scales.get('layers.0.attention.q_proj',
                                      torch.ones(sample_weight.shape[1]))

config = QuantConfig(bits=4, group_size=128)
quantized, scales, zero_points, protection = quantize_weight_awq(
    sample_weight, sample_scale, config
)

print(f"Original weight shape: {sample_weight.shape}")
print(f"Original dtype: {sample_weight.dtype}")
print(f"Quantized shape: {quantized.shape}")
print(f"Quantized dtype: {quantized.dtype}")
print(f"Scale shape: {scales.shape}")

# Dequantize and compare
dequantized = dequantize_weight(
    quantized, scales, zero_points, protection,
    sample_weight.shape[1]
)

error = (sample_weight - dequantized).abs()
print(f"\nQuantization error:")
print(f"  Mean absolute error: {error.mean():.6f}")
print(f"  Max absolute error: {error.max():.6f}")
print(f"  Relative error: {(error / sample_weight.abs().clamp(min=1e-8)).mean():.4%}")


# =============================================================================
# STEP 6: PERPLEXITY COMPARISON
# =============================================================================

explain(
    "STEP 6: MEASURING QUALITY - Perplexity Comparison",
    """
    WHAT IS PERPLEXITY?
    -------------------
    Perplexity measures how "confused" the model is when predicting text.

    - Lower perplexity = Model is confident and usually correct
    - Higher perplexity = Model is uncertain or wrong

    Example:
    - Perplexity 10: Model chooses between ~10 likely words (good!)
    - Perplexity 100: Model chooses between ~100 words (confused)

    WHY AWQ PRESERVES QUALITY:
    --------------------------
    1. Protects weights that cause large activations
    2. Uses group-wise quantization for finer granularity
    3. Calibration on real data identifies true importance

    TYPICAL PERPLEXITY CHANGES:
    ---------------------------
    | Method      | Perplexity Increase |
    |-------------|---------------------|
    | FP16 → INT8 | +0.1 to +0.5       |
    | FP16 → INT4 | +0.5 to +2.0       |
    | AWQ INT4    | +0.3 to +0.8       | ← Better than naive INT4!
    | GPTQ INT4   | +0.4 to +1.0       |

    AWQ typically achieves the LOWEST perplexity at 4-bit!
    """,
    "Verifying that compression didn't significantly hurt model quality"
)


def compute_perplexity(model: nn.Module, inputs: List[torch.Tensor]) -> float:
    """Compute perplexity on given inputs."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for inp in inputs:
            output = model(inp)
            # Shift for causal LM loss
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


# Compute perplexity before quantization
print("\nComputing perplexity (this measures model quality):")
print("-"*50)

# Create evaluation inputs
eval_inputs = [torch.randint(0, 1000, (1, 64)) for _ in range(20)]

# Before quantization
ppl_before = compute_perplexity(model, eval_inputs)
print(f"Perplexity BEFORE quantization: {ppl_before:.2f}")


# Simulate quantization effect by adding noise proportional to quantization error
def simulate_quantized_model(model, activation_scales, config):
    """Create a simulated quantized model by quantizing and dequantizing weights."""
    model_copy = SimplifiedTransformer()
    model_copy.load_state_dict(model.state_dict())

    with torch.no_grad():
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear) and 'head' not in name:
                weight = module.weight.data
                scale = activation_scales.get(name, torch.ones(weight.shape[1]))

                # Ensure scale has right size
                if scale.shape[0] != weight.shape[1]:
                    scale = torch.ones(weight.shape[1])

                # Quantize and dequantize
                q, s, z, p = quantize_weight_awq(weight, scale, config)
                deq = dequantize_weight(q, s, z, p, weight.shape[1])
                module.weight.data = deq

    return model_copy


# AWQ quantized model
model_awq = simulate_quantized_model(model, activation_scales, QuantConfig(bits=4))
ppl_awq = compute_perplexity(model_awq, eval_inputs)
print(f"Perplexity AFTER AWQ INT4: {ppl_awq:.2f}")

# Naive quantization (no activation awareness)
naive_scales = {name: torch.ones_like(s) for name, s in activation_scales.items()}
model_naive = simulate_quantized_model(model, naive_scales, QuantConfig(bits=4))
ppl_naive = compute_perplexity(model_naive, eval_inputs)
print(f"Perplexity AFTER Naive INT4: {ppl_naive:.2f}")

# Results table
print(f"""
┌────────────────────┬─────────────┬────────────────┐
│ Method             │ Perplexity  │ vs Original    │
├────────────────────┼─────────────┼────────────────┤
│ Original (FP32)    │ {ppl_before:>9.2f}   │     baseline   │
│ AWQ INT4           │ {ppl_awq:>9.2f}   │ {((ppl_awq/ppl_before)-1)*100:>+10.1f}%  │
│ Naive INT4         │ {ppl_naive:>9.2f}   │ {((ppl_naive/ppl_before)-1)*100:>+10.1f}%  │
└────────────────────┴─────────────┴────────────────┘
""")


# =============================================================================
# STEP 7: WHY AWQ PRESERVES QUALITY
# =============================================================================

explain(
    "STEP 7: WHY AWQ WORKS - The Science Behind It",
    """
    AWQ's quality preservation comes from three key insights:

    1. NOT ALL WEIGHTS ARE EQUAL
       -------------------------
       Some weights are "salient" - they consistently produce large outputs.
       Quantization errors in these weights get AMPLIFIED.

       Example:
       - Weight A: always multiplied by activation 0.001 → error × 0.001 = tiny
       - Weight B: always multiplied by activation 100.0 → error × 100.0 = HUGE!

       AWQ protects Weight B while aggressively quantizing Weight A.

    2. ACTIVATION PATTERNS ARE CONSISTENT
       ----------------------------------
       The same weights tend to be important across different inputs.
       This is why calibration works - we can identify important weights
       with just a few hundred samples.

    3. SCALING PRESERVES RELATIVE ACCURACY
       -----------------------------------
       By scaling important weights UP before quantization:
       - They occupy more of the available range (0-15 for INT4)
       - Quantization error becomes relatively smaller
       - After scaling back down, precision is preserved

    COMPARISON WITH OTHER METHODS:
    ------------------------------
    ┌─────────────┬─────────────────────────────────────────────────┐
    │ Method      │ How it works                                    │
    ├─────────────┼─────────────────────────────────────────────────┤
    │ RTN         │ Round-To-Nearest, no intelligence               │
    │ GPTQ        │ Error compensation, layer-by-layer              │
    │ AWQ         │ Activation-aware scaling, protects important    │
    │ SqueezeLLM  │ Non-uniform quantization based on sensitivity   │
    └─────────────┴─────────────────────────────────────────────────┘

    AWQ typically achieves the best quality at 4-bit because it
    directly addresses the amplification problem that causes most
    quantization degradation.
    """,
    "Understanding the theoretical foundation of AWQ's effectiveness"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("  SUMMARY: AWQ QUANTIZATION")
print("="*70)

print("""
WHAT WE DEMONSTRATED:
---------------------
1. ✅ Calibration dataset selection and purpose
2. ✅ Quantization of attention layers (q_proj, k_proj, v_proj, o_proj)
3. ✅ Quantization of MLP layers (gate_proj, up_proj, down_proj)
4. ✅ Perplexity comparison before and after
5. ✅ Why AWQ preserves quality (activation-aware scaling)

KEY TAKEAWAYS:
--------------
• AWQ looks at HOW the model uses weights, not just the weights themselves
• Important weights (high activation) get more protection
• This is done via smart scaling before/after quantization
• Result: INT4 models that perform nearly as well as FP16!

PRACTICAL USAGE:
----------------
For real AWQ quantization, use the official library:

    pip install autoawq

    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
        }
    )
    model.save_quantized("mistral-7b-awq")

MEMORY SAVINGS:
---------------
• Mistral-7B FP16: ~14 GB → AWQ INT4: ~4 GB (3.5x smaller!)
• Llama-70B FP16: ~140 GB → AWQ INT4: ~40 GB (fits on 2x 24GB GPUs!)
""")

print("="*70)
print("  DEMO COMPLETE")
print("="*70)
