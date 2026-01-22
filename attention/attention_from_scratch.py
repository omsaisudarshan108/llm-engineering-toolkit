"""
Scaled Dot-Product Attention from Scratch
==========================================

This implementation links every line of code to the underlying mathematics.

THE ATTENTION EQUATION:
-----------------------
                    Q × K^T
Attention(Q,K,V) = softmax(─────────) × V
                      √d_k

Where:
- Q (Query):  "What am I looking for?" - shape: (batch, seq_len, d_k)
- K (Key):    "What do I contain?"     - shape: (batch, seq_len, d_k)
- V (Value):  "What information do I provide?" - shape: (batch, seq_len, d_v)
- d_k:        Dimension of keys (used for scaling)

WHY SCALE BY √d_k?
------------------
Without scaling, dot products grow large as d_k increases.
Large values push softmax into regions with tiny gradients (vanishing gradient).
Scaling keeps the variance stable regardless of d_k.

Proof: If q, k are random vectors with mean 0, variance 1:
       Var(q·k) = d_k  (variance grows with dimension)
       Var(q·k / √d_k) = 1  (scaling normalizes it)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple


# =============================================================================
# PART 1: BASIC SCALED DOT-PRODUCT ATTENTION
# =============================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,      # Shape: (batch, seq_len_q, d_k)
    key: torch.Tensor,        # Shape: (batch, seq_len_k, d_k)
    value: torch.Tensor,      # Shape: (batch, seq_len_k, d_v)
    mask: Optional[torch.Tensor] = None,  # Shape: (batch, seq_len_q, seq_len_k)
    temperature: float = 1.0,  # Softmax temperature for controlling sharpness
    dropout: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    MATHEMATICAL STEPS:
    1. Compute attention scores: scores = Q × K^T
    2. Scale by √d_k: scaled_scores = scores / √d_k
    3. Apply temperature: tempered_scores = scaled_scores / temperature
    4. Apply mask (optional): masked_scores = tempered_scores + mask
    5. Softmax: attention_weights = softmax(masked_scores)
    6. Apply dropout (training only)
    7. Weighted sum: output = attention_weights × V

    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Optional attention mask (0 = attend, -inf = ignore)
        temperature: Softmax temperature (>1 = softer, <1 = sharper)
        dropout: Dropout probability
        training: Whether in training mode

    Returns:
        output: Attention output
        attention_weights: Attention weight matrix (for visualization)
    """
    # Get dimensions
    d_k = query.size(-1)  # Dimension of keys/queries

    # =========================================================================
    # STEP 1: Compute raw attention scores
    # =========================================================================
    # Formula: scores = Q × K^T
    # Shape: (batch, seq_len_q, d_k) × (batch, d_k, seq_len_k) = (batch, seq_len_q, seq_len_k)
    #
    # Each score[i,j] represents how much query[i] should attend to key[j]
    # Higher score = more relevant = more attention
    scores = torch.matmul(query, key.transpose(-2, -1))

    # =========================================================================
    # STEP 2: Scale by √d_k
    # =========================================================================
    # Formula: scaled_scores = scores / √d_k
    #
    # WHY? Without scaling:
    #   - If d_k = 64, and q,k have unit variance
    #   - Then Var(q·k) = 64 (too large!)
    #   - Softmax of large values → one-hot (gradients vanish)
    #
    # With scaling:
    #   - Var(q·k / √64) = 64/64 = 1 (normalized)
    #   - Softmax works in a good gradient region
    scale = math.sqrt(d_k)
    scaled_scores = scores / scale

    # =========================================================================
    # STEP 3: Apply temperature scaling
    # =========================================================================
    # Formula: tempered_scores = scaled_scores / temperature
    #
    # Temperature controls the "sharpness" of attention:
    #   - temperature > 1: Softer attention (more uniform distribution)
    #   - temperature < 1: Sharper attention (more focused on max)
    #   - temperature = 1: Standard attention
    #
    # Useful for:
    #   - temperature > 1 during training: Encourages exploration
    #   - temperature < 1 during inference: More confident predictions
    if temperature != 1.0:
        scaled_scores = scaled_scores / temperature

    # =========================================================================
    # STEP 4: Apply attention mask (optional)
    # =========================================================================
    # Mask shape: (batch, seq_len_q, seq_len_k) or broadcastable
    # Mask values: 0 = attend, -inf = don't attend
    #
    # Common masks:
    #   - Causal mask: Prevent attending to future tokens (autoregressive)
    #   - Padding mask: Prevent attending to padding tokens
    #
    # After adding -inf, softmax(-inf) = 0, so masked positions get zero weight
    if mask is not None:
        # mask should be 0 for positions to attend, -inf for positions to ignore
        scaled_scores = scaled_scores + mask

    # =========================================================================
    # STEP 5: Apply softmax to get attention weights
    # =========================================================================
    # Formula: attention_weights = softmax(scaled_scores, dim=-1)
    #
    # Softmax converts scores to probabilities:
    #   - All weights are positive
    #   - Weights sum to 1 along the key dimension
    #   - Higher scores → higher weights (exponential relationship)
    #
    # softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    attention_weights = F.softmax(scaled_scores, dim=-1)

    # =========================================================================
    # STEP 6: Apply dropout (training only)
    # =========================================================================
    # Dropout randomly zeros some attention weights
    # This acts as regularization, preventing over-reliance on specific tokens
    if dropout > 0.0 and training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=training)

    # =========================================================================
    # STEP 7: Compute weighted sum of values
    # =========================================================================
    # Formula: output = attention_weights × V
    # Shape: (batch, seq_len_q, seq_len_k) × (batch, seq_len_k, d_v) = (batch, seq_len_q, d_v)
    #
    # Each output position is a weighted combination of all values,
    # where weights indicate relevance determined by query-key matching
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# =============================================================================
# PART 2: MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.

    MATHEMATICAL FORMULATION:
    -------------------------
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

    where head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)

    WHY MULTIPLE HEADS?
    -------------------
    1. Different heads can learn different types of relationships:
       - Head 1: Syntactic dependencies (subject-verb)
       - Head 2: Positional relationships (nearby words)
       - Head 3: Semantic similarities

    2. More expressive than single attention with same parameters:
       - Single attention: one attention pattern
       - Multi-head: h different attention patterns combined

    3. Parallel computation:
       - All heads computed simultaneously
       - No sequential dependency between heads

    DIMENSION BREAKDOWN:
    --------------------
    Input: (batch, seq_len, d_model)

    Per head:
    - Q, K: (batch, seq_len, d_k) where d_k = d_model / num_heads
    - V: (batch, seq_len, d_v) where d_v = d_model / num_heads

    After concat: (batch, seq_len, d_model)
    After W_O: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,        # Model dimension (e.g., 512, 768, 1024)
        num_heads: int,      # Number of attention heads (e.g., 8, 12, 16)
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.d_v = d_model // num_heads

        # =====================================================================
        # Linear projections for Q, K, V
        # =====================================================================
        # These learn what queries, keys, and values to extract from input
        #
        # W_Q: Projects input to queries - "What should I look for?"
        # W_K: Projects input to keys - "How should I be found?"
        # W_V: Projects input to values - "What information should I contribute?"
        #
        # Shape: (d_model, d_model) - projects all heads at once for efficiency
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)

        # =====================================================================
        # Output projection
        # =====================================================================
        # Combines information from all heads back to model dimension
        # This learns how to merge the different "perspectives" from each head
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = dropout

        # For storing attention weights (useful for visualization)
        self.attention_weights = None

    def forward(
        self,
        query: torch.Tensor,              # Shape: (batch, seq_len_q, d_model)
        key: torch.Tensor,                # Shape: (batch, seq_len_k, d_model)
        value: torch.Tensor,              # Shape: (batch, seq_len_k, d_model)
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        For self-attention: query = key = value = input
        For cross-attention: query = decoder, key = value = encoder
        """
        batch_size = query.size(0)

        # =====================================================================
        # Step 1: Linear projections
        # =====================================================================
        # Project input to Q, K, V spaces
        # Each projection learns different aspects of the input
        Q = self.W_Q(query)  # (batch, seq_len_q, d_model)
        K = self.W_K(key)    # (batch, seq_len_k, d_model)
        V = self.W_V(value)  # (batch, seq_len_k, d_model)

        # =====================================================================
        # Step 2: Reshape for multi-head attention
        # =====================================================================
        # Split d_model into (num_heads, d_k) for parallel attention
        # Transpose to get (batch, num_heads, seq_len, d_k)
        #
        # This allows each head to attend independently and in parallel
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Shape after reshape: (batch, num_heads, seq_len, d_k)

        # =====================================================================
        # Step 3: Compute attention for all heads in parallel
        # =====================================================================
        # The scaled_dot_product_attention function handles:
        # - Score computation: Q × K^T
        # - Scaling by √d_k
        # - Masking
        # - Softmax
        # - Weighted sum with V

        # Expand mask for multi-head: (batch, 1, seq_q, seq_k)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension

        output, self.attention_weights = scaled_dot_product_attention(
            Q, K, V,
            mask=mask,
            temperature=temperature,
            dropout=self.dropout,
            training=self.training,
        )

        # =====================================================================
        # Step 4: Concatenate heads
        # =====================================================================
        # Transpose back: (batch, seq_len, num_heads, d_v)
        # Reshape to concat: (batch, seq_len, d_model)
        #
        # This merges information from all heads into a single representation
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        # =====================================================================
        # Step 5: Final linear projection
        # =====================================================================
        # Project concatenated heads back to model dimension
        # This learns how to combine the different "perspectives"
        output = self.W_O(output)

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from last forward pass (for visualization)."""
        return self.attention_weights


# =============================================================================
# PART 3: CAUSAL (AUTOREGRESSIVE) MASK
# =============================================================================

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal attention mask for autoregressive models.

    CAUSAL MASKING:
    ---------------
    In autoregressive models (GPT, decoder-only), each position can only
    attend to previous positions (and itself), not future positions.

    This is because during generation, future tokens don't exist yet!

    Visual representation for seq_len=4:
                    Key positions
                    0    1    2    3
    Query    0  [  0,  -∞,  -∞,  -∞ ]   Position 0 only sees itself
    positions 1  [  0,   0,  -∞,  -∞ ]   Position 1 sees 0, 1
             2  [  0,   0,   0,  -∞ ]   Position 2 sees 0, 1, 2
             3  [  0,   0,   0,   0 ]   Position 3 sees all

    After softmax(-∞) = 0, so masked positions contribute nothing.
    """
    # Create lower triangular matrix of ones
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # Convert to -inf where mask is 1 (upper triangle)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_attention():
    """Demonstrate attention mechanisms with examples."""

    print("="*70)
    print("  SCALED DOT-PRODUCT ATTENTION DEMONSTRATION")
    print("="*70)

    # -------------------------------------------------------------------------
    # Demo 1: Basic attention computation
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("DEMO 1: Basic Scaled Dot-Product Attention")
    print("-"*70)

    batch_size, seq_len, d_k = 1, 4, 8

    # Create simple Q, K, V tensors
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)

    print(f"\nInput shapes:")
    print(f"  Query (Q): {Q.shape} - 'What am I looking for?'")
    print(f"  Key (K):   {K.shape} - 'What do I contain?'")
    print(f"  Value (V): {V.shape} - 'What information do I provide?'")

    output, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    print(f"\nAttention weights (how much each query attends to each key):")
    print(weights[0].numpy().round(3))
    print(f"\nNote: Each row sums to 1.0 (probability distribution)")
    print(f"Row sums: {weights[0].sum(dim=-1).numpy().round(3)}")

    # -------------------------------------------------------------------------
    # Demo 2: Temperature effect on attention
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("DEMO 2: Temperature Scaling Effect")
    print("-"*70)

    print("\nTemperature controls attention 'sharpness':")
    print("  - High temp (>1): Softer, more uniform attention")
    print("  - Low temp (<1):  Sharper, more focused attention")

    for temp in [0.5, 1.0, 2.0]:
        _, weights = scaled_dot_product_attention(Q, K, V, temperature=temp)
        entropy = -(weights * torch.log(weights + 1e-9)).sum(-1).mean()
        max_weight = weights.max(dim=-1)[0].mean()
        print(f"\n  Temperature = {temp}:")
        print(f"    Entropy: {entropy:.3f} (higher = more uniform)")
        print(f"    Max weight: {max_weight:.3f} (higher = more focused)")
        print(f"    First row weights: {weights[0, 0].numpy().round(3)}")

    # -------------------------------------------------------------------------
    # Demo 3: Causal masking
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("DEMO 3: Causal Masking (Autoregressive)")
    print("-"*70)

    mask = create_causal_mask(seq_len)
    print(f"\nCausal mask (0 = attend, -inf = block):")
    print(mask.numpy().round(1))

    _, weights_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
    print(f"\nAttention weights WITH causal mask:")
    print(weights_causal[0].numpy().round(3))
    print("\nNote: Upper triangle is 0 (can't attend to future)")

    # -------------------------------------------------------------------------
    # Demo 4: Multi-head attention
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("DEMO 4: Multi-Head Attention")
    print("-"*70)

    d_model = 64
    num_heads = 8
    seq_len = 16

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)

    # Self-attention: query = key = value
    x = torch.randn(batch_size, seq_len, d_model)
    output = mha(x, x, x)

    print(f"\nMulti-Head Attention Configuration:")
    print(f"  d_model (total): {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_k per head: {d_model // num_heads}")

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nParameter count:")
    print(f"  W_Q: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_K: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_V: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  W_O: {d_model} × {d_model} = {d_model * d_model:,}")
    print(f"  Biases: 4 × {d_model} = {4 * d_model}")
    print(f"  Total: {total_params:,}")

    # -------------------------------------------------------------------------
    # Demo 5: Attention pattern visualization
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("DEMO 5: What Different Heads Learn")
    print("-"*70)

    # Get attention weights from all heads
    weights = mha.get_attention_weights()  # (batch, num_heads, seq_q, seq_k)

    print(f"\nAttention weights shape: {weights.shape}")
    print(f"  - Batch: {weights.shape[0]}")
    print(f"  - Heads: {weights.shape[1]}")
    print(f"  - Query positions: {weights.shape[2]}")
    print(f"  - Key positions: {weights.shape[3]}")

    print("\nEntropy per head (measures attention spread):")
    for head in range(num_heads):
        head_weights = weights[0, head]
        entropy = -(head_weights * torch.log(head_weights + 1e-9)).sum(-1).mean()
        max_attn = head_weights.max(dim=-1)[0].mean()
        print(f"  Head {head}: entropy={entropy:.3f}, max_weight={max_attn:.3f}")

    print("\nInterpretation:")
    print("  - Low entropy + high max: Focused attention (specific relationships)")
    print("  - High entropy + low max: Distributed attention (broad context)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("  SUMMARY: ATTENTION MATHEMATICS → CODE")
    print("="*70)
    print("""
    EQUATION                           CODE
    ────────────────────────────────────────────────────────────────────
    scores = Q × K^T                   scores = torch.matmul(Q, K.transpose(-2,-1))

    scaled = scores / √d_k             scaled_scores = scores / math.sqrt(d_k)

    masked = scaled + mask             scaled_scores = scaled_scores + mask

    weights = softmax(masked)          attention_weights = F.softmax(scaled_scores, dim=-1)

    output = weights × V               output = torch.matmul(attention_weights, V)

    MultiHead = Concat(heads) × W_O    output = self.W_O(concat_heads)
    """)


if __name__ == "__main__":
    demonstrate_attention()
