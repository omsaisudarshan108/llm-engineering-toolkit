"""
Minimal LoRA Example - Core Concepts Only
==========================================

This stripped-down example shows the essential LoRA setup without the full training
infrastructure. Use this to understand the core mechanics.

WHY LoRA?
---------
Full fine-tuning of a 7B model:
  - Parameters: 7,000,000,000 (7 billion)
  - Memory for weights (fp32): ~28 GB
  - Memory for optimizer (Adam): ~56 GB (2 states per param)
  - Memory for gradients: ~28 GB
  - Total: ~112 GB VRAM (impossible on consumer hardware)

LoRA fine-tuning of a 7B model (r=16):
  - Trainable parameters: ~4,000,000 (0.06% of total)
  - Additional memory: <100 MB
  - Can train on 16GB GPU with 4-bit quantization

The math: For a weight matrix W ∈ R^(d×k), LoRA adds:
  - B ∈ R^(d×r) and A ∈ R^(r×k) where r << min(d,k)
  - Forward: y = Wx + BAx
  - Only B and A are trained; W stays frozen
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def setup_lora_model(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_4bit: bool = True,
):
    """
    Minimal setup for LoRA fine-tuning.

    Parameters:
    -----------
    model_name : str
        HuggingFace model identifier
    lora_r : int
        LoRA rank - controls adapter capacity (higher = more params)
    lora_alpha : int
        LoRA scaling factor (effective scale = alpha/r)
    use_4bit : bool
        Whether to use 4-bit quantization for memory efficiency
    """

    # 1. TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. QUANTIZATION CONFIG (optional but recommended for large models)
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",            # Normal Float 4
            bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16
            bnb_4bit_use_double_quant=True,        # Nested quantization
        )

    # 3. LOAD BASE MODEL (FROZEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Disable KV caching (incompatible with gradient checkpointing)
    model.config.use_cache = False

    # Prepare for k-bit training if quantized
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 4. CONFIGURE LoRA
    # Key decision: which modules to adapt?
    # - Attention (q, k, v, o projections): Most common, efficient
    # - MLP (gate, up, down projections): More capacity, more params
    lora_config = LoraConfig(
        r=lora_r,                          # Rank of decomposition
        lora_alpha=lora_alpha,             # Scaling factor
        lora_dropout=0.05,                 # Regularization
        target_modules=[
            "q_proj", "k_proj", "v_proj",  # Attention
            "o_proj",                       # Attention output
            # "gate_proj", "up_proj", "down_proj",  # MLP (uncomment for more capacity)
        ],
        bias="none",                       # Don't train biases
        task_type=TaskType.CAUSAL_LM,
    )

    # 5. APPLY LoRA TO MODEL
    model = get_peft_model(model, lora_config)

    # 6. VERIFY FROZEN VS TRAINABLE
    print("\n" + "="*50)
    print("Parameter Analysis")
    print("="*50)

    total, trainable = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            # Uncomment to see which params are trained:
            # print(f"  TRAINABLE: {name} ({param.numel():,})")

    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %:      {100*trainable/total:.4f}%")
    print(f"Reduction factor: {total/trainable:.0f}x fewer gradients")
    print("="*50 + "\n")

    return model, tokenizer


def minimal_training_step(model, tokenizer, text: str):
    """
    Single training step to demonstrate gradient flow.

    Note: In LoRA, gradients only flow to adapter weights (A, B matrices).
    The base model weights W receive zero gradient.
    """
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Forward pass with labels (for loss computation)
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],  # Causal LM: predict next token
    )

    print(f"Loss: {outputs.loss.item():.4f}")

    # Backward pass
    outputs.loss.backward()

    # Check gradients
    print("\nGradient status (sample):")
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.grad is not None:
            print(f"  {name}: grad_norm = {param.grad.norm().item():.6f}")
            break  # Just show one example

    # In real training: optimizer.step(), scheduler.step(), optimizer.zero_grad()
    model.zero_grad()


def inference_example(model, tokenizer, prompt: str):
    """Generate text with the (fine-tuned) model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# LoRA MATH VISUALIZATION
# =============================================================================

def visualize_lora_math():
    """
    Visualize the LoRA decomposition mathematically.

    For a weight matrix W with shape (output_dim, input_dim):
    - Traditional: W' = W + ΔW, where ΔW has same shape as W
    - LoRA:        W' = W + BA, where B is (output_dim, r) and A is (r, input_dim)

    Example with concrete numbers:
    """
    print("\n" + "="*60)
    print("LoRA MATHEMATICAL DECOMPOSITION")
    print("="*60)

    # Example dimensions (simplified)
    input_dim = 4096   # Model hidden dimension
    output_dim = 4096  # Same for attention projections
    rank = 16          # LoRA rank

    # Original weight matrix
    W_params = input_dim * output_dim
    print(f"\nOriginal weight matrix W:")
    print(f"  Shape: ({output_dim}, {input_dim})")
    print(f"  Parameters: {W_params:,}")

    # LoRA decomposition
    B_params = output_dim * rank
    A_params = rank * input_dim
    lora_params = B_params + A_params

    print(f"\nLoRA decomposition (rank={rank}):")
    print(f"  Matrix B: ({output_dim}, {rank}) = {B_params:,} params")
    print(f"  Matrix A: ({rank}, {input_dim}) = {A_params:,} params")
    print(f"  Total LoRA: {lora_params:,} params")

    # Compression ratio
    compression = W_params / lora_params
    print(f"\nCompression ratio: {compression:.1f}x")
    print(f"Memory reduction: {(1 - lora_params/W_params)*100:.2f}%")

    # For a full model (Mistral-7B has ~32 layers, each with 4 attention projections)
    num_layers = 32
    num_projections = 4  # q, k, v, o
    total_lora = lora_params * num_layers * num_projections

    print(f"\nFull model LoRA overhead ({num_layers} layers, {num_projections} projections each):")
    print(f"  Total LoRA params: {total_lora:,}")
    print(f"  Memory (fp16): ~{total_lora * 2 / 1e6:.1f} MB")
    print("="*60)


if __name__ == "__main__":
    # Show the math first
    visualize_lora_math()

    # Uncomment below to actually load and test the model
    # (requires GPU with sufficient VRAM)
    """
    print("\nLoading model with LoRA...")
    model, tokenizer = setup_lora_model(
        model_name="mistralai/Mistral-7B-v0.1",
        lora_r=16,
        lora_alpha=32,
        use_4bit=True,
    )

    # Test training step
    print("\nRunning minimal training step...")
    minimal_training_step(
        model, tokenizer,
        "The Federal Reserve announced interest rates will remain unchanged."
    )

    # Test inference
    print("\nRunning inference...")
    response = inference_example(
        model, tokenizer,
        "Analyze the financial implications: Company XYZ reported Q3 earnings of $2.50 per share, beating estimates by 15%."
    )
    print(f"Generated: {response}")
    """
