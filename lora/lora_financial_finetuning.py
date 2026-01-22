"""
LoRA Fine-Tuning for Financial Domain Tasks
============================================

This script demonstrates how to fine-tune a Llama/Mistral-class model using LoRA
(Low-Rank Adaptation) for financial domain tasks.

WHY LoRA OVER FULL FINE-TUNING?
-------------------------------
1. MEMORY EFFICIENCY: Full fine-tuning a 7B parameter model requires ~28GB+ VRAM
   just for model weights (fp32), plus optimizer states (2x for Adam), gradients,
   and activations. LoRA reduces trainable parameters by 99%+, making it feasible
   on consumer GPUs.

2. CATASTROPHIC FORGETTING PREVENTION: By freezing the base model and only training
   low-rank adapters, we preserve the model's general knowledge while adding
   domain-specific capabilities.

3. MODULARITY: LoRA adapters are small (~10-100MB) and can be swapped, combined,
   or versioned independently of the base model. You can have multiple adapters
   for different tasks sharing one base model.

4. TRAINING SPEED: Fewer parameters to update means faster backward passes and
   optimizer steps, reducing training time significantly.

5. MATHEMATICAL INTUITION: LoRA decomposes weight updates as W' = W + BA, where
   B ∈ R^(d×r) and A ∈ R^(r×k), with rank r << min(d,k). This exploits the
   hypothesis that task-specific adaptations have low "intrinsic rank".

Author: AI Engineer Example
Task: Financial sentiment analysis and instruction following
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LoRAConfig:
    """
    LoRA hyperparameters - these are the key knobs to tune.

    r (rank): The rank of the low-rank decomposition. Higher rank = more capacity
              but more parameters. Typical values: 8, 16, 32, 64.

    lora_alpha: Scaling factor. The actual scaling is alpha/r. Higher alpha
                means stronger adapter influence. Often set equal to r or 2*r.

    lora_dropout: Dropout probability for LoRA layers. Helps prevent overfitting
                  on small datasets. Typical: 0.05-0.1.

    target_modules: Which layers to apply LoRA to. For transformer models:
                    - q_proj, v_proj: Attention queries and values (most common)
                    - k_proj: Attention keys
                    - o_proj: Attention output projection
                    - gate_proj, up_proj, down_proj: MLP layers (for deeper adaptation)
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Target modules for Llama/Mistral architecture
    # We target attention projections (most parameter-efficient) and optionally MLP
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",   # Query projection - captures what to attend to
        "v_proj",   # Value projection - captures what information to extract
        "k_proj",   # Key projection - helps with attention patterns
        "o_proj",   # Output projection - combines attention results
        # Uncomment below for deeper adaptation (more parameters, more capacity)
        # "gate_proj",  # MLP gating
        # "up_proj",    # MLP up projection
        # "down_proj",  # MLP down projection
    ])
    bias: str = "none"  # Can be "none", "all", or "lora_only"
    task_type: TaskType = TaskType.CAUSAL_LM


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Model
    model_name: str = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Llama-2-7b-hf"

    # Quantization for memory efficiency
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"  # Normal Float 4-bit
    use_nested_quant: bool = True  # Double quantization

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 512

    # Paths
    output_dir: str = "./lora_financial_model"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# FINANCIAL DOMAIN DATASET
# =============================================================================

# Sample financial domain data - in practice, use a larger dataset like:
# - Financial PhraseBank (sentiment)
# - FiQA (financial QA)
# - Bloomberg financial news
# - SEC filings

FINANCIAL_TRAINING_DATA = [
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "The company reported a 25% increase in quarterly revenue, exceeding analyst expectations.",
        "output": "POSITIVE. The statement indicates strong financial performance with revenue growth surpassing market predictions, which is typically bullish for stock valuation."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Due to supply chain disruptions, we expect margins to compress by 200 basis points next quarter.",
        "output": "NEGATIVE. Margin compression indicates reduced profitability. Supply chain issues suggest operational challenges that may persist, creating downward pressure on earnings."
    },
    {
        "instruction": "Explain this financial metric in simple terms.",
        "input": "The company's P/E ratio expanded from 15x to 22x over the fiscal year.",
        "output": "The Price-to-Earnings ratio increased significantly. This means investors are now willing to pay $22 for every $1 of earnings, up from $15. This expansion typically indicates growing market optimism about future growth, though it could also signal overvaluation."
    },
    {
        "instruction": "Assess the risk level of this investment scenario.",
        "input": "A biotech startup with no revenue is seeking Series B funding at a $500M valuation based on Phase 2 trial results.",
        "output": "HIGH RISK. Pre-revenue biotech valuations are speculative. Phase 2 success doesn't guarantee Phase 3 approval (historically ~50% fail). The high valuation creates significant downside risk if trials disappoint. Suitable only for risk-tolerant investors with diversified portfolios."
    },
    {
        "instruction": "Interpret this earnings call statement.",
        "input": "We are cautiously optimistic about the second half, though macro headwinds persist.",
        "output": "NEUTRAL TO SLIGHTLY NEGATIVE. 'Cautiously optimistic' is hedged language suggesting uncertainty. 'Macro headwinds' acknowledges external challenges (inflation, rates, demand). Management is tempering expectations while avoiding alarm. Guidance may be conservative."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "The Federal Reserve signaled potential rate cuts in the upcoming quarters amid cooling inflation.",
        "output": "POSITIVE for equities. Rate cuts reduce borrowing costs, making stocks more attractive relative to bonds. Cooling inflation suggests a 'soft landing' scenario. Growth stocks and rate-sensitive sectors (real estate, utilities) typically benefit most."
    },
    {
        "instruction": "Evaluate this merger announcement.",
        "input": "TechCorp announces acquisition of DataStart for $2.5B, representing a 40% premium to market price.",
        "output": "MIXED. For DataStart shareholders: POSITIVE (40% premium is substantial). For TechCorp: CAUTIOUS - large premiums require significant synergies to justify. Key questions: Is it cash or stock? What's the integration risk? Does it create regulatory concerns? Historically, most M&A destroys acquirer value."
    },
    {
        "instruction": "Explain the implications of this debt covenant.",
        "input": "The company must maintain a debt-to-EBITDA ratio below 4.0x or face accelerated repayment clauses.",
        "output": "This covenant limits leverage and protects lenders. If EBITDA falls or debt rises such that the ratio exceeds 4.0x, lenders can demand immediate repayment. This creates refinancing risk during downturns. Companies approaching this threshold often cut costs or sell assets to maintain compliance."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Short interest in the stock has increased to 25% of float, with days-to-cover at 8.5.",
        "output": "BEARISH SIGNAL with SQUEEZE POTENTIAL. High short interest (25%) indicates significant negative sentiment. However, 8.5 days-to-cover is elevated, meaning shorts would take over a week to exit. This creates short squeeze risk if positive catalysts emerge, potentially causing violent upward moves."
    },
    {
        "instruction": "Assess this dividend announcement.",
        "input": "The board approved a 15% dividend increase, marking the 25th consecutive year of dividend growth.",
        "output": "POSITIVE. Consistent dividend growth for 25 years qualifies as 'Dividend Aristocrat' status. The 15% increase signals board confidence in cash flow sustainability. This attracts income-focused investors and typically provides stock price support. Strong indicator of financial health and shareholder-friendly management."
    },
    # Add more examples for better training...
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Operating cash flow turned negative this quarter due to working capital build.",
        "output": "CAUTIOUS. Negative operating cash flow is concerning but context matters. Working capital build (inventory, receivables) could indicate growth preparation or could signal demand softness. One quarter isn't a trend, but warrants monitoring. Check if it's seasonal or structural."
    },
    {
        "instruction": "Explain this trading pattern.",
        "input": "The stock formed a death cross with the 50-day MA crossing below the 200-day MA.",
        "output": "BEARISH TECHNICAL SIGNAL. A 'death cross' occurs when short-term momentum (50-day moving average) falls below long-term trend (200-day). Historically associated with further downside, though it's a lagging indicator. Often triggers algorithmic selling. Not predictive alone but confirms existing downtrend."
    },
]


class FinancialInstructionDataset(Dataset):
    """
    Dataset for financial instruction tuning.

    Format follows Alpaca-style: instruction + input + output
    Converted to prompt format for causal language modeling.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _format_prompt(self, example: Dict[str, str]) -> str:
        """
        Format example into instruction-following prompt.

        This format is crucial for instruction tuning - the model learns
        to follow the pattern and generate appropriate responses.
        """
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        prompt = self._format_prompt(example)

        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # For causal LM, labels = input_ids (model predicts next token)
        # We mask padding tokens in labels with -100 (ignored in loss)
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# MODEL SETUP WITH LoRA
# =============================================================================

def create_quantization_config(config: TrainingConfig) -> Optional[BitsAndBytesConfig]:
    """
    Create 4-bit quantization config for memory-efficient training.

    Quantization reduces model memory footprint by ~4x, enabling
    training on consumer GPUs. Combined with LoRA, this allows
    fine-tuning 7B+ models on 16GB VRAM.
    """
    if not config.use_4bit:
        return None

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )


def load_model_and_tokenizer(
    training_config: TrainingConfig,
    lora_config: LoRAConfig,
) -> tuple:
    """
    Load base model with quantization and apply LoRA adapters.

    This function demonstrates the key steps:
    1. Load quantized base model (frozen)
    2. Prepare for k-bit training
    3. Apply LoRA adapters (trainable)
    """
    print(f"Loading model: {training_config.model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        training_config.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # For causal LM, pad on right

    # Quantization config
    quant_config = create_quantization_config(training_config)

    # Load base model
    # NOTE: Base model weights are FROZEN - we don't compute gradients for them
    model = AutoModelForCausalLM.from_pretrained(
        training_config.model_name,
        quantization_config=quant_config,
        device_map="auto",  # Automatically distribute across available GPUs
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Disable caching for training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    model.config.pretraining_tp = 1  # Tensor parallelism degree

    # Prepare model for k-bit training
    # This handles:
    # - Casting layer norms to float32 for stability
    # - Enabling gradient checkpointing for memory efficiency
    # - Setting up proper gradient computation for quantized weights
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    # Apply LoRA to model
    # This wraps target modules with LoRA layers:
    # Original: y = Wx
    # With LoRA: y = Wx + BAx (where B, A are low-rank matrices)
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    print_trainable_parameters(model)

    return model, tokenizer


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters.

    This demonstrates LoRA's efficiency - we typically train <1% of parameters.
    """
    trainable_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_pct = 100 * trainable_params / total_params

    print(f"\n{'='*60}")
    print(f"PARAMETER EFFICIENCY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %:          {trainable_pct:.4f}%")
    print(f"Memory saved:         ~{(1 - trainable_pct/100)*100:.2f}% fewer gradients")
    print(f"{'='*60}\n")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def evaluate(
    model,
    eval_dataloader: DataLoader,
    device: str,
) -> float:
    """
    Evaluate model on validation set.

    Returns average loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def train(
    model,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    training_config: TrainingConfig,
) -> Dict[str, List[float]]:
    """
    Main training loop with gradient accumulation and evaluation.

    Key aspects:
    1. Only LoRA parameters receive gradients (base model frozen)
    2. Gradient accumulation for effective larger batch sizes
    3. Learning rate warmup for stable training
    4. Regular evaluation to track progress
    """
    device = training_config.device

    # Optimizer - only operates on trainable (LoRA) parameters
    # AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # Learning rate scheduler with warmup
    total_steps = (
        len(train_dataloader)
        * training_config.num_epochs
        // training_config.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * training_config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Track metrics
    history = {
        "train_loss": [],
        "eval_loss": [],
        "learning_rate": [],
    }

    # Evaluate before training (baseline)
    print("\nEvaluating baseline (before training)...")
    baseline_loss = evaluate(model, eval_dataloader, device)
    print(f"Baseline eval loss: {baseline_loss:.4f}")
    history["eval_loss"].append(baseline_loss)

    # Training loop
    global_step = 0

    for epoch in range(training_config.num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{training_config.num_epochs}",
            leave=True,
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / training_config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            epoch_loss += loss.item() * training_config.gradient_accumulation_steps

            # Update weights every gradient_accumulation_steps
            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * training_config.gradient_accumulation_steps:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

        # Epoch statistics
        avg_train_loss = epoch_loss / len(train_dataloader)
        history["train_loss"].append(avg_train_loss)
        history["learning_rate"].append(scheduler.get_last_lr()[0])

        # Evaluate after each epoch
        eval_loss = evaluate(model, eval_dataloader, device)
        history["eval_loss"].append(eval_loss)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Loss:  {eval_loss:.4f}")
        print(f"  LR:         {scheduler.get_last_lr()[0]:.2e}")

    return history


# =============================================================================
# INFERENCE AND COMPARISON
# =============================================================================

def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a response using the fine-tuned model."""
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    response = response.split("### Response:")[-1].strip()

    return response


def compare_before_after(history: Dict[str, List[float]]) -> None:
    """Print before/after comparison of model performance."""
    print("\n" + "="*60)
    print("TRAINING RESULTS: BEFORE vs AFTER")
    print("="*60)

    before_loss = history["eval_loss"][0]
    after_loss = history["eval_loss"][-1]
    improvement = ((before_loss - after_loss) / before_loss) * 100

    print(f"\nEvaluation Loss:")
    print(f"  Before training: {before_loss:.4f}")
    print(f"  After training:  {after_loss:.4f}")
    print(f"  Improvement:     {improvement:.2f}%")

    print(f"\nTraining Loss Progression:")
    for i, loss in enumerate(history["train_loss"]):
        print(f"  Epoch {i+1}: {loss:.4f}")

    # Perplexity (exp of loss) - more interpretable metric
    print(f"\nPerplexity:")
    print(f"  Before: {np.exp(before_loss):.2f}")
    print(f"  After:  {np.exp(after_loss):.2f}")

    print("="*60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("LoRA FINE-TUNING FOR FINANCIAL DOMAIN")
    print("="*60)

    # Initialize configurations
    training_config = TrainingConfig()
    lora_config = LoRAConfig()

    print(f"\nLoRA Configuration:")
    print(f"  Rank (r):        {lora_config.r}")
    print(f"  Alpha:           {lora_config.lora_alpha}")
    print(f"  Dropout:         {lora_config.lora_dropout}")
    print(f"  Target modules:  {lora_config.target_modules}")

    print(f"\nTraining Configuration:")
    print(f"  Model:           {training_config.model_name}")
    print(f"  Epochs:          {training_config.num_epochs}")
    print(f"  Batch size:      {training_config.batch_size}")
    print(f"  Learning rate:   {training_config.learning_rate}")
    print(f"  4-bit quant:     {training_config.use_4bit}")

    # Check for GPU
    if training_config.device == "cuda":
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\nWARNING: No GPU detected. Training will be slow.")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(training_config, lora_config)

    # Prepare datasets
    # In practice, split into proper train/val/test sets
    train_data = FINANCIAL_TRAINING_DATA[:10]
    eval_data = FINANCIAL_TRAINING_DATA[10:]

    train_dataset = FinancialInstructionDataset(
        train_data, tokenizer, training_config.max_seq_length
    )
    eval_dataset = FinancialInstructionDataset(
        eval_data, tokenizer, training_config.max_seq_length
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,  # Set >0 for faster data loading
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_dataset)} examples")
    print(f"  Evaluation: {len(eval_dataset)} examples")

    # Train the model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    history = train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        training_config=training_config,
    )

    # Show results comparison
    compare_before_after(history)

    # Save the LoRA adapter
    print(f"\nSaving LoRA adapter to: {training_config.output_dir}")
    model.save_pretrained(training_config.output_dir)
    tokenizer.save_pretrained(training_config.output_dir)

    # Save training history
    with open(f"{training_config.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Demonstrate inference
    print("\n" + "="*60)
    print("INFERENCE DEMONSTRATION")
    print("="*60)

    test_examples = [
        {
            "instruction": "Analyze the sentiment of this financial statement.",
            "input": "The company announced a strategic restructuring plan expected to result in $500M in annual cost savings but will involve 10,000 layoffs.",
        },
        {
            "instruction": "Explain this financial concept.",
            "input": "What does it mean when a company's free cash flow exceeds its net income?",
        },
    ]

    print("\nGenerating responses for test examples:")
    for i, example in enumerate(test_examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Instruction: {example['instruction']}")
        print(f"Input: {example['input']}")

        response = generate_response(
            model, tokenizer,
            example["instruction"],
            example["input"],
        )
        print(f"Response: {response}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nThe LoRA adapter has been saved to: {training_config.output_dir}")
    print("To load the fine-tuned model later:")
    print("""
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = PeftModel.from_pretrained(base_model, "./lora_financial_model")
    """)

    return model, tokenizer, history


if __name__ == "__main__":
    # For demonstration without actual model loading (CPU/small GPU)
    # Set environment variable to use a smaller model
    if os.getenv("DEMO_MODE"):
        print("Running in demo mode with smaller model...")
        # Could substitute with a smaller model for testing

    model, tokenizer, history = main()
