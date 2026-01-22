"""
LoRA Fine-Tuning Demo with Step-by-Step Explanations
=====================================================

This demo uses a SMALL model (GPT-2) so it runs on any machine.
The concepts are identical to fine-tuning Llama/Mistral - just smaller scale.

Run with: python lora_demo_explained.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import time

# For reproducibility
torch.manual_seed(42)


def print_step(step_num: int, title: str, explanation: str):
    """Print a formatted step explanation."""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {title}")
    print("="*70)
    print(f"\n📖 WHAT THIS MEANS:\n{explanation}\n")


def print_substep(title: str):
    """Print a substep header."""
    print(f"\n  → {title}")


# =============================================================================
# STEP 1: THE PROBLEM WE'RE SOLVING
# =============================================================================

print_step(1, "THE PROBLEM WE'RE SOLVING", """
Imagine you have a very smart assistant (like ChatGPT) that knows a lot about
everything, but you need it to be an EXPERT in YOUR specific field - like
financial analysis.

THE CHALLENGE:
- These AI models have BILLIONS of parameters (like 7,000,000,000 numbers)
- Training all those numbers requires MASSIVE computers ($10,000+ GPUs)
- If you train too hard, the model "forgets" its general knowledge

THE SOLUTION - LoRA (Low-Rank Adaptation):
- Instead of changing ALL 7 billion numbers, we add TINY "adapter" layers
- We only train these adapters (~4 million numbers = 0.06% of the model)
- The original model stays FROZEN (untouched)
- Result: Expert knowledge WITHOUT forgetting, trainable on normal computers!

Think of it like this:
- Full fine-tuning = Rebuilding an entire car to make it faster
- LoRA = Adding a turbocharger - small addition, big impact
""")

input("Press Enter to continue to Step 2...")


# =============================================================================
# STEP 2: PREPARE THE TRAINING DATA
# =============================================================================

print_step(2, "PREPARE THE TRAINING DATA", """
We need examples that teach the model HOW to respond in our domain.

Each example has:
- INSTRUCTION: What we want the model to do
- INPUT: The specific content to analyze
- OUTPUT: The correct/desired response

The model learns the PATTERN: "When given financial text, respond like this."

In real applications, you'd have hundreds or thousands of examples.
We're using 12 for this demo.
""")

# Our financial domain training data
FINANCIAL_DATA = [
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "The company reported a 25% increase in quarterly revenue, exceeding analyst expectations.",
        "output": "POSITIVE. Strong revenue growth beating expectations signals robust business performance and typically drives stock appreciation."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Due to supply chain disruptions, margins will compress by 200 basis points.",
        "output": "NEGATIVE. Margin compression indicates reduced profitability. Supply chain issues may persist, creating earnings pressure."
    },
    {
        "instruction": "Explain this financial concept simply.",
        "input": "The P/E ratio expanded from 15x to 22x.",
        "output": "Investors now pay $22 for every $1 of earnings (up from $15). This suggests growing optimism about future growth, but could signal overvaluation."
    },
    {
        "instruction": "Assess the risk level.",
        "input": "A pre-revenue biotech seeks $500M valuation based on Phase 2 trials.",
        "output": "HIGH RISK. No revenue means pure speculation. Phase 2 success doesn't guarantee approval (~50% fail Phase 3). Only for risk-tolerant investors."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "The Federal Reserve signaled potential rate cuts amid cooling inflation.",
        "output": "POSITIVE for equities. Rate cuts reduce borrowing costs, making stocks more attractive. Growth stocks benefit most."
    },
    {
        "instruction": "Interpret this earnings call language.",
        "input": "We remain cautiously optimistic despite macro headwinds.",
        "output": "NEUTRAL TO NEGATIVE. 'Cautiously optimistic' is hedged language. 'Headwinds' acknowledges challenges. Management is tempering expectations."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Short interest reached 25% of float with 8 days to cover.",
        "output": "BEARISH with SQUEEZE RISK. High short interest = negative sentiment. But 8 days to cover creates squeeze potential if positive news hits."
    },
    {
        "instruction": "Evaluate this dividend announcement.",
        "input": "Board approved 15% dividend increase, marking 25 consecutive years of growth.",
        "output": "POSITIVE. 25 years of growth = Dividend Aristocrat status. 15% increase signals confidence. Attracts income investors."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Operating cash flow turned negative due to working capital build.",
        "output": "CAUTIOUS. Negative cash flow is concerning but context matters. Working capital build could indicate growth prep or demand issues. Monitor closely."
    },
    {
        "instruction": "Explain this technical pattern.",
        "input": "The stock formed a death cross - 50-day MA crossed below 200-day MA.",
        "output": "BEARISH SIGNAL. Short-term momentum falling below long-term trend. Often triggers algorithmic selling. Confirms existing downtrend."
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Company announced $2B share buyback program.",
        "output": "POSITIVE. Buybacks reduce share count, boosting EPS. Signals management believes stock is undervalued. Returns cash to shareholders."
    },
    {
        "instruction": "Assess this debt situation.",
        "input": "Debt-to-EBITDA covenant requires ratio below 4.0x or triggers acceleration.",
        "output": "RISK FACTOR. If EBITDA falls or debt rises past 4.0x, lenders can demand immediate repayment. Creates refinancing risk in downturns."
    },
]

print(f"📊 Created {len(FINANCIAL_DATA)} training examples")
print("\n📝 Example training sample:")
print(f"   Instruction: {FINANCIAL_DATA[0]['instruction']}")
print(f"   Input: {FINANCIAL_DATA[0]['input'][:50]}...")
print(f"   Output: {FINANCIAL_DATA[0]['output'][:50]}...")

input("\nPress Enter to continue to Step 3...")


# =============================================================================
# STEP 3: CREATE THE DATASET CLASS
# =============================================================================

print_step(3, "CREATE THE DATASET CLASS", """
Neural networks don't understand text - they only understand NUMBERS.

We need to:
1. Convert text → numbers (called "tokenization")
   - "The stock rose" → [464, 4283, 6348]

2. Format examples consistently so the model learns the pattern:
   ### Instruction: [what to do]
   ### Input: [the content]
   ### Response: [correct answer]

3. Create "labels" - the numbers the model should PREDICT
   - The model reads tokens 1-10 and tries to predict token 11
   - We compare its prediction to the actual token 11
   - The difference is the "loss" (error) we minimize

Think of it like flashcards: show the question, check if the answer matches.
""")


class FinancialDataset(Dataset):
    """Converts our text examples into number sequences the model can learn from."""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def format_prompt(self, example):
        """Create the instruction format the model will learn."""
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

    def __getitem__(self, idx):
        # Get the text example
        example = self.data[idx]
        prompt = self.format_prompt(example)

        # Convert text to numbers (tokens)
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        # Labels = input_ids (model predicts next token)
        # -100 means "ignore this position in loss calculation" (for padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


print("✅ Dataset class created!")
print("\n🔍 What happens inside:")
print('   Text: "The stock rose" ')
print("   ↓ Tokenization")
print("   Numbers: [464, 4283, 6348]")
print("   ↓ Padding (to fixed length)")
print("   Padded: [464, 4283, 6348, 0, 0, 0, ...]")

input("\nPress Enter to continue to Step 4...")


# =============================================================================
# STEP 4: LOAD THE BASE MODEL
# =============================================================================

print_step(4, "LOAD THE BASE MODEL (THE 'FROZEN' BRAIN)", """
Now we load a pre-trained language model. This model already:
- Understands grammar and language structure
- Has general world knowledge
- Can generate coherent text

We're using GPT-2 (small) for this demo - same architecture concepts as Llama/Mistral,
just 100x smaller so it runs on any computer.

IMPORTANT: We will FREEZE this model - none of its weights will change!
It's like having a smart employee who already knows a lot - we don't want
to mess with their existing knowledge, just teach them new specialized skills.
""")

print_substep("Loading tokenizer (the text-to-numbers converter)...")

# Using GPT-2 for demo (same concepts apply to Llama/Mistral)
MODEL_NAME = "gpt2"  # 124M parameters - runs on CPU

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 needs this
print(f"   ✅ Tokenizer loaded! Vocabulary size: {tokenizer.vocab_size:,} words/tokens")

print_substep("Loading the base model...")
start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
)
model.config.use_cache = False  # Needed for training

load_time = time.time() - start_time
print(f"   ✅ Model loaded in {load_time:.1f} seconds")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"   📊 Total parameters: {total_params:,}")
print(f"   📊 Model size: ~{total_params * 4 / 1e6:.0f} MB (float32)")

input("\nPress Enter to continue to Step 5 (THE KEY STEP - LoRA!)...")


# =============================================================================
# STEP 5: APPLY LoRA (THE MAGIC!)
# =============================================================================

print_step(5, "APPLY LoRA ADAPTERS (THE MAGIC!)", """
This is where the magic happens! Instead of training all 124 million parameters,
we ADD tiny "adapter" layers and ONLY train those.

HOW LoRA WORKS (simplified):
- Original layer: output = input × Weight_Matrix
- With LoRA:      output = input × Weight_Matrix + input × A × B

Where:
- Weight_Matrix is HUGE and FROZEN (not trained)
- A and B are TINY matrices that we train
- A × B approximates the changes we need

EXAMPLE with real numbers:
- Weight_Matrix: 768 × 768 = 589,824 parameters (FROZEN)
- A matrix: 768 × 16 = 12,288 parameters (TRAINED)
- B matrix: 16 × 768 = 12,288 parameters (TRAINED)
- Total LoRA: 24,576 parameters = 4% of original!

We're trading a tiny bit of capacity for MASSIVE memory/speed savings.

WHICH LAYERS TO ADAPT?
- We target the "attention" layers (c_attn, c_proj in GPT-2)
- These control what the model "pays attention to" in the input
- Adapting attention = teaching the model what's important in financial text
""")

print_substep("Configuring LoRA parameters...")

# LoRA Configuration - these are the key "knobs"
lora_config = LoraConfig(
    r=16,                    # Rank: size of A and B matrices. Higher = more capacity
    lora_alpha=32,           # Scaling factor. Effective scale = alpha/r = 2
    lora_dropout=0.05,       # Dropout for regularization (prevents overfitting)
    target_modules=[         # Which layers to add adapters to
        "c_attn",            # Attention layer (q, k, v projections combined in GPT-2)
        "c_proj",            # Output projection
    ],
    bias="none",             # Don't train biases
    task_type=TaskType.CAUSAL_LM,
)

print(f"""   LoRA Configuration:
   • Rank (r): {lora_config.r} - Controls adapter capacity
   • Alpha: {lora_config.lora_alpha} - Scaling factor
   • Dropout: {lora_config.lora_dropout} - Prevents overfitting
   • Target modules: {lora_config.target_modules}
""")

print_substep("Applying LoRA to the model...")

# This is where we wrap the model with LoRA adapters
model = get_peft_model(model, lora_config)

# Now let's see the parameter breakdown
print_substep("Analyzing trainable vs frozen parameters...")

trainable_params = 0
frozen_params = 0
lora_layers = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params += param.numel()
        if "lora" in name.lower():
            lora_layers.append(name)
    else:
        frozen_params += param.numel()

print(f"""
   ╔══════════════════════════════════════════════════════════════╗
   ║                    PARAMETER BREAKDOWN                       ║
   ╠══════════════════════════════════════════════════════════════╣
   ║  🧊 FROZEN (base model):     {frozen_params:>12,} parameters       ║
   ║  🔥 TRAINABLE (LoRA only):   {trainable_params:>12,} parameters       ║
   ║  📊 Trainable percentage:    {100*trainable_params/(frozen_params+trainable_params):>11.4f}%           ║
   ║  💾 Memory saved:            ~{100*frozen_params/(frozen_params+trainable_params):>10.1f}% less gradients  ║
   ╚══════════════════════════════════════════════════════════════╝
""")

print(f"   🔧 LoRA adapter layers created: {len(lora_layers)}")
print(f"   Example LoRA layer: {lora_layers[0] if lora_layers else 'None'}")

input("\nPress Enter to continue to Step 6...")


# =============================================================================
# STEP 6: PREPARE DATA FOR TRAINING
# =============================================================================

print_step(6, "PREPARE DATA FOR TRAINING", """
Now we:
1. Split data into TRAINING (what model learns from) and VALIDATION (how we test it)
2. Create "DataLoaders" that feed batches of examples to the model

WHY BATCHES?
- Processing one example at a time is slow
- Processing all at once may not fit in memory
- Batches (e.g., 4 examples at a time) balance speed and memory

WHY VALIDATION?
- We need to test on data the model HASN'T seen during training
- If training loss goes down but validation loss goes up = OVERFITTING
- Overfitting = model memorized examples instead of learning patterns
""")

# Split data: 10 for training, 2 for validation
train_data = FINANCIAL_DATA[:10]
val_data = FINANCIAL_DATA[10:]

print(f"   📚 Training examples: {len(train_data)}")
print(f"   📝 Validation examples: {len(val_data)}")

# Create datasets
train_dataset = FinancialDataset(train_data, tokenizer, max_length=256)
val_dataset = FinancialDataset(val_data, tokenizer, max_length=256)

# Create data loaders
BATCH_SIZE = 2

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n   🔢 Batch size: {BATCH_SIZE}")
print(f"   🔄 Training batches per epoch: {len(train_loader)}")
print(f"   ✅ Data loaders ready!")

# Show what a batch looks like
sample_batch = next(iter(train_loader))
print(f"\n   📦 Sample batch shape: {sample_batch['input_ids'].shape}")
print(f"      → {BATCH_SIZE} examples, each with 256 tokens")

input("\nPress Enter to continue to Step 7...")


# =============================================================================
# STEP 7: SET UP THE OPTIMIZER
# =============================================================================

print_step(7, "SET UP THE OPTIMIZER (THE LEARNING MECHANISM)", """
The optimizer is what actually CHANGES the weights to reduce errors.

HOW LEARNING WORKS:
1. Model makes a prediction
2. We calculate the ERROR (loss) between prediction and correct answer
3. We calculate which direction to nudge each weight to reduce error (gradient)
4. Optimizer nudges the weights by a small amount (learning rate)
5. Repeat thousands of times until error is low!

KEY SETTINGS:
- Learning Rate: How big each nudge is
  - Too high: Model overshoots, never converges
  - Too low: Training takes forever
  - Sweet spot for LoRA: ~2e-4 (0.0002)

- Weight Decay: Slight penalty for large weights
  - Prevents any single weight from dominating
  - Acts as regularization

- Warmup: Start with tiny learning rate, gradually increase
  - Prevents early training from making wild jumps
  - Stabilizes training
""")

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# Only optimize LoRA parameters (the trainable ones)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

# Learning rate scheduler with warmup
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"""   ⚙️ Optimizer Settings:
   • Algorithm: AdamW (Adam with proper weight decay)
   • Learning rate: {LEARNING_RATE}
   • Weight decay: 0.01
   • Total training steps: {total_steps}
   • Warmup steps: {warmup_steps}

   📈 Learning Rate Schedule:
   • Steps 0-{warmup_steps}: Gradually increase from 0 to {LEARNING_RATE}
   • Steps {warmup_steps}-{total_steps}: Linearly decrease to 0
""")

input("\nPress Enter to continue to Step 8 (TRAINING!)...")


# =============================================================================
# STEP 8: THE TRAINING LOOP
# =============================================================================

print_step(8, "TRAINING THE MODEL", """
This is where learning happens! For each batch of examples:

1. FORWARD PASS: Feed data through the model, get predictions
2. CALCULATE LOSS: Compare predictions to correct answers
3. BACKWARD PASS: Calculate gradients (which way to nudge weights)
4. UPDATE WEIGHTS: Optimizer nudges the LoRA weights slightly

The "loss" number tells us how wrong the model is:
- Higher loss = more wrong
- Lower loss = better predictions

We want to see loss DECREASE over time. If it stops decreasing or goes up,
something might be wrong (overfitting, learning rate too high, etc.)
""")


def evaluate(model, dataloader, device):
    """Calculate average loss on a dataset (without training)."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)


# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   🖥️  Training on: {device.upper()}")
model = model.to(device)

# Evaluate BEFORE training (baseline)
print("\n   📊 Evaluating baseline (before any training)...")
baseline_loss = evaluate(model, val_loader, device)
print(f"   📉 Baseline validation loss: {baseline_loss:.4f}")

# Training history
history = {
    "train_loss": [],
    "val_loss": [baseline_loss],
    "learning_rates": [],
}

print(f"\n   🚀 Starting training for {NUM_EPOCHS} epochs...")
print("   " + "-"*60)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0

    progress = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)

    for batch_idx, batch in enumerate(progress):
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass - get model predictions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        epoch_loss += loss.item()

        # Backward pass - calculate gradients
        loss.backward()

        # Gradient clipping - prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update progress bar
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # End of epoch - evaluate on validation set
    avg_train_loss = epoch_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, device)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(val_loss)
    history["learning_rates"].append(scheduler.get_last_lr()[0])

    print(f"   📊 Epoch {epoch+1} Results:")
    print(f"      Training Loss:   {avg_train_loss:.4f}")
    print(f"      Validation Loss: {val_loss:.4f}")
    print()

print("   " + "-"*60)
print("   ✅ Training complete!")

input("\nPress Enter to see the RESULTS...")


# =============================================================================
# STEP 9: RESULTS AND COMPARISON
# =============================================================================

print_step(9, "RESULTS: BEFORE vs AFTER", """
Let's see how much the model improved!

LOSS measures how "wrong" the model's predictions are:
- Lower loss = better predictions
- We compare the BASELINE (before training) to FINAL (after training)

PERPLEXITY is another way to measure quality:
- It's roughly "how many words the model is confused between"
- Perplexity of 100 = model is choosing between ~100 likely next words
- Perplexity of 10 = model is much more confident (better!)
- Perplexity = exp(loss)
""")

import math

before_loss = history["val_loss"][0]
after_loss = history["val_loss"][-1]
improvement = ((before_loss - after_loss) / before_loss) * 100

print(f"""
   ╔══════════════════════════════════════════════════════════════╗
   ║                    TRAINING RESULTS                          ║
   ╠══════════════════════════════════════════════════════════════╣
   ║                                                              ║
   ║  VALIDATION LOSS:                                            ║
   ║    Before training: {before_loss:>8.4f}                               ║
   ║    After training:  {after_loss:>8.4f}                               ║
   ║    Improvement:     {improvement:>8.2f}%                              ║
   ║                                                              ║
   ║  PERPLEXITY (lower = better):                                ║
   ║    Before: {math.exp(before_loss):>8.2f}                                         ║
   ║    After:  {math.exp(after_loss):>8.2f}                                         ║
   ║                                                              ║
   ╚══════════════════════════════════════════════════════════════╝
""")

print("   📈 Training Loss Progression:")
for i, loss in enumerate(history["train_loss"]):
    bar_len = int(50 * (1 - loss/history["train_loss"][0])) if i > 0 else 0
    bar = "█" * bar_len + "░" * (50 - bar_len)
    print(f"      Epoch {i+1}: {loss:.4f} [{bar}]")

input("\nPress Enter to see the model in action...")


# =============================================================================
# STEP 10: SEE THE MODEL IN ACTION
# =============================================================================

print_step(10, "MODEL IN ACTION", """
Now let's test our fine-tuned model on NEW examples it hasn't seen!

We'll give it financial statements and see if it responds appropriately.
Remember: the model learned the PATTERN of how to analyze financial text,
not just memorized our specific examples.
""")


def generate_response(model, tokenizer, instruction, input_text, max_tokens=100):
    """Generate a response from the fine-tuned model."""
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    if "### Response:" in full_response:
        response = full_response.split("### Response:")[-1].strip()
    else:
        response = full_response

    # Truncate at a reasonable point
    response = response.split("\n\n")[0].split("###")[0].strip()

    return response


# Test examples the model HASN'T seen
test_examples = [
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Tesla announced it will cut 10% of its workforce while simultaneously reporting record vehicle deliveries.",
    },
    {
        "instruction": "Analyze the sentiment of this financial statement.",
        "input": "Apple's services revenue grew 18% year-over-year, now representing 25% of total revenue.",
    },
    {
        "instruction": "Explain this financial concept simply.",
        "input": "The yield curve has inverted with 2-year treasury rates exceeding 10-year rates.",
    },
]

print("\n   Testing on UNSEEN examples:\n")

for i, example in enumerate(test_examples, 1):
    print(f"   {'='*60}")
    print(f"   TEST {i}")
    print(f"   {'='*60}")
    print(f"   📋 Instruction: {example['instruction']}")
    print(f"   📥 Input: {example['input']}")
    print()

    response = generate_response(
        model, tokenizer,
        example["instruction"],
        example["input"],
    )

    print(f"   🤖 Model Response:")
    print(f"   {response}")
    print()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("                         SUMMARY")
print("="*70)
print(f"""
🎯 WHAT WE ACCOMPLISHED:
   • Loaded a pre-trained language model ({total_params:,} parameters)
   • Applied LoRA adapters (only {trainable_params:,} trainable = {100*trainable_params/(frozen_params+trainable_params):.2f}%)
   • Fine-tuned on {len(train_data)} financial examples for {NUM_EPOCHS} epochs
   • Reduced validation loss by {improvement:.1f}%

💡 KEY TAKEAWAYS:
   1. LoRA lets us fine-tune huge models on small hardware
   2. We only trained ~{100*trainable_params/(frozen_params+trainable_params):.1f}% of parameters (99%+ frozen!)
   3. The model learned financial analysis patterns
   4. Same technique works for 7B+ parameter models (Llama, Mistral)

📁 TO USE WITH LARGER MODELS:
   • Change MODEL_NAME to "mistralai/Mistral-7B-v0.1" or "meta-llama/Llama-2-7b-hf"
   • Add 4-bit quantization (BitsAndBytesConfig) to fit in GPU memory
   • Use more training data for better results

🔧 THE LoRA ADAPTER:
   • Can be saved separately (~{trainable_params * 4 / 1e6:.1f} MB)
   • Loaded on top of any copy of the base model
   • Multiple adapters can share one base model!
""")
print("="*70)
print("                    DEMO COMPLETE!")
print("="*70)
