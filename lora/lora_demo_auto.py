"""
LoRA Fine-Tuning Demo - Automated Version (No Input Prompts)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)

def header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

# ============== STEP 1: THE DATA ==============
header("STEP 1: FINANCIAL TRAINING DATA")
print("""
We teach the model with examples of:
  - INSTRUCTION: What to do
  - INPUT: Financial text to analyze
  - OUTPUT: Correct expert response
""")

FINANCIAL_DATA = [
    {"instruction": "Analyze sentiment.", "input": "Revenue up 25%, beating expectations.", "output": "POSITIVE. Strong growth signals robust performance."},
    {"instruction": "Analyze sentiment.", "input": "Margins to compress 200 basis points.", "output": "NEGATIVE. Reduced profitability ahead."},
    {"instruction": "Explain simply.", "input": "P/E expanded from 15x to 22x.", "output": "Investors pay more per dollar of earnings - signals optimism or overvaluation."},
    {"instruction": "Assess risk.", "input": "Pre-revenue biotech at $500M valuation.", "output": "HIGH RISK. No revenue, speculative valuation."},
    {"instruction": "Analyze sentiment.", "input": "Fed signals rate cuts.", "output": "POSITIVE for equities. Lower rates boost stocks."},
    {"instruction": "Interpret language.", "input": "Cautiously optimistic despite headwinds.", "output": "NEUTRAL. Hedged language, tempering expectations."},
    {"instruction": "Analyze sentiment.", "input": "Short interest at 25% of float.", "output": "BEARISH with squeeze risk."},
    {"instruction": "Evaluate dividend.", "input": "15% dividend hike, 25th year.", "output": "POSITIVE. Dividend Aristocrat, signals confidence."},
    {"instruction": "Analyze sentiment.", "input": "Negative operating cash flow.", "output": "CAUTIOUS. Context matters, monitor closely."},
    {"instruction": "Explain pattern.", "input": "Death cross formed.", "output": "BEARISH. Short-term below long-term trend."},
    {"instruction": "Analyze sentiment.", "input": "$2B buyback announced.", "output": "POSITIVE. Boosts EPS, signals undervaluation."},
    {"instruction": "Assess debt.", "input": "Debt covenant at 4.0x.", "output": "RISK. Breach triggers acceleration."},
]

print(f"Created {len(FINANCIAL_DATA)} training examples")

# ============== STEP 2: DATASET CLASS ==============
header("STEP 2: CONVERT TEXT TO NUMBERS")
print("""
Neural networks need numbers, not text:
  "Stock rose" → [464, 4283, 6348]
""")

class FinancialDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        ids, mask = enc["input_ids"].squeeze(), enc["attention_mask"].squeeze()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

# ============== STEP 3: LOAD BASE MODEL ==============
header("STEP 3: LOAD PRE-TRAINED MODEL (FROZEN)")
print("""
Loading GPT-2 (124M parameters) - same concepts as Llama/Mistral.
This model will be FROZEN - we won't change its weights.
""")

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded (vocab: {tokenizer.vocab_size:,})")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.config.use_cache = False
total_base = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded ({total_base:,} parameters)")

# ============== STEP 4: APPLY LoRA ==============
header("STEP 4: APPLY LoRA ADAPTERS (THE MAGIC)")
print("""
Instead of training 124M parameters, we add TINY adapters:
  - Original: output = input × W (W is frozen)
  - With LoRA: output = input × W + input × A × B (A,B are tiny & trained)

Config:
  - Rank (r=16): Size of adapter matrices
  - Alpha (32): Scaling factor
  - Target: Attention layers (c_attn, c_proj)
""")

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
pct = 100 * trainable / (trainable + frozen)

print(f"""
╔════════════════════════════════════════════════╗
║  🧊 FROZEN:    {frozen:>12,} params (untouched) ║
║  🔥 TRAINABLE: {trainable:>12,} params (LoRA)    ║
║  📊 Training only {pct:.2f}% of the model!       ║
╚════════════════════════════════════════════════╝
""")

# ============== STEP 5: PREPARE DATA ==============
header("STEP 5: PREPARE TRAINING DATA")
train_data, val_data = FINANCIAL_DATA[:10], FINANCIAL_DATA[10:]
train_dataset = FinancialDataset(train_data, tokenizer)
val_dataset = FinancialDataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
print(f"Training: {len(train_data)} examples | Validation: {len(val_data)} examples")

# ============== STEP 6: OPTIMIZER ==============
header("STEP 6: SETUP OPTIMIZER")
print("""
The optimizer nudges weights to reduce prediction errors:
  - Learning rate: How big each nudge is (2e-4)
  - Warmup: Start small, then increase (stability)
""")

LR, EPOCHS = 2e-4, 3
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
print(f"Learning rate: {LR} | Epochs: {EPOCHS} | Steps: {total_steps}")

# ============== STEP 7: TRAINING ==============
header("STEP 7: TRAINING LOOP")
print("""
For each batch:
  1. Forward pass → get predictions
  2. Calculate loss → how wrong are we?
  3. Backward pass → compute gradients
  4. Update → nudge LoRA weights
""")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}\n")
model = model.to(device)

def evaluate(model, loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), labels=b["labels"].to(device))
            total += out.loss.item()
    return total / len(loader)

baseline_loss = evaluate(model, val_loader)
print(f"📊 BASELINE loss (before training): {baseline_loss:.4f}")
print(f"   Perplexity: {math.exp(baseline_loss):.1f}\n")

history = {"train": [], "val": [baseline_loss]}

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    avg_train = epoch_loss / len(train_loader)
    val_loss = evaluate(model, val_loader)
    history["train"].append(avg_train)
    history["val"].append(val_loss)
    print(f"   Train: {avg_train:.4f} | Val: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.1f}")

# ============== STEP 8: RESULTS ==============
header("STEP 8: BEFORE vs AFTER COMPARISON")

before, after = history["val"][0], history["val"][-1]
improvement = (before - after) / before * 100

print(f"""
╔══════════════════════════════════════════════════════╗
║              TRAINING RESULTS                        ║
╠══════════════════════════════════════════════════════╣
║  Validation Loss:                                    ║
║    BEFORE: {before:.4f}  →  AFTER: {after:.4f}             ║
║    Improvement: {improvement:.1f}%                              ║
║                                                      ║
║  Perplexity (lower = better):                        ║
║    BEFORE: {math.exp(before):.1f}  →  AFTER: {math.exp(after):.1f}               ║
║    (Model is {math.exp(before)/math.exp(after):.1f}x more confident!)             ║
╚══════════════════════════════════════════════════════╝
""")

# ============== STEP 9: GENERATE ==============
header("STEP 9: TEST ON NEW EXAMPLES")
print("Let's see how the fine-tuned model responds to UNSEEN inputs:\n")

def generate(instruction, input_text):
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True,
                            top_p=0.9, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.2)
    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    return resp.split("### Response:")[-1].strip().split("\n")[0] if "### Response:" in resp else resp[-100:]

tests = [
    ("Analyze sentiment.", "Tesla cuts 10% workforce while reporting record deliveries."),
    ("Analyze sentiment.", "Apple services revenue grew 18% YoY."),
    ("Explain simply.", "Yield curve inverted - 2yr exceeds 10yr rates."),
]

for i, (inst, inp) in enumerate(tests, 1):
    print(f"TEST {i}:")
    print(f"  Input: {inp}")
    print(f"  Model: {generate(inst, inp)}")
    print()

# ============== SUMMARY ==============
header("SUMMARY: WHAT WE LEARNED")
print(f"""
✓ Loaded GPT-2 with {total_base:,} parameters
✓ Applied LoRA - trained only {trainable:,} params ({pct:.2f}%)
✓ Fine-tuned on {len(train_data)} financial examples
✓ Reduced loss by {improvement:.1f}% (perplexity: {math.exp(before):.0f} → {math.exp(after):.0f})

KEY INSIGHT: LoRA lets us specialize huge models on consumer hardware
by training <1% of parameters while keeping the rest frozen.

Same technique works for Llama-7B, Mistral-7B, etc. - just add:
  - 4-bit quantization (BitsAndBytesConfig)
  - More training data
  - GPU with 16GB+ VRAM
""")
