#!/usr/bin/env python3
"""
QLoRA fine-tuning of Phi-3.5-mini-instruct on insurance Q&A data.

Hardware modes (auto-detected):
  CUDA  — Full QLoRA: 4-bit NF4 quantization + LoRA. Requires NVIDIA GPU + bitsandbytes.
           Model fits in ~4 GB VRAM. This is the production training path.
  MPS   — Apple Silicon (M1/M2/M3): LoRA in float16, no quantization.
           Slow, but functional for smoke-testing on macOS with small datasets.
  CPU   — LoRA in float32, no quantization. Very slow; dev/CI only.

Steps executed:
  1. Load ChatML-formatted JSONL dataset
  2. Configure quantization (4-bit NF4 on CUDA, none otherwise)
  3. Load Phi-3.5-mini-instruct
  4. Inject trainable LoRA adapters via get_peft_model()
  5. Run SFTTrainer
  6. Save LoRA adapter (and optionally merge into base model)

Requirements:
  pip install -r requirements-train.txt
  # bitsandbytes is only required on CUDA; the script skips it on macOS/CPU

Usage:
  python train.py
  python train.py --data_path data/formatted/train.jsonl --output_dir models/my-adapter
  python train.py --merge_weights
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = detect_device()
USE_4BIT = DEVICE == "cuda"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def bnb_config():
    """4-bit NF4 quantization — CUDA only."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if USE_4BIT:
        load_kwargs["quantization_config"] = bnb_config()
        load_kwargs["torch_dtype"] = torch.float16
    elif DEVICE == "mps":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = {"": "mps"}
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    print(f"Device: {DEVICE.upper()}  |  4-bit quantization: {USE_4BIT}")
    if DEVICE != "cuda":
        print(
            f"\n  NOTE: Running on {DEVICE.upper()} without 4-bit quantization.\n"
            "  The full model (~7.6 GB) will load in fp16/fp32 — this is slow and\n"
            "  memory-intensive. For real training, use a machine with an NVIDIA GPU.\n"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset
    print(f"\n[1/6] Loading dataset: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"      {len(dataset):,} examples loaded.")

    # 2 & 3. Load model (quantization config is embedded in load_model_and_tokenizer)
    print(f"[2/6] Loading base model: {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # 4. Apply LoRA adapters
    print(f"[3/6] Applying LoRA adapters  (r={args.lora_r}, alpha={args.lora_alpha})")
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(args.lora_r, args.lora_alpha, args.lora_dropout))
    model.print_trainable_parameters()

    # 5. Train
    print("[4/6] Configuring SFTTrainer...")
    use_fp16 = DEVICE == "cuda"
    use_bf16 = False

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=False,
        optim="paged_adamw_32bit" if USE_4BIT else "adamw_torch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    print("[5/6] Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 6. Save
    print(f"[6/6] Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_weights:
        _merge_and_save(args.output_dir)

    print("\nDone.")


def _merge_and_save(adapter_dir: str):
    from peft import AutoPeftModelForCausalLM
    merged_dir = adapter_dir.rstrip("/") + "-merged"
    print(f"Merging LoRA weights → {merged_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for the insurance SLM")
    p.add_argument("--model_id", default=BASE_MODEL_ID)
    p.add_argument("--data_path", default="data/formatted/train.jsonl")
    p.add_argument("--output_dir", default="models/insurance-phi35-lora")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--merge_weights",
        action="store_true",
        help="After training, merge LoRA adapter into the base model weights",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
