"""
Model loading logic for the inference server.

Environment variables (set in fly.toml or .env):
  BASE_MODEL_ID   HuggingFace model ID          (default: microsoft/Phi-3.5-mini-instruct)
  ADAPTER_PATH    Path to saved LoRA adapter     (default: /app/adapter)
  MODEL_CACHE_DIR Path where HF downloads cache  (default: /models/cache)
  USE_4BIT        Enable 4-bit NF4 quantization  (default: true)
  MERGE_ON_LOAD   Merge LoRA into base at load   (default: false)
  HF_TOKEN        HuggingFace access token       (optional)
"""

import logging
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "microsoft/Phi-3.5-mini-instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/app/adapter")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models/cache")
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"
MERGE_ON_LOAD = os.getenv("MERGE_ON_LOAD", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN") or None


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model():
    """
    Load the base model + optional LoRA adapter.

    Resolution order for the tokenizer:
      1. ADAPTER_PATH (contains tokenizer files if saved with trainer.save_model())
      2. BASE_MODEL_ID (fallback to HuggingFace Hub)

    Returns:
      (model, tokenizer) — model is in eval mode, moved to available device.
    """
    # A real adapter directory contains adapter_config.json; a bare mkdir or .gitkeep does not.
    has_adapter = os.path.isfile(os.path.join(ADAPTER_PATH, "adapter_config.json"))
    tokenizer_source = ADAPTER_PATH if has_adapter else BASE_MODEL_ID
    device = detect_device()
    use_4bit = USE_4BIT and device == "cuda"

    if USE_4BIT and device != "cuda":
        logger.warning("USE_4BIT requested, but %s is active. Falling back to non-quantized load.", device)

    logger.info("Loading tokenizer from %s", tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s  (device=%s, 4-bit=%s)", BASE_MODEL_ID, device, use_4bit)
    load_kwargs = dict(
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="eager",
    )

    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        load_kwargs["device_map"] = {"": "mps"}
    else:
        load_kwargs["device_map"] = {"": "cpu"}

    if use_4bit:
        load_kwargs["quantization_config"] = _bnb_config()
        load_kwargs["torch_dtype"] = torch.float16
    elif device == "mps":
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)

    if has_adapter:
        logger.info("Loading LoRA adapter from %s", ADAPTER_PATH)
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        if MERGE_ON_LOAD:
            logger.info("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
    else:
        logger.warning(
            "No adapter found at %s — serving base model without fine-tuning.", ADAPTER_PATH
        )

    model.eval()
    logger.info("Model ready on device: %s", next(model.parameters()).device)
    return model, tokenizer
