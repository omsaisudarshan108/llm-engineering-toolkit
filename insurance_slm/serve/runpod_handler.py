#!/usr/bin/env python3
"""
RunPod Serverless worker for the Insurance SLM.

The model loads ONCE when the container starts (outside the handler function).
RunPod then feeds inference jobs to the handler one at a time.

Job input schema:
  {
    "input": {
      "question":        "What is a deductible?",      # required
      "system_prompt":   "You are ...",                 # optional
      "max_new_tokens":  512,                           # optional, default 512
      "temperature":     0.1,                           # optional, default 0.1
      "top_p":           0.9                            # optional, default 0.9
    }
  }

Job output schema:
  {
    "answer":     "A deductible is ...",
    "latency_ms": 1234.5
  }

Deploy:
  docker buildx build --platform linux/amd64 \
    -t <dockerhub-user>/insurance-slm:latest \
    -f deploy/runpod.Dockerfile . --push
  Then create a Serverless endpoint in the RunPod console pointing at that image.
"""

import os
import sys
import time

import runpod
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_loader import load_model

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable insurance specialist for Crum & Forster. "
    "Answer questions accurately, concisely, and in plain language. "
    "Only answer based on documented policy terms. "
    "If you are unsure, say so rather than guessing."
)

# ---------------------------------------------------------------------------
# Model initialization — runs once at container start, not per request
# ---------------------------------------------------------------------------
print("[runpod_handler] Loading model...", flush=True)
_model, _tokenizer = load_model()
print("[runpod_handler] Model ready.", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|system|>\n{system_prompt.strip()}<|end|>\n"
        f"<|user|>\n{question.strip()}<|end|>\n"
        f"<|assistant|>\n"
    )


# ---------------------------------------------------------------------------
# RunPod handler — called for every job in the queue
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    question = job_input.get("question", "").strip()
    if not question:
        return {"error": "Missing required field: 'question'"}

    system_prompt  = job_input.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    max_new_tokens = max(1,   min(int(job_input.get("max_new_tokens", 512)), 2048))
    temperature    = max(0.0, min(float(job_input.get("temperature",   0.1)), 2.0))
    top_p          = max(0.0, min(float(job_input.get("top_p",         0.9)), 1.0))

    prompt = _build_prompt(question, system_prompt)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.eos_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {"answer": answer, "latency_ms": round(latency_ms, 1)}


runpod.serverless.start({"handler": handler})
