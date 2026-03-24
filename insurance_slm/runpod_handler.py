import time

import runpod
import torch

from serve.model_loader import load_model

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable insurance specialist. "
    "Answer questions accurately, concisely, and in plain language. "
    "If you are unsure, say so rather than guessing."
)

_model = None
_tokenizer = None


def _ensure_loaded():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_model()


def _build_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|system|>\n{system_prompt.strip()}<|end|>\n"
        f"<|user|>\n{question.strip()}<|end|>\n"
        f"<|assistant|>\n"
    )


def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
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
    return answer, round(latency_ms, 1)


def handler(job):
    _ensure_loaded()

    job_input = job.get("input", {})
    question = job_input.get("question")
    if not isinstance(question, str) or not question.strip():
        return {"error": "input.question must be a non-empty string"}

    system_prompt = job_input.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    max_new_tokens = int(job_input.get("max_new_tokens", 512))
    temperature = float(job_input.get("temperature", 0.1))
    top_p = float(job_input.get("top_p", 0.9))

    prompt = _build_prompt(question, system_prompt)
    answer, latency_ms = _generate(prompt, max_new_tokens, temperature, top_p)

    return {
        "answer": answer,
        "latency_ms": latency_ms,
    }


runpod.serverless.start({"handler": handler})
