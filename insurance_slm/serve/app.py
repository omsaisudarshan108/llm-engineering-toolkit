#!/usr/bin/env python3
"""
FastAPI inference server for the insurance SLM.

Endpoints:
  GET  /health          — liveness / readiness probe
  POST /v1/chat         — single-turn Q&A
  POST /v1/chat/stream  — streaming Q&A (SSE)

Run locally:
  uvicorn serve.app:app --host 0.0.0.0 --port 8080 --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — populated once at startup
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable insurance specialist. "
    "Answer questions accurately, concisely, and in plain language. "
    "If you are unsure, say so rather than guessing."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _model, _tokenizer
    logger.info("Loading model — this may take a minute on first start...")
    _model, _tokenizer = load_model()
    logger.info("Model ready. Server accepting requests.")
    yield
    del _model, _tokenizer
    logger.info("Model unloaded.")


app = FastAPI(
    title="Insurance SLM",
    description="Fine-tuned Phi-3.5-mini-instruct for insurance Q&A",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    system_prompt: Optional[str] = Field(None, description="Override default system prompt")
    max_new_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    answer: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|system|>\n{system_prompt.strip()}<|end|>\n"
        f"<|user|>\n{question.strip()}<|end|>\n"
        f"<|assistant|>\n"
    )


def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> tuple[str, float]:
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    system = req.system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt = _build_prompt(req.question, system)
    answer, latency_ms = _generate(prompt, req.max_new_tokens, req.temperature, req.top_p)
    return ChatResponse(answer=answer, latency_ms=latency_ms)


@app.post("/v1/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Server-Sent Events streaming endpoint.
    Yields answer tokens as they are generated using TextIteratorStreamer.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    from threading import Thread
    from transformers import TextIteratorStreamer

    system = req.system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt = _build_prompt(req.question, system)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature if req.temperature > 0 else None,
        top_p=req.top_p,
        do_sample=req.temperature > 0,
        pad_token_id=_tokenizer.eos_token_id,
        eos_token_id=_tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=_model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()

    def event_generator():
        for token in streamer:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
