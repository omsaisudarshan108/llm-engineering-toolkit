#!/usr/bin/env python3
"""
Convert raw insurance Q&A pairs to ChatML JSONL for Phi-3.5-mini-instruct fine-tuning.

Supported input formats:
  - CSV   : columns must include question/answer (configurable field names)
  - JSON  : top-level array of objects
  - JSONL : one JSON object per line

Output:
  JSONL with a single "text" field per line containing the ChatML-formatted string.

Usage examples:
  python format_dataset.py data/raw/qa.csv
  python format_dataset.py data/raw/qa.json --output data/formatted/train.jsonl
  python format_dataset.py data/raw/qa.csv --preview 3
  python format_dataset.py data/raw/qa.csv --question-key "Q" --answer-key "A"
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable insurance specialist. "
    "Answer questions accurately, concisely, and in plain language. "
    "If you are unsure, say so rather than guessing."
)

# Phi-3.5-mini uses the ChatML token format
CHATML_TEMPLATE = (
    "<|system|>\n{system}<|end|>\n"
    "<|user|>\n{user}<|end|>\n"
    "<|assistant|>\n{assistant}<|end|>"
)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_example(question: str, answer: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> dict:
    text = CHATML_TEMPLATE.format(
        system=system_prompt.strip(),
        user=question.strip(),
        assistant=answer.strip(),
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------

def iter_csv(path: str) -> Iterator[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def iter_json(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level array of objects.")
    yield from data


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_raw(path: str) -> Iterator[dict]:
    ext = Path(path).suffix.lower()
    loaders = {".csv": iter_csv, ".json": iter_json, ".jsonl": iter_jsonl, ".ndjson": iter_jsonl}
    if ext not in loaders:
        raise ValueError(f"Unsupported extension '{ext}'. Use .csv, .json, or .jsonl")
    return loaders[ext](path)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(
    input_path: str,
    output_path: str,
    system_prompt: str,
    question_key: str,
    answer_key: str,
    system_key: str,
) -> tuple[int, int]:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for row in load_raw(input_path):
            question = row.get(question_key, "").strip()
            answer = row.get(answer_key, "").strip()
            sys_prompt = row.get(system_key, "").strip() or system_prompt

            if not question or not answer:
                skipped += 1
                continue

            out.write(json.dumps(format_example(question, answer, sys_prompt)) + "\n")
            count += 1

    return count, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Format insurance Q&A pairs into ChatML JSONL for Phi-3.5-mini fine-tuning."
    )
    parser.add_argument("input", help="Input file (.csv, .json, or .jsonl)")
    parser.add_argument(
        "--output", default="data/formatted/train.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="Default system prompt text"
    )
    parser.add_argument(
        "--question-key", default="question", help="Field name for question column"
    )
    parser.add_argument(
        "--answer-key", default="answer", help="Field name for answer column"
    )
    parser.add_argument(
        "--system-key",
        default="system_prompt",
        help="Optional field name for per-row system prompt overrides",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Print N formatted examples to stdout and exit without writing a file",
    )
    args = parser.parse_args()

    if args.preview > 0:
        for i, row in enumerate(load_raw(args.input)):
            if i >= args.preview:
                break
            q = row.get(args.question_key, "").strip()
            a = row.get(args.answer_key, "").strip()
            sp = row.get(args.system_key, "").strip() or args.system_prompt
            print(format_example(q, a, sp)["text"])
            print("---")
        return

    count, skipped = convert(
        args.input,
        args.output,
        args.system_prompt,
        args.question_key,
        args.answer_key,
        args.system_key,
    )
    print(f"Wrote {count} examples to {args.output}  ({skipped} skipped — missing question or answer)")


if __name__ == "__main__":
    main()
