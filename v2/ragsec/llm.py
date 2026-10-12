"""OpenAI chat calls with retries and a running token and cost count."""
from __future__ import annotations

import json
import os
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

from .config import REPO

load_dotenv(REPO / ".env")

# USD per million tokens (input, output)
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60)}

_client: OpenAI | None = None
_lock = threading.Lock()
USAGE: dict[str, list[int]] = {}


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _record(model: str, usage) -> None:
    with _lock:
        tin, tout = USAGE.setdefault(model, [0, 0])
        USAGE[model] = [tin + usage.prompt_tokens, tout + usage.completion_tokens]


def cost_of(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (tokens_in * pin + tokens_out * pout) / 1e6


def total_cost() -> float:
    return sum(cost_of(m, *t) for m, t in USAGE.items())


def chat(model: str, messages: list[dict], temperature: float = 0.0, json_mode: bool = False,
         max_tokens: int = 800) -> tuple[str, dict]:
    """Returns (text, {"tokens_in", "tokens_out", "seconds", "cost"})."""
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    for attempt in range(5):
        try:
            t = time.time()
            resp = client().chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
            _record(model, resp.usage)
            meta = {"tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens,
                    "seconds": round(time.time() - t, 2)}
            meta["cost"] = cost_of(model, meta["tokens_in"], meta["tokens_out"])
            return resp.choices[0].message.content or "", meta
        except Exception as e:  # rate limits and transient errors
            if attempt == 4:
                raise
            time.sleep(2 ** attempt + 1)
            last = e  # noqa: F841
    raise RuntimeError("unreachable")


def chat_json(model: str, prompt: str) -> dict | None:
    text, _ = chat(model, [{"role": "user", "content": prompt}], json_mode=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
