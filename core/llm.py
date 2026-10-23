"""core/llm.py – Ollama HTTP client, JSON-mode extraction, tool-calling stub."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

_CFG = yaml.safe_load(Path("config.yaml").read_text())["ollama"]

BASE_URL: str = _CFG["base_url"]
CHAT_MODEL: str = _CFG["chat_model"]
EMBED_MODEL: str = _CFG["embed_model"]
TIMEOUT: float = float(_CFG["timeout_s"])
MAX_RETRIES: int = int(_CFG["max_retries"])

_GENERATE_URL = f"{BASE_URL}/api/generate"
_CHAT_URL = f"{BASE_URL}/api/chat"
_EMBED_URL = f"{BASE_URL}/api/embeddings"
_TAGS_URL = f"{BASE_URL}/api/tags"


# ─── Health check ─────────────────────────────────────────────────────────────

def check_ollama() -> None:
    """Raise a clear RuntimeError if Ollama is not reachable."""
    try:
        r = httpx.get(f"{BASE_URL}/api/version", timeout=5)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        raise RuntimeError(
            f"\n[Ollama not running]\n"
            f"  Start Ollama: run `ollama serve` in a terminal\n"
            f"  Then verify:  ollama list\n"
            f"  Original error: {exc}"
        ) from exc


# ─── Chat (streaming=False) ────────────────────────────────────────────────────

def chat(
    prompt: str,
    system: str = "",
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> str:
    """Single-turn chat. Returns assistant content string."""
    check_ollama()
    mdl = model or CHAT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": mdl,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    r = httpx.post(_CHAT_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chat_with_retry(
    prompt: str,
    system: str = "",
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> str:
    """chat() with up to MAX_RETRIES retries on failure."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return chat(prompt, system=system, model=model,
                        json_mode=json_mode, temperature=temperature)
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts") from last_exc


# ─── Multi-turn (tool-calling simulation) ─────────────────────────────────────

def multi_turn_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Multi-turn conversation. messages = [{role, content}, ...]."""
    check_ollama()
    payload: dict[str, Any] = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = httpx.post(_CHAT_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ─── Embeddings ───────────────────────────────────────────────────────────────

def embed(text: str, model: str | None = None) -> list[float]:
    check_ollama()
    payload = {"model": model or EMBED_MODEL, "prompt": text}
    r = httpx.post(_EMBED_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["embedding"]
