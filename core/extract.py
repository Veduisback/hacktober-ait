"""core/extract.py – Extract structured disruption info from supplier emails/POs.

Supports English and Hindi input. Uses Ollama JSON-mode with a Pydantic schema.
Retries once on invalid output.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from core.llm import chat_with_retry

# ─── Schema ───────────────────────────────────────────────────────────────────

class DisruptionExtract(BaseModel):
    """Structured supply-chain disruption extracted from a supplier email or PO."""

    item: str = Field(description="Name of the medical supply / medicine affected.")
    qty: float | None = Field(None, description="Quantity affected (in original units). Null if not mentioned.")
    new_eta_days: int = Field(description="Additional delay in days before shipment arrives.")
    reason: str = Field(description="Brief reason for the disruption (1–2 sentences).")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in the extraction (0.0–1.0).",
    )
    language_detected: Literal["en", "hi", "other"] = Field(
        default="en",
        description="Language of the input text.",
    )

    @field_validator("new_eta_days")
    @classmethod
    def eta_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("new_eta_days must be ≥ 0")
        return v

    @field_validator("item")
    @classmethod
    def item_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("item must not be empty")
        return v


# ─── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM = """You are a supply-chain disruption parser for government hospitals.
Extract information from supplier emails, purchase orders, or SMS notifications.
The input may be in English or Hindi (Devanagari or Roman script).
Return ONLY a valid JSON object matching the schema below. No extra text.

Schema:
{
  "item": "<medicine or supply name, in English>",
  "qty": <number or null>,
  "new_eta_days": <integer, additional delay days>,
  "reason": "<brief reason>",
  "confidence": <float 0.0-1.0>,
  "language_detected": "<en|hi|other>"
}

Rules:
- Translate item names to English if the input is in Hindi.
- If the email mentions a date, convert it to delay days from today.
- If quantity is not mentioned, set qty to null.
- Set confidence low (<0.6) if you are unsure about any field.
- Never invent information not present in the email.
"""

_USER_TMPL = "Parse the following supplier communication:\n\n---\n{text}\n---"


# ─── Extractor ────────────────────────────────────────────────────────────────

def extract_disruption(raw_text: str) -> DisruptionExtract:
    """
    Extract structured disruption data from a supplier email / PO.

    Retries once if the LLM returns invalid JSON or fails Pydantic validation.
    Raises ValueError on second failure.
    """
    prompt = _USER_TMPL.format(text=raw_text.strip())

    def _attempt() -> DisruptionExtract:
        content = chat_with_retry(prompt, system=_SYSTEM, json_mode=True, temperature=0.0)
        # Strip markdown fences if model wraps the JSON
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.DOTALL)
        data = json.loads(cleaned)
        return DisruptionExtract.model_validate(data)

    try:
        return _attempt()
    except (json.JSONDecodeError, ValueError, Exception):
        # One retry with a stricter reminder
        stricter_prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY the raw JSON object. No markdown. No extra text."
        )
        content = chat_with_retry(
            stricter_prompt, system=_SYSTEM, json_mode=True, temperature=0.0
        )
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.DOTALL)
        data = json.loads(cleaned)
        return DisruptionExtract.model_validate(data)
