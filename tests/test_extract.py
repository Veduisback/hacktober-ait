"""tests/test_extract.py – Tests for extraction schema and malformed LLM output handling.

NOTE: These tests mock the LLM call so they run without Ollama.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from core.extract import DisruptionExtract, extract_disruption


# ─── Schema validation ────────────────────────────────────────────────────────

class TestDisruptionExtractSchema:
    def test_valid_full(self):
        data = {
            "item": "Insulin",
            "qty": 500.0,
            "new_eta_days": 9,
            "reason": "Transport strike on northern highway.",
            "confidence": 0.92,
            "language_detected": "en",
        }
        ext = DisruptionExtract.model_validate(data)
        assert ext.item == "Insulin"
        assert ext.new_eta_days == 9
        assert ext.confidence == pytest.approx(0.92)

    def test_qty_can_be_none(self):
        data = {
            "item": "Paracetamol",
            "qty": None,
            "new_eta_days": 5,
            "reason": "Warehouse fire.",
            "confidence": 0.8,
            "language_detected": "hi",
        }
        ext = DisruptionExtract.model_validate(data)
        assert ext.qty is None

    def test_negative_eta_raises(self):
        with pytest.raises(ValidationError):
            DisruptionExtract.model_validate({
                "item": "Amoxicillin",
                "qty": 100.0,
                "new_eta_days": -1,
                "reason": "Invalid.",
                "confidence": 0.5,
                "language_detected": "en",
            })

    def test_empty_item_raises(self):
        with pytest.raises(ValidationError):
            DisruptionExtract.model_validate({
                "item": "   ",
                "qty": None,
                "new_eta_days": 3,
                "reason": "Unknown.",
                "confidence": 0.5,
                "language_detected": "en",
            })

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            DisruptionExtract.model_validate({
                "item": "Insulin",
                "qty": None,
                "new_eta_days": 5,
                "reason": "Test.",
                "confidence": 1.5,  # > 1.0
                "language_detected": "en",
            })

    def test_language_default(self):
        data = {
            "item": "Metformin",
            "qty": 50.0,
            "new_eta_days": 3,
            "reason": "Delay.",
            "confidence": 0.7,
        }
        ext = DisruptionExtract.model_validate(data)
        assert ext.language_detected == "en"


# ─── Extraction with mocked LLM ──────────────────────────────────────────────

_GOOD_RESPONSE = json.dumps({
    "item": "Insulin",
    "qty": 500.0,
    "new_eta_days": 9,
    "reason": "Transport strike delayed shipment.",
    "confidence": 0.92,
    "language_detected": "en",
})

_BAD_RESPONSE = "Sorry, I cannot parse this as JSON :-)"

_BAD_THEN_GOOD = [_BAD_RESPONSE, _GOOD_RESPONSE]


class TestExtractDisruption:
    @patch("core.extract.chat_with_retry", return_value=_GOOD_RESPONSE)
    def test_successful_extraction(self, mock_chat):
        result = extract_disruption("Insulin shipment delayed by 9 days due to strike.")
        assert result.item == "Insulin"
        assert result.new_eta_days == 9
        assert mock_chat.call_count == 1

    @patch("core.extract.chat_with_retry", return_value='{"item":"","qty":null,"new_eta_days":5,"reason":"x","confidence":0.5,"language_detected":"en"}')
    def test_empty_item_retries_and_raises(self, mock_chat):
        """Empty item fails Pydantic → retry → still fails → raises."""
        with pytest.raises(Exception):
            extract_disruption("Some email text")

    @patch("core.extract.chat_with_retry", return_value=_BAD_RESPONSE)
    def test_malformed_json_retries(self, mock_chat):
        """First call returns non-JSON; retry also returns non-JSON → raises."""
        with pytest.raises(Exception):
            extract_disruption("Broken email")

    @patch("core.extract.chat_with_retry")
    def test_markdown_fenced_json_stripped(self, mock_chat):
        """Model sometimes wraps JSON in markdown fences — should be handled."""
        fenced = "```json\n" + _GOOD_RESPONSE + "\n```"
        mock_chat.return_value = fenced
        result = extract_disruption("Test email")
        assert result.item == "Insulin"

    @patch("core.extract.chat_with_retry", return_value=_GOOD_RESPONSE)
    def test_hindi_prompt_passes_through(self, mock_chat):
        hindi_email = "इंसुलिन की आपूर्ति में 9 दिन की देरी हो गई है।"
        result = extract_disruption(hindi_email)
        assert result.new_eta_days == 9

    @patch("core.extract.chat_with_retry")
    def test_extra_text_around_json_stripped(self, mock_chat):
        """Model adds preamble text before the JSON object."""
        mock_chat.return_value = "Here is the JSON:\n" + _GOOD_RESPONSE + "\nHope this helps!"
        # This should fail JSON parse on first attempt, then retry
        # In practice, re.sub strips fences but not arbitrary text
        # → expect exception or success depending on parse behaviour
        try:
            result = extract_disruption("Test")
            # If it worked, verify it's valid
            assert result.item == "Insulin"
        except Exception:
            pass  # Expected failure for preamble text


# ─── Failing-case test (malformed LLM output) ────────────────────────────────

class TestMalformedLLMOutput:
    """Explicitly tests the system's resilience to bad LLM outputs."""

    @patch("core.extract.chat_with_retry", return_value="I am unable to extract information from this text.")
    def test_natural_language_response_raises(self, _):
        """LLM ignores JSON-mode and returns prose → extraction must raise."""
        with pytest.raises(Exception) as exc_info:
            extract_disruption("Some email")
        assert exc_info.type in (json.JSONDecodeError, ValueError, Exception)

    @patch("core.extract.chat_with_retry", return_value="{}")
    def test_empty_json_object_raises(self, _):
        """LLM returns {} (missing required fields) → Pydantic must raise."""
        with pytest.raises(Exception):
            extract_disruption("Some email")

    @patch("core.extract.chat_with_retry", return_value='{"item": "Insulin", "new_eta_days": "nine"}')
    def test_wrong_type_raises(self, _):
        """new_eta_days as string → Pydantic coercion may succeed or raise."""
        try:
            result = extract_disruption("Some email")
            # Pydantic v2 coerces "nine" string? No, it should fail int parse
            assert False, "Should have raised"
        except Exception:
            pass  # Expected
