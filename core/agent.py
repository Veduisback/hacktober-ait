"""core/agent.py – LLM agent loop with tool-calling and RAG-grounded recommendations.

The agent uses a ReAct-style loop:
  Thought → Tool Call → Observation → ... → Final Answer

Tools available:
  get_stock(item, facility)          → current stock qty
  forecast(item, facility)           → days of stock remaining
  find_surplus_facility(item, facility) → list of facilities with surplus
  affected_departments(item)         → departments depending on this item
  search_guidelines(query)           → relevant SOP / policy snippets

The full trace (thoughts + tool calls + observations) is returned alongside
the final recommendation for display in the UI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core import db
from core.extract import DisruptionExtract
from core.forecast import forecast_days_of_stock
from core.llm import multi_turn_chat
from core.rag import retrieve_guidelines

_CFG = yaml.safe_load(Path("config.yaml").read_text())

# ─── Tool definitions ─────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    kind: str          # "thought" | "tool_call" | "observation" | "answer"
    content: str


@dataclass
class AgentResult:
    recommendation: str
    trace: list[TraceStep] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)  # RAG sources


# ─── Tool implementations ─────────────────────────────────────────────────────

def _tool_get_stock(item_name: str, facility_id: str) -> str:
    item = db.get_item_by_name(item_name)
    if not item:
        return json.dumps({"error": f"Item '{item_name}' not found in database."})
    rows = db.get_stock(facility_id, item["id"])
    if not rows:
        return json.dumps({"error": f"No stock records for '{item_name}' at {facility_id}."})
    total = sum(r["qty"] for r in rows)
    return json.dumps({"item": item_name, "facility": facility_id,
                       "total_qty": total, "unit": rows[0]["unit"]})


def _tool_forecast(item_name: str, facility_id: str) -> str:
    item = db.get_item_by_name(item_name)
    if not item:
        return json.dumps({"error": f"Item '{item_name}' not found."})
    stock_rows = db.get_stock(facility_id, item["id"])
    total_stock = sum(r["qty"] for r in stock_rows)
    consumption = db.get_consumption(facility_id, item["id"], days=30)
    result = forecast_days_of_stock(
        consumption_records=consumption,
        current_stock=total_stock,
        item_id=item["id"],
        facility_id=facility_id,
        unit=item.get("unit", "units"),
    )
    days = result.days_of_stock
    return json.dumps({
        "item": item_name,
        "facility": facility_id,
        "days_of_stock_remaining": days if days != float("inf") else 9999,
        "daily_consumption": result.daily_consumption,
        "method": result.method,
        "unit": result.unit,
    })


def _tool_find_surplus(item_name: str, facility_id: str) -> str:
    item = db.get_item_by_name(item_name)
    if not item:
        return json.dumps({"error": f"Item '{item_name}' not found."})
    surplus = db.find_surplus_facilities(item["id"], facility_id)
    return json.dumps({"surplus_facilities": surplus})


def _tool_affected_departments(item_name: str) -> str:
    item = db.get_item_by_name(item_name)
    if not item:
        return json.dumps({"error": f"Item '{item_name}' not found."})
    depts = db.get_affected_departments(item["id"])
    return json.dumps({"item": item_name, "affected_departments": depts})


def _tool_search_guidelines(query: str) -> str:
    results = retrieve_guidelines(query, n=3)
    return json.dumps({"results": results})


_TOOLS: dict[str, Any] = {
    "get_stock": _tool_get_stock,
    "forecast": _tool_forecast,
    "find_surplus_facility": _tool_find_surplus,
    "affected_departments": _tool_affected_departments,
    "search_guidelines": _tool_search_guidelines,
}


# ─── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_AGENT = """You are a hospital supply-chain risk analyst AI assistant.
You have access to tools to look up stock levels, forecast stock-outs,
find surplus at other facilities, and search operational guidelines.

Available tools (call them as JSON in your response):
- get_stock(item, facility): Returns current on-hand stock.
- forecast(item, facility): Returns how many days of stock remain.
- find_surplus_facility(item, facility): Returns other facilities with surplus stock.
- affected_departments(item): Returns departments/procedures that use this item.
- search_guidelines(query): Searches internal SOPs and policies.

RESPONSE FORMAT (strictly follow this):
When you want to call a tool, respond EXACTLY like:
THOUGHT: <your reasoning>
TOOL: <tool_name>
ARGS: {"arg1": "val1", "arg2": "val2"}

When you have enough information to write the final recommendation:
THOUGHT: <final reasoning>
ANSWER: <your recommendation (2-4 sentences, cite sources, flag for human review)>

Rules:
- Never claim certainty. Always show confidence levels.
- If risk is HIGH, explicitly say "FLAG FOR HUMAN REVIEW".
- Cite the guideline snippet and the original email in your answer.
- Keep the recommendation concise and actionable.
"""


def _parse_tool_call(text: str) -> tuple[str | None, dict]:
    """Extract tool name and args from a TOOL:/ARGS: block."""
    tool_match = re.search(r"TOOL:\s*(\w+)", text)
    args_match = re.search(r"ARGS:\s*(\{.*?\})", text, re.DOTALL)
    if not tool_match or not args_match:
        return None, {}
    tool_name = tool_match.group(1).strip()
    try:
        args = json.loads(args_match.group(1))
    except json.JSONDecodeError:
        args = {}
    return tool_name, args


def _parse_answer(text: str) -> str | None:
    match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL)
    return match.group(1).strip() if match else None


# ─── Agent loop ───────────────────────────────────────────────────────────────

def run_agent(
    extraction: DisruptionExtract,
    facility_id: str,
    max_steps: int = 8,
) -> AgentResult:
    """
    Run the ReAct agent loop for a given disruption extraction.

    Returns AgentResult with the final recommendation and full trace.
    """
    trace: list[TraceStep] = []
    citations: list[dict] = []

    initial_message = (
        f"A supply disruption has been reported for facility '{facility_id}':\n"
        f"  Item: {extraction.item}\n"
        f"  Quantity affected: {extraction.qty or 'not specified'}\n"
        f"  Delay: {extraction.new_eta_days} additional days\n"
        f"  Reason: {extraction.reason}\n"
        f"  Extraction confidence: {extraction.confidence:.0%}\n\n"
        f"Analyze the risk, identify affected departments, check for surplus "
        f"at other facilities, consult guidelines, and write a recommendation."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_AGENT},
        {"role": "user", "content": initial_message},
    ]

    for step in range(max_steps):
        response = multi_turn_chat(messages)
        messages.append({"role": "assistant", "content": response})

        # Extract thought
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=TOOL:|ANSWER:|$)", response, re.DOTALL)
        if thought_match:
            trace.append(TraceStep(kind="thought", content=thought_match.group(1).strip()))

        # Check for final answer
        answer = _parse_answer(response)
        if answer:
            trace.append(TraceStep(kind="answer", content=answer))
            # Collect RAG citations from the last search_guidelines call
            break

        # Check for tool call
        tool_name, args = _parse_tool_call(response)
        if not tool_name or tool_name not in _TOOLS:
            # Model didn't follow format — force an answer
            trace.append(TraceStep(kind="thought", content="Could not parse tool call, stopping."))
            answer = "Unable to complete analysis — model did not follow expected format. Please review manually."
            trace.append(TraceStep(kind="answer", content=answer))
            break

        trace.append(TraceStep(kind="tool_call",
                               content=f"{tool_name}({json.dumps(args)})"))

        # Execute tool
        fn = _TOOLS[tool_name]
        try:
            obs = fn(**args)
        except TypeError as exc:
            obs = json.dumps({"error": str(exc)})

        # Collect guideline citations
        if tool_name == "search_guidelines":
            try:
                results = json.loads(obs).get("results", [])
                citations.extend(results)
            except Exception:
                pass

        trace.append(TraceStep(kind="observation", content=obs))
        messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})

    else:
        # Exceeded max_steps
        answer = (
            "Analysis incomplete — reached maximum reasoning steps. "
            "Please escalate to the supply-chain manager for manual review."
        )
        trace.append(TraceStep(kind="answer", content=answer))

    return AgentResult(
        recommendation=answer or "No recommendation generated.",
        trace=trace,
        citations=citations,
    )
