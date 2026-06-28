"""
agent.py — Trip Concierge Agent (hand-rolled agentic loop).

Architecture
────────────
The agent sends the user goal + conversation history to Claude via the
Anthropic API, receives tool_use blocks, dispatches to local Python tools,
feeds results back, and repeats until the model emits a final text response
(no more tool calls) or the step limit is reached.

Reliability safeguard
─────────────────────
MAX_STEPS caps the number of agentic turns (default 10).  If the agent
hasn't produced a final answer by then, the loop aborts and returns a
structured error payload rather than running forever.

Safety mitigation
─────────────────
All tool arguments are validated inside each tool (see tools.py) *before*
any data access occurs.  Arguments come from the LLM and must be treated as
untrusted input — the validator rejects out-of-range values, wrong types, and
known-bad patterns (e.g. non-IATA strings for airport codes, division by zero).
This guards against prompt-injection attacks that might try to smuggle
malformed arguments through a tool result.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic

from tools import TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_STEPS = 10          # Hard cap on agentic turns (reliability safeguard)
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Tool schemas for the Anthropic API
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "search_flights",
        "description": (
            "Search available flights between two airports. "
            "Returns a list of flight options sorted cheapest-first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "IATA 3-letter departure airport code, e.g. LHR.",
                },
                "destination": {
                    "type": "string",
                    "description": "IATA 3-letter arrival airport code, e.g. OPO.",
                },
                "max_price_eur": {
                    "type": "number",
                    "description": "Optional maximum price in euros per person.",
                },
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "search_hotels",
        "description": (
            "Search hotels in a city for a given number of nights. "
            "Returns hotels sorted by best value (rating/price ratio)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. Porto.",
                },
                "nights": {
                    "type": "integer",
                    "description": "Number of nights to stay (1-30).",
                },
                "max_price_per_night_eur": {
                    "type": "number",
                    "description": "Optional maximum price per night in euros.",
                },
                "min_stars": {
                    "type": "integer",
                    "description": "Optional minimum hotel star rating (1-5).",
                },
            },
            "required": ["city", "nights"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Perform arithmetic: add, subtract, multiply, divide, or sum a list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "sum"],
                    "description": "The arithmetic operation to perform.",
                },
                "operands": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numbers to operate on.",
                },
            },
            "required": ["operation", "operands"],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Trip Concierge Agent. Your job is to help users plan
affordable trips by searching flights and hotels, then calculating total costs.

When given a trip planning goal:
1. Search for flights that fit within the budget.
2. Search for hotels that fit within the remaining budget.
3. Use the calculate tool to sum all costs and verify the total is within budget.
4. Return a structured JSON result (wrapped in ```json ... ```) with this exact shape:

{
  "trip_plan": {
    "destination": "<city>",
    "duration_days": <N>,
    "flight": {
      "airline": "<name>",
      "flight_id": "<id>",
      "origin": "<IATA>",
      "destination": "<IATA>",
      "price_eur": <number>
    },
    "hotel": {
      "name": "<name>",
      "stars": <N>,
      "price_per_night_eur": <number>,
      "nights": <N>,
      "total_hotel_eur": <number>
    },
    "cost_breakdown": {
      "flight_eur": <number>,
      "hotel_eur": <number>,
      "total_eur": <number>,
      "budget_eur": <number>,
      "within_budget": <true|false>
    },
    "notes": "<any important remarks>"
  }
}

Always use the tools — do not guess prices. Always verify the total with calculate."""


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, arguments: dict) -> dict:
    """Look up a tool by name and call it with the given arguments."""
    if name not in TOOL_REGISTRY:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    try:
        return TOOL_REGISTRY[name](**arguments)
    except TypeError as exc:
        # Wrong argument names from the model
        return {"ok": False, "error": f"Bad arguments for '{name}': {exc}"}
    except Exception as exc:
        return {"ok": False, "error": f"Tool '{name}' raised an exception: {exc}"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(goal: str, budget_eur: float = 600.0, verbose: bool = True) -> dict:
    """
    Run the Trip Concierge Agent for a given goal.

    Args:
        goal:       Natural-language trip planning request.
        budget_eur: Overall trip budget in euros (passed into the goal string).
        verbose:    Print step-by-step trace to stdout.

    Returns:
        dict with keys:
            "status"  – "success" | "step_limit_reached" | "error"
            "steps"   – list of step logs
            "result"  – parsed trip plan JSON (if successful)
            "raw_answer" – final text from the model
    """
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from environment

    messages: list[dict] = [
        {"role": "user", "content": goal}
    ]

    step_log: list[dict] = []
    final_text: str | None = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"AGENT START  |  goal: {goal}")
        print(f"MAX_STEPS: {MAX_STEPS}  |  budget: €{budget_eur}")
        print(f"{'='*60}\n")

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"── Step {step}/{MAX_STEPS} ─────────────────────────────────")

        # ── Call the model ──────────────────────────────────────────────
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        step_entry: dict[str, Any] = {
            "step": step,
            "stop_reason": response.stop_reason,
            "tool_calls": [],
        }

        # ── Collect text blocks ─────────────────────────────────────────
        text_blocks = [b.text for b in response.content if b.type == "text"]
        if text_blocks:
            step_entry["text"] = " ".join(text_blocks)
            if verbose:
                print(f"[model text] {step_entry['text'][:300]}")

        # ── Check for end of agentic loop ───────────────────────────────
        if response.stop_reason == "end_turn":
            final_text = "\n".join(text_blocks)
            step_log.append(step_entry)
            if verbose:
                print(f"\n[end_turn reached at step {step}]")
            break

        # ── Process tool calls ──────────────────────────────────────────
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            # Model stopped but no tool calls and no end_turn → unexpected
            final_text = "\n".join(text_blocks)
            step_log.append(step_entry)
            break

        # Append the assistant's message (tool_use blocks) to history
        messages.append({"role": "assistant", "content": response.content})

        # Build tool_result blocks
        tool_result_content: list[dict] = []
        for tool_use in tool_use_blocks:
            tool_name = tool_use.name
            tool_args = tool_use.input

            if verbose:
                print(f"  → tool call : {tool_name}({json.dumps(tool_args)})")

            t0 = time.perf_counter()
            result = dispatch_tool(tool_name, tool_args)
            elapsed = time.perf_counter() - t0

            if verbose:
                status = "✓ ok" if result.get("ok") else f"✗ error: {result.get('error')}"
                print(f"  ← tool result: {status}  ({elapsed*1000:.1f} ms)")
                if result.get("ok"):
                    # Print a short summary of the result
                    preview = json.dumps(result)[:200]
                    print(f"     {preview}{'...' if len(json.dumps(result)) > 200 else ''}")

            step_entry["tool_calls"].append({
                "tool": tool_name,
                "args": tool_args,
                "result_ok": result.get("ok"),
                "result_error": result.get("error"),
                "elapsed_ms": round(elapsed * 1000, 1),
            })

            tool_result_content.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result),
            })

        # Append tool results to history
        messages.append({"role": "user", "content": tool_result_content})
        step_log.append(step_entry)

    else:
        # Loop exhausted without break → step limit reached
        if verbose:
            print(f"\n[WARNING] Step limit ({MAX_STEPS}) reached without final answer.")
        return {
            "status": "step_limit_reached",
            "steps": step_log,
            "result": None,
            "raw_answer": final_text,
        }

    # ── Parse the structured JSON from the final answer ─────────────────
    parsed_result = _extract_json(final_text or "")

    if verbose:
        print(f"\n{'='*60}")
        print("FINAL STRUCTURED RESULT:")
        print(json.dumps(parsed_result, indent=2) if parsed_result else "(could not parse JSON)")
        print(f"{'='*60}\n")

    return {
        "status": "success" if parsed_result else "json_parse_error",
        "steps": step_log,
        "result": parsed_result,
        "raw_answer": final_text,
    }


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a markdown code block or raw text."""
    import re
    # Try ```json ... ``` first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GOAL = (
        "Plan a 3-day trip to Porto for one person departing from London Heathrow (LHR). "
        "My total budget is €600. Find the best value flight and hotel that keep the total "
        "under budget, then give me a full cost breakdown."
    )

    output = run_agent(goal=GOAL, budget_eur=600.0, verbose=True)

    print("\n── Machine-readable output ────────────────────────────────")
    print(json.dumps(output, indent=2))
