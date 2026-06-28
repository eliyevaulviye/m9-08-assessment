"""
tools.py — Three tools available to the Trip Concierge Agent.

Each tool:
  1. Validates its arguments before touching any data (safety mitigation).
  2. Returns a typed dict so callers can detect errors without parsing strings.
  3. Never executes side-effects that can't be rolled back (read-only data).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> Any:
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _tool_error(message: str) -> dict:
    """Standardised error envelope returned by every tool on bad input."""
    return {"ok": False, "error": message}


def _tool_ok(payload: dict) -> dict:
    """Standardised success envelope."""
    return {"ok": True, **payload}


# ---------------------------------------------------------------------------
# SAFETY: argument schema validation
# ---------------------------------------------------------------------------
# We treat every tool call as coming from an untrusted source (the LLM can
# hallucinate arguments, and a prompt-injection in tool *results* could try to
# alter the next call).  validate_args() is therefore called at the very top of
# each tool before we touch any data.

_IATA_RE = re.compile(r"^[A-Z]{3}$")   # 3-letter airport code
_POSITIVE_INT_RE = re.compile(r"^\d+$")

ALLOWED_OPERATIONS = {"add", "subtract", "multiply", "divide", "sum"}


def _require_str(value: Any, name: str, max_len: int = 200) -> str | None:
    """Return cleaned string or None if invalid."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if len(cleaned) == 0 or len(cleaned) > max_len:
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Tool 1 — search_flights
# ---------------------------------------------------------------------------

def search_flights(origin: str, destination: str, max_price_eur: float | None = None) -> dict:
    """
    Search available flights between two airports.

    Args:
        origin:         IATA code for the departure airport (e.g. "LHR").
        destination:    IATA code for the arrival airport   (e.g. "OPO").
        max_price_eur:  Optional upper price cap in euros.

    Returns:
        { ok: True, flights: [...] }  or  { ok: False, error: "..." }
    """
    # ── Argument validation (safety mitigation) ────────────────────────────
    origin = _require_str(origin, "origin", 3) or ""
    destination = _require_str(destination, "destination", 3) or ""

    if not _IATA_RE.match(origin):
        return _tool_error(
            f"'origin' must be a 3-letter IATA airport code (got '{origin}')."
        )
    if not _IATA_RE.match(destination):
        return _tool_error(
            f"'destination' must be a 3-letter IATA airport code (got '{destination}')."
        )
    if origin == destination:
        return _tool_error("'origin' and 'destination' cannot be the same airport.")

    if max_price_eur is not None:
        try:
            max_price_eur = float(max_price_eur)
        except (TypeError, ValueError):
            return _tool_error("'max_price_eur' must be a number.")
        if max_price_eur <= 0 or max_price_eur > 10_000:
            return _tool_error(
                f"'max_price_eur' must be between 0 and 10 000 (got {max_price_eur})."
            )
    # ── Core logic ─────────────────────────────────────────────────────────
    try:
        data = _load_json("flights.json")
    except FileNotFoundError as exc:
        return _tool_error(str(exc))

    results = [
        f for f in data["routes"]
        if f["origin"] == origin and f["destination"] == destination
    ]

    if max_price_eur is not None:
        results = [f for f in results if f["price_eur"] <= max_price_eur]

    # Sort cheapest first so the agent can trivially pick the best option.
    results.sort(key=lambda f: f["price_eur"])

    return _tool_ok({"flights": results, "count": len(results)})


# ---------------------------------------------------------------------------
# Tool 2 — search_hotels
# ---------------------------------------------------------------------------

def search_hotels(
    city: str,
    nights: int,
    max_price_per_night_eur: float | None = None,
    min_stars: int | None = None,
) -> dict:
    """
    Search available hotels in a city.

    Args:
        city:                      City name (e.g. "Porto").
        nights:                    Number of nights (1-30).
        max_price_per_night_eur:   Optional per-night price cap.
        min_stars:                 Optional minimum star rating (1-5).

    Returns:
        { ok: True, hotels: [...], nights: N }  or  { ok: False, error: "..." }
    """
    # ── Argument validation ────────────────────────────────────────────────
    city_clean = _require_str(city, "city", 100)
    if city_clean is None:
        return _tool_error("'city' must be a non-empty string (max 100 chars).")

    try:
        nights = int(nights)
    except (TypeError, ValueError):
        return _tool_error("'nights' must be an integer.")
    if not (1 <= nights <= 30):
        return _tool_error(f"'nights' must be between 1 and 30 (got {nights}).")

    if max_price_per_night_eur is not None:
        try:
            max_price_per_night_eur = float(max_price_per_night_eur)
        except (TypeError, ValueError):
            return _tool_error("'max_price_per_night_eur' must be a number.")
        if max_price_per_night_eur <= 0 or max_price_per_night_eur > 5_000:
            return _tool_error(
                "'max_price_per_night_eur' must be between 0 and 5 000."
            )

    if min_stars is not None:
        try:
            min_stars = int(min_stars)
        except (TypeError, ValueError):
            return _tool_error("'min_stars' must be an integer.")
        if not (1 <= min_stars <= 5):
            return _tool_error(f"'min_stars' must be between 1 and 5 (got {min_stars}).")

    # ── Core logic ─────────────────────────────────────────────────────────
    try:
        data = _load_json("hotels.json")
    except FileNotFoundError as exc:
        return _tool_error(str(exc))

    results = [
        h for h in data["hotels"]
        if h["city"].lower() == city_clean.lower()
    ]

    if max_price_per_night_eur is not None:
        results = [h for h in results if h["price_per_night_eur"] <= max_price_per_night_eur]

    if min_stars is not None:
        results = [h for h in results if h["stars"] >= min_stars]

    # Sort best-value first (rating / price ratio)
    results.sort(key=lambda h: h["rating"] / h["price_per_night_eur"], reverse=True)

    # Annotate each hotel with the total cost for the requested stay.
    for h in results:
        h["total_for_stay_eur"] = round(h["price_per_night_eur"] * nights, 2)

    return _tool_ok({"hotels": results, "count": len(results), "nights": nights})


# ---------------------------------------------------------------------------
# Tool 3 — calculate
# ---------------------------------------------------------------------------

def calculate(operation: str, operands: list[float]) -> dict:
    """
    Perform a simple arithmetic calculation.

    Args:
        operation:  One of: add, subtract, multiply, divide, sum.
        operands:   List of numbers to operate on.

    Returns:
        { ok: True, result: <number> }  or  { ok: False, error: "..." }
    """
    # ── Argument validation ────────────────────────────────────────────────
    op = _require_str(operation, "operation", 20)
    if op is None:
        return _tool_error("'operation' must be a non-empty string.")

    op = op.lower()
    if op not in ALLOWED_OPERATIONS:
        return _tool_error(
            f"'operation' must be one of {sorted(ALLOWED_OPERATIONS)} (got '{op}')."
        )

    if not isinstance(operands, list):
        return _tool_error("'operands' must be a JSON array of numbers.")
    if len(operands) == 0:
        return _tool_error("'operands' must not be empty.")
    if len(operands) > 100:
        return _tool_error("'operands' must have at most 100 elements.")

    cleaned: list[float] = []
    for i, v in enumerate(operands):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return _tool_error(f"operands[{i}] is not a number: {v!r}")
        if not math.isfinite(fv):
            return _tool_error(f"operands[{i}] must be a finite number (got {fv}).")
        if abs(fv) > 1e12:
            return _tool_error(
                f"operands[{i}] exceeds the allowed range ±1 000 000 000 000."
            )
        cleaned.append(fv)

    # Binary-operation arity check
    if op in {"subtract", "multiply", "divide"} and len(cleaned) != 2:
        return _tool_error(
            f"'{op}' requires exactly 2 operands (got {len(cleaned)})."
        )

    # ── Core logic ─────────────────────────────────────────────────────────
    try:
        if op == "add":
            result = cleaned[0] + cleaned[1]
        elif op == "subtract":
            result = cleaned[0] - cleaned[1]
        elif op == "multiply":
            result = cleaned[0] * cleaned[1]
        elif op == "divide":
            if cleaned[1] == 0:
                return _tool_error("Division by zero is not allowed.")
            result = cleaned[0] / cleaned[1]
        elif op == "sum":
            result = sum(cleaned)
        else:
            return _tool_error(f"Unknown operation '{op}'.")  # unreachable
    except Exception as exc:  # pragma: no cover
        return _tool_error(f"Arithmetic error: {exc}")

    return _tool_ok({"result": round(result, 4), "operation": op, "operands": cleaned})


# ---------------------------------------------------------------------------
# Tool registry — used by the agent loop
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "search_flights": search_flights,
    "search_hotels":  search_hotels,
    "calculate":      calculate,
}
