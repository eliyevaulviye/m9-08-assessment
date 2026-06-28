"""
tests/test_tools.py — Unit tests for all three tools.

Run with:  pytest tests/test_tools.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import search_flights, search_hotels, calculate

# ============================================================
# search_flights
# ============================================================

class TestSearchFlights:

    def test_happy_path(self):
        r = search_flights("LHR", "OPO")
        assert r["ok"] is True
        assert len(r["flights"]) > 0
        # Sorted cheapest first
        prices = [f["price_eur"] for f in r["flights"]]
        assert prices == sorted(prices)

    def test_price_filter(self):
        r = search_flights("LHR", "OPO", max_price_eur=60.0)
        assert r["ok"] is True
        for f in r["flights"]:
            assert f["price_eur"] <= 60.0

    def test_no_results_returns_empty_list(self):
        r = search_flights("JFK", "OPO")
        assert r["ok"] is True
        assert r["flights"] == []

    def test_invalid_origin_rejected(self):
        r = search_flights("london", "OPO")
        assert r["ok"] is False
        assert "IATA" in r["error"]

    def test_same_airport_rejected(self):
        r = search_flights("LHR", "LHR")
        assert r["ok"] is False

    def test_negative_price_rejected(self):
        r = search_flights("LHR", "OPO", max_price_eur=-10)
        assert r["ok"] is False

    def test_sql_injection_attempt(self):
        """Simulate a prompt-injection attempt passing a non-IATA string."""
        r = search_flights("'; DROP TABLE flights; --", "OPO")
        assert r["ok"] is False

    def test_oversized_string_rejected(self):
        r = search_flights("A" * 201, "OPO")
        assert r["ok"] is False


# ============================================================
# search_hotels
# ============================================================

class TestSearchHotels:

    def test_happy_path(self):
        r = search_hotels("Porto", 2)
        assert r["ok"] is True
        assert r["nights"] == 2
        assert len(r["hotels"]) > 0

    def test_total_for_stay_annotated(self):
        r = search_hotels("Porto", 3)
        for h in r["hotels"]:
            assert "total_for_stay_eur" in h
            assert abs(h["total_for_stay_eur"] - h["price_per_night_eur"] * 3) < 0.01

    def test_price_filter(self):
        r = search_hotels("Porto", 2, max_price_per_night_eur=60.0)
        assert r["ok"] is True
        for h in r["hotels"]:
            assert h["price_per_night_eur"] <= 60.0

    def test_min_stars_filter(self):
        r = search_hotels("Porto", 2, min_stars=4)
        assert r["ok"] is True
        for h in r["hotels"]:
            assert h["stars"] >= 4

    def test_unknown_city_returns_empty(self):
        r = search_hotels("Narnia", 2)
        assert r["ok"] is True
        assert r["hotels"] == []

    def test_zero_nights_rejected(self):
        r = search_hotels("Porto", 0)
        assert r["ok"] is False

    def test_too_many_nights_rejected(self):
        r = search_hotels("Porto", 31)
        assert r["ok"] is False

    def test_invalid_city_type_rejected(self):
        r = search_hotels(12345, 2)  # type: ignore
        assert r["ok"] is False

    def test_invalid_stars_rejected(self):
        r = search_hotels("Porto", 2, min_stars=6)
        assert r["ok"] is False


# ============================================================
# calculate
# ============================================================

class TestCalculate:

    def test_add(self):
        r = calculate("add", [100.0, 200.0])
        assert r["ok"] is True
        assert r["result"] == 300.0

    def test_subtract(self):
        r = calculate("subtract", [500.0, 89.99])
        assert r["ok"] is True
        assert abs(r["result"] - 410.01) < 0.001

    def test_multiply(self):
        r = calculate("multiply", [72.0, 2])
        assert r["ok"] is True
        assert r["result"] == 144.0

    def test_divide(self):
        r = calculate("divide", [144.0, 2])
        assert r["ok"] is True
        assert r["result"] == 72.0

    def test_sum(self):
        r = calculate("sum", [54.99, 165.0])
        assert r["ok"] is True
        assert abs(r["result"] - 219.99) < 0.001

    def test_division_by_zero_rejected(self):
        r = calculate("divide", [100.0, 0])
        assert r["ok"] is False
        assert "zero" in r["error"].lower()

    def test_unknown_operation_rejected(self):
        r = calculate("exfiltrate", [1, 2])
        assert r["ok"] is False

    def test_non_numeric_operand_rejected(self):
        r = calculate("add", ["DROP TABLE", 2])
        assert r["ok"] is False

    def test_infinity_rejected(self):
        r = calculate("add", [float("inf"), 1])
        assert r["ok"] is False

    def test_too_large_number_rejected(self):
        r = calculate("add", [2e12, 1])
        assert r["ok"] is False

    def test_empty_operands_rejected(self):
        r = calculate("sum", [])
        assert r["ok"] is False

    def test_wrong_arity_subtract(self):
        r = calculate("subtract", [1, 2, 3])
        assert r["ok"] is False

    def test_wrong_operation_type_rejected(self):
        r = calculate(999, [1, 2])  # type: ignore
        assert r["ok"] is False
