"""Regression: zero/None entry_price must not crash grading."""
from __future__ import annotations
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from learning import outcome_grader as og


def test_grade_at_horizon_skips_zero_entry_price():
    decision = SimpleNamespace(
        ticker="AAPL",
        timestamp=(date.today() - timedelta(days=10)).isoformat() + "T15:00:00",
        entry_price=0.0,
        stop_price=None,
        target_price=None,
    )
    pending = [("state/shadow/fake.jsonl", 0, decision)]

    with patch.object(og.sj, "ungraded_decisions", return_value=pending), patch.object(
        og, "_price_on_or_after", return_value=100.0
    ), patch.object(og.sj, "update_decision") as upd:
        result = og.grade_at_horizon(5)

    assert result["graded"] == 0
    assert result["skipped"] >= 1
    upd.assert_not_called()


def test_grade_at_horizon_skips_none_entry_price():
    decision = SimpleNamespace(
        ticker="AAPL",
        timestamp=(date.today() - timedelta(days=10)).isoformat() + "T15:00:00",
        entry_price=None,
        stop_price=None,
        target_price=None,
    )
    pending = [("state/shadow/fake.jsonl", 0, decision)]

    with patch.object(og.sj, "ungraded_decisions", return_value=pending), patch.object(
        og, "_price_on_or_after", return_value=100.0
    ), patch.object(og.sj, "update_decision") as upd:
        result = og.grade_at_horizon(5)

    assert result["graded"] == 0
    assert result["skipped"] >= 1
    upd.assert_not_called()
