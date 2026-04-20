"""Tests — proves risk gates actually block bad trades."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.risk_manager import (
    RiskConfig, TradeIntent, AccountSnapshot, approve_trade
)


def make_account(**overrides):
    base = dict(
        equity=100_000,
        cash=50_000,
        open_positions=[],
        daily_pnl_pct=0.0,
        trades_today=0,
        new_positions_today=0,
        spy_above_ma200=True,
        spy_above_ma50=True,
        spy_high_vol_breakdown=False,
    )
    base.update(overrides)
    return AccountSnapshot(**base)


def test_happy_path_super_stock():
    intent = TradeIntent("NVDA", "buy", 100.0, 93.0, 100, "super", "VCP breakout")
    d = approve_trade(intent, make_account())
    assert d.approved, d.reasons


def test_rejects_weak_signal():
    intent = TradeIntent("FOO", "buy", 100.0, 93.0, 100, "watchlist", "just watching")
    d = approve_trade(intent, make_account())
    assert not d.approved
    assert "tier" in d.reasons[0].lower()


def test_rejects_stop_too_wide_minervini():
    """Minervini config has 8% max stop."""
    from harness.risk_manager import minervini_config
    intent = TradeIntent("NVDA", "buy", 100.0, 85.0, 10, "super", "15% stop")
    d = approve_trade(intent, make_account(), minervini_config())
    assert not d.approved
    assert "exceeds" in d.reasons[0].lower()


def test_minervini_config_allows_7_percent_stop():
    """A 7% stop should be approved under Minervini config."""
    from harness.risk_manager import minervini_config
    intent = TradeIntent("NVDA", "buy", 100.0, 93.0, 10, "super", "7% stop")
    d = approve_trade(intent, make_account(), minervini_config())
    assert d.approved, d.reasons


def test_cs_momentum_config_allows_15_percent_stop():
    """XS momentum config allows wide synthetic stops."""
    from harness.risk_manager import cs_momentum_config
    # Small position to pass sizing gate: $100k × 2% = $2k risk budget, $15/share risk → 133 max
    intent = TradeIntent("NVDA", "buy", 100.0, 85.0, 100, "leader", "wide stop OK")
    d = approve_trade(intent, make_account(), cs_momentum_config())
    assert d.approved, d.reasons


def test_cs_momentum_config_allows_more_positions():
    """XS momentum allows 15 positions vs Minervini's 5."""
    from harness.risk_manager import cs_momentum_config
    intent = TradeIntent("NEW", "buy", 100.0, 82.0, 50, "leader", "10th position OK")
    acct = make_account(open_positions=[
        {"ticker": t, "shares": 10, "entry": 100} for t in
        ["A", "B", "C", "D", "E", "F", "G", "H", "I"]  # 9 positions
    ])
    d = approve_trade(intent, acct, cs_momentum_config())
    assert d.approved, d.reasons


def test_rejects_averaging_down():
    intent = TradeIntent("NVDA", "buy", 100.0, 93.0, 100, "super", "avg down")
    acct = make_account(open_positions=[{"ticker": "NVDA", "shares": 50, "entry": 110.0}])
    d = approve_trade(intent, acct)
    assert not d.approved
    assert "averaging" in d.reasons[0].lower()


def test_rejects_at_max_positions():
    intent = TradeIntent("NEW", "buy", 100.0, 93.0, 100, "super", "6th position")
    acct = make_account(open_positions=[
        {"ticker": t, "shares": 50, "entry": 100} for t in "ABCDE"
    ])
    d = approve_trade(intent, acct)
    assert not d.approved


def test_daily_loss_circuit_breaker():
    intent = TradeIntent("NVDA", "buy", 100.0, 93.0, 100, "super", "after big loss")
    d = approve_trade(intent, make_account(daily_pnl_pct=-0.025))
    assert not d.approved


def test_spy_breakdown_halts_entries():
    intent = TradeIntent("NVDA", "buy", 100.0, 93.0, 100, "super", "bad regime")
    acct = make_account(spy_above_ma50=False, spy_high_vol_breakdown=True)
    d = approve_trade(intent, acct)
    assert not d.approved


def test_position_size_capped_by_risk():
    # $100k equity × 0.75% = $750 risk budget
    # $7/share risk → max 107 shares
    intent = TradeIntent("NVDA", "buy", 100.0, 93.0, 500, "super", "too big")
    d = approve_trade(intent, make_account())
    assert not d.approved
    assert d.adjusted["shares"] < 500


def test_weak_regime_halves_risk():
    intent = TradeIntent("NVDA", "buy", 100.0, 93.0, 100, "super", "test")
    d = approve_trade(intent, make_account(spy_above_ma200=False))
    # Now budget is $375, max 53 shares
    assert not d.approved
    assert d.adjusted["shares"] < 100


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
    print(f"\nRan {len(tests)} tests")
