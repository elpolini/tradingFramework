"""Strict validation entrypoint.

The pinned static mirror contains a terminal weekly row that was captured before
that week completed. Production validation must never treat a partial weekly
bar as a completed close. We therefore drop exactly the terminal row from the
static snapshot and verify the new terminal row against a Binance-direct
cross-check observed before this fix was committed.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import validate as base

EXPECTED_LAST_COMPLETE_WEEK = pd.Timestamp("2025-12-29")
EXPECTED_LAST_COMPLETE_CLOSE = {"BTC": 91529.73, "ETH": 3144.70}

_original_load_prices = base.load_prices


def strict_load_prices():
    prices, hashes = _original_load_prices()
    if len(prices) < 2:
        raise ValueError("insufficient weekly history")

    dropped_date = prices.index[-1]
    prices = prices.iloc[:-1].copy()

    if prices.index[-1] != EXPECTED_LAST_COMPLETE_WEEK:
        raise ValueError(
            f"unexpected last completed mirror week: {prices.index[-1]} != {EXPECTED_LAST_COMPLETE_WEEK}"
        )
    for symbol, expected in EXPECTED_LAST_COMPLETE_CLOSE.items():
        actual = float(prices[symbol].iloc[-1])
        if not np.isclose(actual, expected, atol=1e-8, rtol=0.0):
            raise ValueError(f"{symbol} cross-check failed: {actual} != {expected}")

    hashes["terminal_bar_policy"] = "drop_last_static_row_as_potentially_incomplete"
    hashes["terminal_bar_dropped"] = dropped_date.date().isoformat()
    hashes["binance_direct_crosscheck_week"] = EXPECTED_LAST_COMPLETE_WEEK.date().isoformat()
    return prices, hashes


base.load_prices = strict_load_prices

if __name__ == "__main__":
    sys.exit(base.main())
