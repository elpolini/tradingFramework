from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from engine import FROZEN_CONFIG, build_target_weights, effective_weights, strategy_returns
from run_validation import strict_load_prices


def shadow_targets(prices: pd.DataFrame) -> pd.DataFrame:
    c = FROZEN_CONFIG
    out = pd.DataFrame(0.0, index=prices.index, columns=["BTC", "ETH"])
    ret = prices.pct_change()

    for i in range(len(prices)):
        if i < c.trend_weeks - 1 or i < c.momentum_weeks or i < c.vol_weeks:
            continue

        btc_close = float(prices["BTC"].iloc[i])
        btc_sma = float(prices["BTC"].iloc[i - c.trend_weeks + 1 : i + 1].mean())
        btc_mom = btc_close / float(prices["BTC"].iloc[i - c.momentum_weeks]) - 1.0
        if not (btc_close > btc_sma and btc_mom > 0.0):
            continue

        btc_sigma = float(ret["BTC"].iloc[i - c.vol_weeks + 1 : i + 1].std(ddof=1) * sqrt(c.annualization))
        if not np.isfinite(btc_sigma) or btc_sigma <= 0:
            continue

        eth_close = float(prices["ETH"].iloc[i])
        eth_sma = float(prices["ETH"].iloc[i - c.trend_weeks + 1 : i + 1].mean())
        eth_mom = eth_close / float(prices["ETH"].iloc[i - c.momentum_weeks]) - 1.0
        eth_active = eth_close > eth_sma and eth_mom > 0.0

        if eth_active:
            eth_sigma = float(ret["ETH"].iloc[i - c.vol_weeks + 1 : i + 1].std(ddof=1) * sqrt(c.annualization))
            inv_btc = 1.0 / btc_sigma
            inv_eth = 1.0 / eth_sigma
            denom = inv_btc + inv_eth
            base = np.array([inv_btc / denom, inv_eth / denom], dtype=float)
        else:
            base = np.array([1.0, 0.0], dtype=float)

        window = ret[["BTC", "ETH"]].iloc[i - c.vol_weeks + 1 : i + 1]
        cov = np.cov(window.to_numpy(dtype=float).T, ddof=1) * c.annualization
        variance = float(base @ cov @ base)
        pvol = sqrt(max(variance, 0.0))
        if not np.isfinite(pvol) or pvol <= 0:
            continue
        scale = min(1.0, c.vol_target / pvol)
        out.iloc[i, 0] = base[0] * scale
        out.iloc[i, 1] = base[1] * scale

    return out


def shadow_net_returns(prices: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    c = FROZEN_CONFIG
    w = target.shift(1).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    gross = (w * asset_ret[["BTC", "ETH"]]).sum(axis=1)

    cash = 1.0 - w.sum(axis=1)
    alloc = pd.concat([w, cash.rename("CASH")], axis=1)
    turnover = 0.5 * alloc.diff().abs().sum(axis=1)
    if len(turnover):
        first = alloc.iloc[0].to_numpy(dtype=float)
        turnover.iloc[0] = 0.5 * np.abs(first - np.array([0.0, 0.0, 1.0])).sum()
    turnover = turnover.fillna(0.0)
    net = gross - turnover * (c.cost_bps / 10_000.0)
    start = max(c.trend_weeks, c.momentum_weeks + 1, c.vol_weeks + 1)
    return net.iloc[start:]


def explicit_execution_lag_test(prices: pd.DataFrame) -> bool:
    original_target = build_target_weights(prices, FROZEN_CONFIG)
    original_effective = effective_weights(original_target)
    j = max(80, len(prices) // 2)
    mutated = prices.copy()
    mutated.iloc[j, mutated.columns.get_loc("BTC")] *= 4.0
    mutated.iloc[j, mutated.columns.get_loc("ETH")] *= 0.25
    changed_effective = effective_weights(build_target_weights(mutated, FROZEN_CONFIG))
    # A shock first observed at completed week j must not alter the position
    # already held during week j. It may only change j+1 and later.
    return bool(
        np.allclose(
            original_effective.iloc[: j + 1].to_numpy(dtype=float),
            changed_effective.iloc[: j + 1].to_numpy(dtype=float),
            atol=0.0,
            rtol=0.0,
        )
    )


def main() -> int:
    prices, _ = strict_load_prices()
    prod_target = build_target_weights(prices, FROZEN_CONFIG)
    shadow_target = shadow_targets(prices)
    max_weight_diff = float(np.max(np.abs(prod_target.to_numpy() - shadow_target.to_numpy())))

    prod_r, _, _ = strategy_returns(prices, FROZEN_CONFIG, cost_bps=FROZEN_CONFIG.cost_bps)
    shadow_r = shadow_net_returns(prices, shadow_target)
    max_return_diff = float(np.max(np.abs(prod_r.to_numpy() - shadow_r.to_numpy())))
    lag_ok = explicit_execution_lag_test(prices)

    print(f"max_weight_diff={max_weight_diff:.18g}")
    print(f"max_return_diff={max_return_diff:.18g}")
    print(f"execution_lag_test={lag_ok}")

    if max_weight_diff > 1e-12:
        raise SystemExit("shadow target mismatch")
    if max_return_diff > 1e-12:
        raise SystemExit("shadow return mismatch")
    if not lag_ok:
        raise SystemExit("execution lag invariant failed")
    print("SHADOW_IMPLEMENTATION_MATCH=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
