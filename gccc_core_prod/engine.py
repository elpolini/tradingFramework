from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoreConfig:
    """Frozen CORE-PROD v1.0 parameters.

    All decisions are made from completed weekly closes and become effective
    one week later. No parameter is fitted by the engine.
    """

    trend_weeks: int = 40
    momentum_weeks: int = 13
    vol_weeks: int = 12
    vol_target: float = 0.25
    cost_bps: float = 25.0
    annualization: int = 52

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


FROZEN_CONFIG = CoreConfig()


def validate_price_frame(prices: pd.DataFrame) -> None:
    required = ["BTC", "ETH"]
    if list(prices.columns) != required:
        raise ValueError(f"prices columns must be exactly {required}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("duplicate timestamps")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("timestamps must be increasing")
    if prices.isna().any().any():
        raise ValueError("price frame contains NA")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    diffs = prices.index.to_series().diff().dropna().dt.days
    if not (diffs == 7).all():
        bad = diffs[diffs != 7]
        raise ValueError(f"weekly cadence broken at {bad.index[:5].tolist()}")


def _portfolio_vol(
    returns_window: pd.DataFrame,
    base_weights: np.ndarray,
    annualization: int,
) -> float:
    cov = returns_window.cov().to_numpy(dtype=float) * annualization
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return float("nan")
    variance = float(base_weights @ cov @ base_weights)
    return sqrt(max(variance, 0.0))


def build_target_weights(
    prices: pd.DataFrame,
    config: CoreConfig = FROZEN_CONFIG,
) -> pd.DataFrame:
    """Build signal-date target weights.

    The row at t uses information through the completed close at t only.
    Execution must use ``effective_weights`` which applies a one-week lag.
    """

    validate_price_frame(prices)
    ret = prices.pct_change()
    sma = prices.rolling(config.trend_weeks, min_periods=config.trend_weeks).mean()
    mom = prices / prices.shift(config.momentum_weeks) - 1.0
    ann_vol = ret.rolling(config.vol_weeks, min_periods=config.vol_weeks).std(ddof=1) * sqrt(
        config.annualization
    )

    btc_gate = (prices["BTC"] > sma["BTC"]) & (mom["BTC"] > 0.0)
    eth_gate = btc_gate & (prices["ETH"] > sma["ETH"]) & (mom["ETH"] > 0.0)

    out = pd.DataFrame(0.0, index=prices.index, columns=["BTC", "ETH"])
    warmup = max(config.trend_weeks - 1, config.momentum_weeks, config.vol_weeks)

    for i in range(warmup, len(prices)):
        if not bool(btc_gate.iloc[i]):
            continue

        btc_vol = float(ann_vol["BTC"].iloc[i])
        if not np.isfinite(btc_vol) or btc_vol <= 0.0:
            continue

        if bool(eth_gate.iloc[i]):
            eth_vol = float(ann_vol["ETH"].iloc[i])
            if not np.isfinite(eth_vol) or eth_vol <= 0.0:
                continue
            inv = np.array([1.0 / btc_vol, 1.0 / eth_vol], dtype=float)
            base = inv / inv.sum()
        else:
            base = np.array([1.0, 0.0], dtype=float)

        window = ret.iloc[i - config.vol_weeks + 1 : i + 1][["BTC", "ETH"]]
        pvol = _portfolio_vol(window, base, config.annualization)
        if not np.isfinite(pvol) or pvol <= 0.0:
            continue
        scale = min(1.0, config.vol_target / pvol)
        out.iloc[i] = base * scale

    return out


def effective_weights(target_weights: pd.DataFrame) -> pd.DataFrame:
    """One full weekly bar of execution lag: signal t -> position t+1."""

    return target_weights.shift(1).fillna(0.0)


def allocation_with_cash(risky_weights: pd.DataFrame) -> pd.DataFrame:
    alloc = risky_weights.copy()
    alloc["CASH"] = 1.0 - alloc.sum(axis=1)
    return alloc[["BTC", "ETH", "CASH"]]


def turnover_one_way(risky_weights: pd.DataFrame) -> pd.Series:
    """One-way notional turnover including the residual cash sleeve."""

    alloc = allocation_with_cash(risky_weights)
    delta = alloc.diff().abs()
    turnover = 0.5 * delta.sum(axis=1)
    if len(turnover):
        initial = np.array([0.0, 0.0, 1.0])
        turnover.iloc[0] = 0.5 * np.abs(alloc.iloc[0].to_numpy(dtype=float) - initial).sum()
    return turnover.fillna(0.0)


def strategy_returns(
    prices: pd.DataFrame,
    config: CoreConfig = FROZEN_CONFIG,
    cost_bps: float | None = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Net weekly strategy returns, effective weights, and one-way turnover."""

    target = build_target_weights(prices, config)
    weights = effective_weights(target)
    asset_ret = prices.pct_change().fillna(0.0)
    gross = (weights * asset_ret[["BTC", "ETH"]]).sum(axis=1)
    turnover = turnover_one_way(weights)
    rate = (config.cost_bps if cost_bps is None else cost_bps) / 10_000.0
    net = gross - turnover * rate

    # First tradable row occurs only after all slow features are observable and
    # after the explicit one-week execution lag.
    start = max(config.trend_weeks, config.momentum_weeks + 1, config.vol_weeks + 1)
    return net.iloc[start:], weights.iloc[start:], turnover.iloc[start:]


def performance_metrics(weekly_returns: pd.Series, annualization: int = 52) -> Dict[str, float]:
    r = weekly_returns.dropna().astype(float)
    if len(r) < 2:
        return {
            "weeks": float(len(r)),
            "cagr": float("nan"),
            "vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "calmar": float("nan"),
        }
    wealth = (1.0 + r).cumprod()
    years = len(r) / annualization
    final = float(wealth.iloc[-1])
    cagr = final ** (1.0 / years) - 1.0 if final > 0.0 else -1.0
    vol = float(r.std(ddof=1) * sqrt(annualization))
    sharpe = float(r.mean() / r.std(ddof=1) * sqrt(annualization)) if r.std(ddof=1) > 0 else 0.0
    drawdown = wealth / wealth.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0.0 else float("inf")
    return {
        "weeks": float(len(r)),
        "cagr": float(cagr),
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
    }


def latest_signal(prices: pd.DataFrame, config: CoreConfig = FROZEN_CONFIG) -> Dict[str, object]:
    target = build_target_weights(prices, config)
    i = len(prices) - 1
    close = prices.iloc[i]
    sma = prices.rolling(config.trend_weeks).mean().iloc[i]
    mom = (prices / prices.shift(config.momentum_weeks) - 1.0).iloc[i]
    tw = target.iloc[i]
    return {
        "as_of": prices.index[i].date().isoformat(),
        "close": {k: float(close[k]) for k in ["BTC", "ETH"]},
        "sma40w": {k: float(sma[k]) for k in ["BTC", "ETH"]},
        "momentum13w": {k: float(mom[k]) for k in ["BTC", "ETH"]},
        "btc_gate": bool(close["BTC"] > sma["BTC"] and mom["BTC"] > 0),
        "eth_gate": bool(
            close["BTC"] > sma["BTC"]
            and mom["BTC"] > 0
            and close["ETH"] > sma["ETH"]
            and mom["ETH"] > 0
        ),
        "target": {
            "BTC": float(tw["BTC"]),
            "ETH": float(tw["ETH"]),
            "CASH": float(1.0 - tw.sum()),
        },
        "effective_next_week": True,
        "config": config.to_dict(),
    }
