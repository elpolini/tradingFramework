from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import random
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from engine import (
    FROZEN_CONFIG,
    CoreConfig,
    build_target_weights,
    latest_signal,
    performance_metrics,
    strategy_returns,
)

DATA_COMMIT = "646927e67424160b328658cd43c8a135e639374e"
DATA_ROOT = f"https://raw.githubusercontent.com/yanniedog/binance-historical-OHLCV-data/{DATA_COMMIT}"
DATA_URLS = {
    "BTC": f"{DATA_ROOT}/BTCUSDT_1w.csv",
    "ETH": f"{DATA_ROOT}/ETHUSDT_1w.csv",
}
SEED = 20260824

# Gates are frozen before observing this candidate's validation result.
GATE_THRESHOLDS = {
    "full_sharpe_min": 0.80,
    "full_max_dd_floor": -0.50,
    "max_dd_relative_improvement_vs_btc_min": 0.25,
    "sharpe_vs_btc_min_ratio": 1.00,
    "cagr_vs_btc_min_ratio": 0.50,
    "validation_2022_2023_cagr_min": 0.00,
    "validation_2022_2023_max_dd_floor": -0.35,
    "holdout_2024_plus_cagr_min": 0.00,
    "holdout_2024_plus_sharpe_min": 0.50,
    "holdout_2024_plus_max_dd_floor": -0.35,
    "stress_100bps_cagr_min": 0.00,
    "stress_100bps_max_dd_floor": -0.55,
    "positive_full_year_ratio_min": 0.70,
    "sensitivity_positive_cagr_ratio_min": 0.80,
    "sensitivity_sharpe_ge_06_ratio_min": 0.75,
    "sensitivity_median_max_dd_floor": -0.45,
    "bootstrap_positive_cagr_probability_min": 0.90,
    "bootstrap_max_dd_above_minus50_probability_min": 0.95,
}


def _download(url: str) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "gccc-core-prod-validator/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
    return raw, hashlib.sha256(raw).hexdigest()


def load_prices() -> Tuple[pd.DataFrame, Dict[str, str]]:
    series = {}
    hashes = {}
    for symbol, url in DATA_URLS.items():
        raw, digest = _download(url)
        hashes[symbol] = digest
        df = pd.read_csv(io.BytesIO(raw), parse_dates=["timestamp"])
        if "close" not in df.columns:
            raise ValueError(f"{symbol}: close missing")
        s = df.set_index("timestamp")["close"].astype(float).rename(symbol)
        series[symbol] = s
    prices = pd.concat([series["BTC"], series["ETH"]], axis=1, join="inner").dropna()
    return prices, hashes


def benchmark_btc(prices: pd.DataFrame, strategy_index: pd.DatetimeIndex, cost_bps: float) -> pd.Series:
    r = prices["BTC"].pct_change().reindex(strategy_index).fillna(0.0).copy()
    if len(r):
        r.iloc[0] -= cost_bps / 10_000.0
    return r


def period_metrics(r: pd.Series, start: str | None = None, end: str | None = None) -> Dict[str, float]:
    x = r
    if start is not None:
        x = x.loc[pd.Timestamp(start) :]
    if end is not None:
        x = x.loc[: pd.Timestamp(end)]
    return performance_metrics(x)


def annual_returns(r: pd.Series) -> Dict[int, float]:
    out = {}
    for year, block in r.groupby(r.index.year):
        # Only count full-ish calendar years in the stability gate.
        if len(block) >= 45:
            out[int(year)] = float((1.0 + block).prod() - 1.0)
    return out


def causality_test(prices: pd.DataFrame, config: CoreConfig) -> bool:
    cutoff_pos = int(len(prices) * 0.70)
    cutoff = prices.index[cutoff_pos]
    original = build_target_weights(prices, config)
    mutated = prices.copy()
    future_mask = mutated.index > cutoff
    mutated.loc[future_mask, "BTC"] *= 7.0
    mutated.loc[future_mask, "ETH"] *= 0.11
    changed = build_target_weights(mutated, config)
    a = original.loc[:cutoff].to_numpy(dtype=float)
    b = changed.loc[:cutoff].to_numpy(dtype=float)
    return bool(np.allclose(a, b, atol=0.0, rtol=0.0))


def allocation_invariants(weights: pd.DataFrame) -> Dict[str, object]:
    arr = weights.to_numpy(dtype=float)
    sums = weights.sum(axis=1).to_numpy(dtype=float)
    return {
        "finite": bool(np.isfinite(arr).all()),
        "non_negative": bool((arr >= -1e-12).all()),
        "gross_le_one": bool((sums <= 1.0 + 1e-12).all()),
        "cash_non_negative": bool(((1.0 - sums) >= -1e-12).all()),
        "max_gross": float(np.max(sums)) if len(sums) else 0.0,
        "min_cash": float(np.min(1.0 - sums)) if len(sums) else 1.0,
    }


def sensitivity_grid(prices: pd.DataFrame) -> Dict[str, object]:
    rows = []
    for trend, mom, volw, target in itertools.product(
        [32, 40, 48], [10, 13, 16], [10, 12, 16], [0.20, 0.25, 0.30]
    ):
        cfg = CoreConfig(
            trend_weeks=trend,
            momentum_weeks=mom,
            vol_weeks=volw,
            vol_target=target,
            cost_bps=25.0,
        )
        r, _, _ = strategy_returns(prices, cfg, cost_bps=25.0)
        m = performance_metrics(r)
        rows.append(
            {
                "trend_weeks": trend,
                "momentum_weeks": mom,
                "vol_weeks": volw,
                "vol_target": target,
                **m,
            }
        )
    df = pd.DataFrame(rows)
    return {
        "n": int(len(df)),
        "positive_cagr_ratio": float((df["cagr"] > 0.0).mean()),
        "sharpe_ge_06_ratio": float((df["sharpe"] >= 0.60).mean()),
        "median_cagr": float(df["cagr"].median()),
        "median_sharpe": float(df["sharpe"].median()),
        "median_max_dd": float(df["max_dd"].median()),
        "worst_cagr": float(df["cagr"].min()),
        "worst_sharpe": float(df["sharpe"].min()),
        "worst_max_dd": float(df["max_dd"].min()),
        "best_cagr": float(df["cagr"].max()),
        "best_sharpe": float(df["sharpe"].max()),
    }


def circular_block_bootstrap(
    returns: pd.Series,
    n_sims: int = 5000,
    block: int = 13,
    seed: int = SEED,
) -> Dict[str, float]:
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    cagr = np.empty(n_sims)
    maxdd = np.empty(n_sims)
    starts_needed = math.ceil(n / block)
    for s in range(n_sims):
        starts = rng.integers(0, n, size=starts_needed)
        idx = np.concatenate([(np.arange(st, st + block) % n) for st in starts])[:n]
        sample = x[idx]
        wealth = np.cumprod(1.0 + sample)
        final = wealth[-1]
        cagr[s] = final ** (52.0 / n) - 1.0 if final > 0 else -1.0
        running_max = np.maximum.accumulate(wealth)
        maxdd[s] = np.min(wealth / running_max - 1.0)
    return {
        "n_sims": int(n_sims),
        "block_weeks": int(block),
        "p_cagr_gt_0": float(np.mean(cagr > 0.0)),
        "p_max_dd_gt_minus_50": float(np.mean(maxdd > -0.50)),
        "cagr_p05": float(np.quantile(cagr, 0.05)),
        "cagr_median": float(np.median(cagr)),
        "cagr_p95": float(np.quantile(cagr, 0.95)),
        "max_dd_p05": float(np.quantile(maxdd, 0.05)),
        "max_dd_median": float(np.median(maxdd)),
    }


def gate(name: str, condition: bool, value: object, threshold: object) -> Dict[str, object]:
    return {"name": name, "pass": bool(condition), "value": value, "threshold": threshold}


def fmt_pct(x: float) -> str:
    return f"{100*x:.2f}%"


def main() -> int:
    prices, hashes = load_prices()
    config = FROZEN_CONFIG

    # Data integrity is itself a hard production gate.
    weekly_diffs = prices.index.to_series().diff().dropna().dt.days
    data_integrity = {
        "rows": int(len(prices)),
        "start": prices.index.min().date().isoformat(),
        "end": prices.index.max().date().isoformat(),
        "weekly_cadence": bool((weekly_diffs == 7).all()),
        "duplicates": int(prices.index.duplicated().sum()),
        "na": int(prices.isna().sum().sum()),
        "positive_prices": bool((prices > 0).all().all()),
        "sha256": hashes,
        "data_commit": DATA_COMMIT,
    }

    r25, weights, turnover = strategy_returns(prices, config, cost_bps=25.0)
    full = performance_metrics(r25)
    btc_r = benchmark_btc(prices, r25.index, cost_bps=25.0)
    btc = performance_metrics(btc_r)
    dev = period_metrics(r25, end="2021-12-31")
    validation = period_metrics(r25, start="2022-01-01", end="2023-12-31")
    holdout = period_metrics(r25, start="2024-01-01")

    costs = {}
    for bps in [25.0, 50.0, 100.0]:
        rr, _, tt = strategy_returns(prices, config, cost_bps=bps)
        costs[str(int(bps))] = {
            **performance_metrics(rr),
            "annual_turnover": float(tt.mean() * 52.0),
        }

    years = annual_returns(r25)
    positive_year_ratio = float(np.mean(np.array(list(years.values())) > 0.0)) if years else 0.0
    sensitivity = sensitivity_grid(prices)
    bootstrap = circular_block_bootstrap(r25)
    causal = causality_test(prices, config)
    invariants = allocation_invariants(weights)

    btc_dd = abs(float(btc["max_dd"]))
    dd_improvement = 1.0 - abs(float(full["max_dd"])) / btc_dd if btc_dd > 0 else 0.0
    sharpe_ratio_vs_btc = float(full["sharpe"] / btc["sharpe"]) if btc["sharpe"] != 0 else float("inf")
    cagr_ratio_vs_btc = float(full["cagr"] / btc["cagr"]) if btc["cagr"] > 0 else float("inf")

    t = GATE_THRESHOLDS
    gates = [
        gate(
            "data_integrity",
            data_integrity["weekly_cadence"]
            and data_integrity["duplicates"] == 0
            and data_integrity["na"] == 0
            and data_integrity["positive_prices"],
            data_integrity,
            "weekly/no-duplicate/no-NA/positive",
        ),
        gate("causality_future_mutation", causal, causal, True),
        gate(
            "allocation_invariants",
            invariants["finite"]
            and invariants["non_negative"]
            and invariants["gross_le_one"]
            and invariants["cash_non_negative"],
            invariants,
            "0<=weights; gross<=1; cash>=0",
        ),
        gate("full_cagr_positive", full["cagr"] > 0.0, full["cagr"], ">0"),
        gate("full_sharpe", full["sharpe"] >= t["full_sharpe_min"], full["sharpe"], t["full_sharpe_min"]),
        gate("full_max_dd", full["max_dd"] >= t["full_max_dd_floor"], full["max_dd"], t["full_max_dd_floor"]),
        gate(
            "dd_improvement_vs_btc",
            dd_improvement >= t["max_dd_relative_improvement_vs_btc_min"],
            dd_improvement,
            t["max_dd_relative_improvement_vs_btc_min"],
        ),
        gate(
            "sharpe_vs_btc",
            sharpe_ratio_vs_btc >= t["sharpe_vs_btc_min_ratio"],
            sharpe_ratio_vs_btc,
            t["sharpe_vs_btc_min_ratio"],
        ),
        gate(
            "cagr_retention_vs_btc",
            cagr_ratio_vs_btc >= t["cagr_vs_btc_min_ratio"],
            cagr_ratio_vs_btc,
            t["cagr_vs_btc_min_ratio"],
        ),
        gate(
            "validation_2022_2023_cagr",
            validation["cagr"] > t["validation_2022_2023_cagr_min"],
            validation["cagr"],
            f">{t['validation_2022_2023_cagr_min']}",
        ),
        gate(
            "validation_2022_2023_max_dd",
            validation["max_dd"] >= t["validation_2022_2023_max_dd_floor"],
            validation["max_dd"],
            t["validation_2022_2023_max_dd_floor"],
        ),
        gate(
            "holdout_2024_plus_cagr",
            holdout["cagr"] > t["holdout_2024_plus_cagr_min"],
            holdout["cagr"],
            f">{t['holdout_2024_plus_cagr_min']}",
        ),
        gate(
            "holdout_2024_plus_sharpe",
            holdout["sharpe"] >= t["holdout_2024_plus_sharpe_min"],
            holdout["sharpe"],
            t["holdout_2024_plus_sharpe_min"],
        ),
        gate(
            "holdout_2024_plus_max_dd",
            holdout["max_dd"] >= t["holdout_2024_plus_max_dd_floor"],
            holdout["max_dd"],
            t["holdout_2024_plus_max_dd_floor"],
        ),
        gate(
            "stress_100bps_cagr",
            costs["100"]["cagr"] > t["stress_100bps_cagr_min"],
            costs["100"]["cagr"],
            f">{t['stress_100bps_cagr_min']}",
        ),
        gate(
            "stress_100bps_max_dd",
            costs["100"]["max_dd"] >= t["stress_100bps_max_dd_floor"],
            costs["100"]["max_dd"],
            t["stress_100bps_max_dd_floor"],
        ),
        gate(
            "annual_stability",
            positive_year_ratio >= t["positive_full_year_ratio_min"],
            positive_year_ratio,
            t["positive_full_year_ratio_min"],
        ),
        gate(
            "sensitivity_positive_cagr",
            sensitivity["positive_cagr_ratio"] >= t["sensitivity_positive_cagr_ratio_min"],
            sensitivity["positive_cagr_ratio"],
            t["sensitivity_positive_cagr_ratio_min"],
        ),
        gate(
            "sensitivity_sharpe_plateau",
            sensitivity["sharpe_ge_06_ratio"] >= t["sensitivity_sharpe_ge_06_ratio_min"],
            sensitivity["sharpe_ge_06_ratio"],
            t["sensitivity_sharpe_ge_06_ratio_min"],
        ),
        gate(
            "sensitivity_median_drawdown",
            sensitivity["median_max_dd"] >= t["sensitivity_median_max_dd_floor"],
            sensitivity["median_max_dd"],
            t["sensitivity_median_max_dd_floor"],
        ),
        gate(
            "bootstrap_positive_cagr",
            bootstrap["p_cagr_gt_0"] >= t["bootstrap_positive_cagr_probability_min"],
            bootstrap["p_cagr_gt_0"],
            t["bootstrap_positive_cagr_probability_min"],
        ),
        gate(
            "bootstrap_drawdown",
            bootstrap["p_max_dd_gt_minus_50"] >= t["bootstrap_max_dd_above_minus50_probability_min"],
            bootstrap["p_max_dd_gt_minus_50"],
            t["bootstrap_max_dd_above_minus50_probability_min"],
        ),
    ]

    failed = [g["name"] for g in gates if not g["pass"]]
    score = round(100.0 * sum(g["pass"] for g in gates) / len(gates), 1)
    prod_eligible = len(failed) == 0 and score >= 85.0

    results = {
        "model": "GCCC CORE-PROD v1.0",
        "status": "PROD_VALIDATED" if prod_eligible else "RESEARCH_ONLY",
        "validation_score_out_of_100": score,
        "important": "The score is a checklist score, NOT a probability of future profitability or reliability.",
        "frozen_config": config.to_dict(),
        "gate_thresholds": GATE_THRESHOLDS,
        "data": data_integrity,
        "metrics": {
            "full": full,
            "btc_buy_hold": btc,
            "development_through_2021": dev,
            "validation_2022_2023": validation,
            "holdout_2024_plus": holdout,
            "max_dd_relative_improvement_vs_btc": dd_improvement,
            "sharpe_ratio_vs_btc": sharpe_ratio_vs_btc,
            "cagr_ratio_vs_btc": cagr_ratio_vs_btc,
            "mean_gross_exposure": float(weights.sum(axis=1).mean()),
            "annual_turnover": float(turnover.mean() * 52.0),
        },
        "cost_stress": costs,
        "annual_returns": {str(k): v for k, v in years.items()},
        "positive_full_year_ratio": positive_year_ratio,
        "sensitivity_81_neighbours": sensitivity,
        "bootstrap_5000x_13w": bootstrap,
        "causality_test": causal,
        "allocation_invariants": invariants,
        "gates": gates,
        "failed_gates": failed,
        "latest_static_signal": latest_signal(prices, config),
    }

    out = Path(__file__).resolve().parent
    (out / "validation_results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    gate_lines = "\n".join(
        f"- [{'x' if g['pass'] else ' '}] {g['name']}: value={g['value']} | threshold={g['threshold']}"
        for g in gates
    )
    report = f"""# GCCC CORE-PROD v1.0 — Validation Report

**Status:** {results['status']}  
**Validation checklist score:** {score}/100  
**Data:** {data_integrity['start']} → {data_integrity['end']} ({data_integrity['rows']} weekly rows)  
**Pinned source commit:** `{DATA_COMMIT}`

> The score is a validation-checklist score. It is **not** a 100% reliability claim and is not a probability of future profit.

## Frozen specification

```json
{json.dumps(config.to_dict(), indent=2)}
```

Signal: BTC master gate = weekly close > SMA40w AND 13w momentum > 0. ETH participates only if BTC master gate is on and ETH passes the same filters. Active BTC/ETH are inverse-vol weighted using trailing 12w volatility, portfolio exposure is capped to a 25% annualized ex-ante vol target, gross exposure is capped at 1.0, residual stays in cash, and every signal is executed with one full weekly lag.

## Core results (25 bps one-way costs)

| Slice | CAGR | Vol | Sharpe | MaxDD | Calmar |
|---|---:|---:|---:|---:|---:|
| Full | {fmt_pct(full['cagr'])} | {fmt_pct(full['vol'])} | {full['sharpe']:.3f} | {fmt_pct(full['max_dd'])} | {full['calmar']:.3f} |
| BTC buy & hold | {fmt_pct(btc['cagr'])} | {fmt_pct(btc['vol'])} | {btc['sharpe']:.3f} | {fmt_pct(btc['max_dd'])} | {btc['calmar']:.3f} |
| Validation 2022–2023 | {fmt_pct(validation['cagr'])} | {fmt_pct(validation['vol'])} | {validation['sharpe']:.3f} | {fmt_pct(validation['max_dd'])} | {validation['calmar']:.3f} |
| Holdout 2024+ | {fmt_pct(holdout['cagr'])} | {fmt_pct(holdout['vol'])} | {holdout['sharpe']:.3f} | {fmt_pct(holdout['max_dd'])} | {holdout['calmar']:.3f} |

## Cost stress

| One-way cost | CAGR | Sharpe | MaxDD | Annual turnover |
|---|---:|---:|---:|---:|
| 25 bps | {fmt_pct(costs['25']['cagr'])} | {costs['25']['sharpe']:.3f} | {fmt_pct(costs['25']['max_dd'])} | {costs['25']['annual_turnover']:.2f}x |
| 50 bps | {fmt_pct(costs['50']['cagr'])} | {costs['50']['sharpe']:.3f} | {fmt_pct(costs['50']['max_dd'])} | {costs['50']['annual_turnover']:.2f}x |
| 100 bps | {fmt_pct(costs['100']['cagr'])} | {costs['100']['sharpe']:.3f} | {fmt_pct(costs['100']['max_dd'])} | {costs['100']['annual_turnover']:.2f}x |

## Robustness

- Full-year positive ratio: {positive_year_ratio:.1%}
- 81-neighbour sensitivity positive-CAGR ratio: {sensitivity['positive_cagr_ratio']:.1%}
- 81-neighbour sensitivity Sharpe>=0.60 ratio: {sensitivity['sharpe_ge_06_ratio']:.1%}
- Sensitivity median Sharpe: {sensitivity['median_sharpe']:.3f}
- Sensitivity median MaxDD: {fmt_pct(sensitivity['median_max_dd'])}
- Bootstrap P(CAGR>0), historical block recombination only: {bootstrap['p_cagr_gt_0']:.1%}
- Bootstrap P(MaxDD>-50%), historical block recombination only: {bootstrap['p_max_dd_gt_minus_50']:.1%}
- Bootstrap CAGR P5 / median / P95: {fmt_pct(bootstrap['cagr_p05'])} / {fmt_pct(bootstrap['cagr_median'])} / {fmt_pct(bootstrap['cagr_p95'])}
- Bootstrap adverse MaxDD P5: {fmt_pct(bootstrap['max_dd_p05'])}

## Hard gates

{gate_lines}

## Decision

`{'PROMOTE_TO_PRODUCTION' if prod_eligible else 'DO_NOT_PROMOTE'}`

The ranking of altcoins, Cycle Clock timing, Phase5D macro overlay, Phase6A heat overlay and Phase6B derivatives are **not part of this production engine**. They remain research/challenger components until independently validated.
"""
    (out / "validation_report.md").write_text(report, encoding="utf-8")
    print(report)
    print("RESULT_JSON=" + json.dumps(results, sort_keys=True))
    return 0 if prod_eligible else 1


if __name__ == "__main__":
    sys.exit(main())
