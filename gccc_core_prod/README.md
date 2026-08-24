# GCCC CORE-PROD v1.0

**Status: PROD_VALIDATED_INTERNAL — PROMOTION APPROVED 2026-08-24.**

This label means the frozen internal production gates passed. It does **not** mean 100% future reliability, guaranteed profitability, or external/institutional certification.

This production engine is deliberately separated from the experimental Global Crypto Cycle Clock stack. It contains only BTC, ETH and cash. Altcoin ranking, Cycle Clock timing, Phase5D macro, Phase6A heat and Phase6B derivatives are excluded until they pass their own independent gates.

## Frozen rules

- Frequency: completed weekly closes only.
- Master gate: BTC weekly close > SMA40w AND BTC 13-week momentum > 0.
- If the BTC master gate is off: 100% cash.
- ETH can participate only while the BTC master gate is on and ETH independently has close > SMA40w AND 13-week momentum > 0.
- If BTC and ETH are active: inverse trailing-12-week-volatility weights.
- If only BTC is active: BTC receives the risky base sleeve.
- Portfolio ex-ante volatility target: 25% annualized.
- Gross exposure cap: 1.00x. No leverage.
- Residual: cash.
- Execution: one full weekly lag. A signal observed on completed week `t` becomes the position for week `t+1`.
- Baseline execution cost in validation: 25 bps one-way turnover; stress at 50 and 100 bps.
- No optimization or automatic re-tuning in production.

## Frozen validation result

CI run `32736320273` passed the adversarial gates and an independent shadow implementation equivalence check.

- Validation checklist: **100.0/100** (22/22 gates passed). This is a checklist score, not a probability.
- Full sample after warm-up: **24.80% CAGR, 19.79% vol, Sharpe 1.218, MaxDD -17.98%**.
- BTC buy & hold over the same tradable window: **39.13% CAGR, 60.95% vol, Sharpe 0.850, MaxDD -75.15%**.
- 2022–2023 validation slice: **4.12% CAGR, MaxDD -16.96%**.
- 2024+ time-slice holdout: **28.70% CAGR, 21.34% vol, Sharpe 1.288, MaxDD -13.03%**.
- Cost stress 100 bps one-way: **21.57% CAGR, Sharpe 1.084, MaxDD -23.85%**.
- Full-year positive ratio: **85.7%** (2022 was approximately -0.14%).
- 81-neighbour sensitivity grid: **100% positive CAGR; 100% Sharpe >= 0.60; median Sharpe 1.171; median MaxDD -20.49%**.
- 5,000x circular 13-week block bootstrap: historical-recombination P(CAGR>0) **99.6%**, P(MaxDD>-50%) **99.84%**, CAGR P5 **7.48%**, adverse MaxDD P5 **-34.94%**. These are not future probabilities.
- Mean gross exposure: **24.24%**; annual one-way turnover: **3.51x**.
- Future-data mutation test: PASS.
- Explicit one-week execution-lag test: PASS.
- Independent shadow implementation: PASS within `1e-12` tolerance.

## Data integrity

Historical weekly BTCUSDT and ETHUSDT data are pinned to commit `646927e67424160b328658cd43c8a135e639374e` of `yanniedog/binance-historical-OHLCV-data`.

- BTC file SHA-256: `f1bba1a115bd276635d8061b208508f74d5d3ac65b6925ab1df24067fafaa334`
- ETH file SHA-256: `01310a04dcd60b86c91f4e5fe50481c14ae4976abff9ade61950296dc1b397d0`
- The mirror's terminal 2026-01-05 row was detected as incomplete and is always dropped.
- The preceding completed week 2025-12-29 matches Binance direct exactly: BTC 91,529.73; ETH 3,144.70.
- Additional Binance-direct weekly-close cross-checks match exactly at Mar-2020, Nov-2021, Nov-2022 and Mar-2024 for both BTC and ETH.

## Live runtime

GitHub-hosted Azure runners receive Binance HTTP 451 because of runner geography, so **GitHub Actions is validation-only**, not the live market-data transport. The live production decision uses the connected Binance read-only market-data source. A data-source failure is fail-closed: **no new trade**.

Current production decisions must always use the latest *completed* Binance weekly candle and exclude the current partial week.

## Research quarantine

The following components are not allowed to alter CORE-PROD v1.0 positions:

- cross-asset / altcoin Top-K ranking;
- original 1,064/364 Cycle timing rule;
- composite Cycle Clock;
- Phase5D macro overlay;
- Phase6A tactical heat overlay;
- Phase6B derivatives/leverage overlay.

They remain research/challenger layers until independently promoted through frozen OOS/prospective gates.
