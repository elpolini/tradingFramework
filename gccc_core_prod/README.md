# GCCC CORE-PROD v1.0

Status before CI gates: **CANDIDATE / NOT YET PROMOTED**.

This directory is deliberately separated from the experimental Global Crypto Cycle Clock stack. The production candidate contains only BTC, ETH and cash. Altcoin ranking, Cycle Clock timing, Phase5D macro, Phase6A heat and Phase6B derivatives are excluded until they pass their own independent validation gates.

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

`validate.py` contains hard gates frozen before the candidate backtest is observed. A failed hard gate means **DO NOT PROMOTE**. A validation score is a checklist-completion score, never a probability of future profitability or reliability.

## Reproducibility

Historical weekly BTCUSDT and ETHUSDT data are pinned to commit `646927e67424160b328658cd43c8a135e639374e` of `yanniedog/binance-historical-OHLCV-data`. The validator records SHA-256 hashes of the exact downloaded files.
