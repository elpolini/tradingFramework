from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from engine import FROZEN_CONFIG, latest_signal

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def fetch_completed_weekly_closes(symbol: str, limit: int = 120) -> pd.Series:
    query = urllib.parse.urlencode({"symbol": symbol, "interval": "1w", "limit": limit})
    req = urllib.request.Request(
        f"{BINANCE_KLINES}?{query}",
        headers={"User-Agent": "gccc-core-prod-live/1.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list) or len(payload) < 60:
        raise RuntimeError(f"{symbol}: insufficient Binance weekly data")

    now_ms = int(time.time() * 1000)
    rows = []
    for k in payload:
        open_ms = int(k[0])
        close_ms = int(k[6])
        close = float(k[4])
        # The current weekly kline is explicitly excluded until its exchange
        # close timestamp has passed. Fail-closed: no partial-bar inference.
        if close_ms < now_ms:
            rows.append((pd.to_datetime(open_ms, unit="ms", utc=True).tz_localize(None), close))
    if len(rows) < 55:
        raise RuntimeError(f"{symbol}: insufficient completed weekly bars after filtering")
    s = pd.Series(dict(rows), name=symbol, dtype=float).sort_index()
    if s.index.has_duplicates:
        raise RuntimeError(f"{symbol}: duplicate weekly opens")
    return s


def build_live_prices() -> pd.DataFrame:
    btc = fetch_completed_weekly_closes("BTCUSDT").rename("BTC")
    eth = fetch_completed_weekly_closes("ETHUSDT").rename("ETH")
    prices = pd.concat([btc, eth], axis=1, join="inner").dropna()
    if len(prices) < 55:
        raise RuntimeError("insufficient aligned BTC/ETH completed weekly history")
    diffs = prices.index.to_series().diff().dropna().dt.days
    if not (diffs == 7).all():
        raise RuntimeError("non-weekly cadence in Binance live feed")
    return prices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="live_signal.json")
    args = parser.parse_args()

    prices = build_live_prices()
    signal = latest_signal(prices, FROZEN_CONFIG)
    signal["source"] = "Binance spot BTCUSDT/ETHUSDT 1w"
    signal["partial_week_excluded"] = True
    signal["decision"] = "TARGET_FOR_CURRENT_WEEK_FROM_LAST_COMPLETED_WEEK"
    signal["fail_policy"] = "NO_NEW_TRADE_ON_DATA_ERROR"

    text = json.dumps(signal, indent=2, sort_keys=True)
    Path(args.json_out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
