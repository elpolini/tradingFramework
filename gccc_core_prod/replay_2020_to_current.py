from __future__ import annotations

import base64
import io
import json
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import FROZEN_CONFIG, build_target_weights, performance_metrics, strategy_returns

DATA_COMMIT = "646927e67424160b328658cd43c8a135e639374e"
DATA_ROOT = f"https://raw.githubusercontent.com/yanniedog/binance-historical-OHLCV-data/{DATA_COMMIT}"
RECENT_PATH = Path("replay_data/recent_2026_weekly_closes.csv")
OUT = Path("replay_artifacts")
OUT.mkdir(exist_ok=True)


def fetch_static(symbol: str) -> pd.Series:
    url = f"{DATA_ROOT}/{symbol}USDT_1w.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "gccc-core-prod-replay/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["timestamp"])
    return df.set_index("timestamp")["close"].astype(float).rename(symbol)


def load_prices() -> pd.DataFrame:
    static = pd.concat([fetch_static("BTC"), fetch_static("ETH")], axis=1, join="inner").dropna()
    # Pinned snapshot's final 2026-01-05 row was known incomplete during production validation.
    static = static.loc[static.index < pd.Timestamp("2026-01-05")]
    recent = pd.read_csv(RECENT_PATH, parse_dates=["timestamp"]).set_index("timestamp")[["BTC", "ETH"]]
    recent = recent.loc[recent.index >= pd.Timestamp("2026-01-05")]
    prices = pd.concat([static, recent]).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    if not (prices.index.to_series().diff().dropna().dt.days == 7).all():
        raise RuntimeError("weekly cadence broken")
    return prices


def metrics(r: pd.Series) -> dict:
    return performance_metrics(r, annualization=52)


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def savefig(name: str):
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def pct(x):
    return f"{100*x:.2f}%"


def main():
    prices = load_prices()
    target = build_target_weights(prices, FROZEN_CONFIG)
    net, effective, turnover = strategy_returns(prices, FROZEN_CONFIG, cost_bps=FROZEN_CONFIG.cost_bps)

    start_open = pd.Timestamp("2019-12-30")  # week ending 2020-01-05
    end_open = pd.Timestamp("2026-08-24")    # week ending 2026-08-30; latest completed week

    px = prices.loc[start_open:end_open].copy()
    net = net.reindex(px.index).fillna(0.0)
    effective = effective.reindex(px.index).fillna(0.0)
    target = target.reindex(px.index).fillna(0.0)
    turnover = turnover.reindex(px.index).fillna(0.0)

    asset_ret = prices.pct_change().reindex(px.index).fillna(0.0)
    btc_ret = asset_ret["BTC"]
    eth_ret = asset_ret["ETH"]

    cash_eff = 1.0 - effective.sum(axis=1)
    cash_tgt = 1.0 - target.sum(axis=1)
    gross = effective.sum(axis=1)

    eq_strategy = (1.0 + net).cumprod()
    eq_btc = (1.0 + btc_ret).cumprod()
    eq_eth = (1.0 + eth_ret).cumprod()

    sma40 = prices.rolling(FROZEN_CONFIG.trend_weeks).mean().reindex(px.index)
    mom13 = (prices / prices.shift(FROZEN_CONFIG.momentum_weeks) - 1.0).reindex(px.index)
    ann_vol = prices.pct_change().rolling(FROZEN_CONFIG.vol_weeks).std(ddof=1) * math.sqrt(52)
    ann_vol = ann_vol.reindex(px.index)
    btc_gate = (px["BTC"] > sma40["BTC"]) & (mom13["BTC"] > 0)
    eth_gate = btc_gate & (px["ETH"] > sma40["ETH"]) & (mom13["ETH"] > 0)

    week_end = px.index + pd.Timedelta(days=6)

    def regime_from_row(btc_w: float, eth_w: float) -> str:
        if btc_w + eth_w <= 1e-12:
            return "CASH"
        if eth_w <= 1e-12:
            return "BTC"
        return "BTC+ETH"

    effective_regime = pd.Series(
        [regime_from_row(b, e) for b, e in effective[["BTC", "ETH"]].to_numpy()],
        index=px.index,
    )
    signal_regime = pd.Series(
        [regime_from_row(b, e) for b, e in target[["BTC", "ETH"]].to_numpy()],
        index=px.index,
    )

    replay = pd.DataFrame(index=px.index)
    replay["week_end"] = week_end
    replay["btc_close"] = px["BTC"]
    replay["eth_close"] = px["ETH"]
    replay["btc_week_return"] = btc_ret
    replay["eth_week_return"] = eth_ret
    replay["btc_sma40w"] = sma40["BTC"]
    replay["eth_sma40w"] = sma40["ETH"]
    replay["btc_mom13w"] = mom13["BTC"]
    replay["eth_mom13w"] = mom13["ETH"]
    replay["btc_vol12w"] = ann_vol["BTC"]
    replay["eth_vol12w"] = ann_vol["ETH"]
    replay["btc_gate_signal"] = btc_gate
    replay["eth_gate_signal"] = eth_gate
    replay["signal_regime_next_week"] = signal_regime
    replay["target_btc_next_week"] = target["BTC"]
    replay["target_eth_next_week"] = target["ETH"]
    replay["target_cash_next_week"] = cash_tgt
    replay["effective_regime_this_week"] = effective_regime
    replay["effective_btc_this_week"] = effective["BTC"]
    replay["effective_eth_this_week"] = effective["ETH"]
    replay["effective_cash_this_week"] = cash_eff
    replay["gross_exposure_this_week"] = gross
    replay["turnover_one_way"] = turnover
    replay["strategy_net_return"] = net
    replay["strategy_equity"] = eq_strategy
    replay["btc_buyhold_equity"] = eq_btc
    replay["eth_buyhold_equity"] = eq_eth
    replay["strategy_drawdown"] = drawdown(eq_strategy)
    replay["btc_drawdown"] = drawdown(eq_btc)
    replay["eth_drawdown"] = drawdown(eq_eth)

    # A deliberately simple ex-post diagnostic, not a probabilistic accuracy measure.
    risk_on = gross > 1e-12
    replay["simple_call_hit"] = (risk_on & (net > 0)) | ((~risk_on) & (btc_ret <= 0))
    replay["cash_protected_btc_down_week"] = (~risk_on) & (btc_ret < 0)
    replay["cash_missed_btc_up_week"] = (~risk_on) & (btc_ret > 0)
    replay["risk_on_positive_week"] = risk_on & (net > 0)
    replay["risk_on_negative_week"] = risk_on & (net < 0)

    replay.reset_index(names="week_open").to_csv(OUT / "gccc_core_prod_weekly_replay_2020_2026.csv", index=False)

    # Performance metrics
    m_strategy = metrics(net)
    m_btc = metrics(btc_ret)
    m_eth = metrics(eth_ret)

    year_df = pd.DataFrame({"strategy": net, "BTC": btc_ret, "ETH": eth_ret})
    year_df["year"] = week_end.year
    annual = year_df.groupby("year")[["strategy", "BTC", "ETH"]].apply(lambda g: (1.0 + g).prod() - 1.0)
    annual.to_csv(OUT / "annual_returns.csv")

    transitions_mask = effective_regime.ne(effective_regime.shift(1))
    transitions = replay.loc[transitions_mask, [
        "week_end", "effective_regime_this_week", "effective_btc_this_week",
        "effective_eth_this_week", "effective_cash_this_week", "btc_close", "eth_close"
    ]].copy()
    transitions.to_csv(OUT / "regime_transitions.csv", index=False)

    # Chart 1 — equity
    plt.figure(figsize=(12, 6.5))
    plt.plot(week_end, eq_strategy, label="CORE-PROD v1.0")
    plt.plot(week_end, eq_btc, label="BTC buy & hold")
    plt.plot(week_end, eq_eth, label="ETH buy & hold")
    plt.yscale("log")
    plt.title("GCCC CORE-PROD v1.0 — replay walk-forward 2020 to 2026")
    plt.xlabel("Completed week")
    plt.ylabel("Equity index, log scale (start = 1)")
    plt.legend()
    savefig("01_equity_replay.png")

    # Chart 2 — drawdown
    plt.figure(figsize=(12, 6.5))
    plt.plot(week_end, replay["strategy_drawdown"], label="CORE-PROD v1.0")
    plt.plot(week_end, replay["btc_drawdown"], label="BTC buy & hold")
    plt.plot(week_end, replay["eth_drawdown"], label="ETH buy & hold")
    plt.title("Drawdown comparison")
    plt.xlabel("Completed week")
    plt.ylabel("Drawdown")
    plt.legend()
    savefig("02_drawdown.png")

    # Chart 3 — actual weekly allocations
    plt.figure(figsize=(12, 6.5))
    plt.stackplot(week_end, effective["BTC"], effective["ETH"], cash_eff,
                  labels=["BTC", "ETH", "Cash"])
    plt.title("Effective weekly allocation — signal t executed in t+1")
    plt.xlabel("Completed week")
    plt.ylabel("Portfolio weight")
    plt.legend(loc="upper left")
    savefig("03_weekly_allocation.png")

    # Chart 4 — gross crypto exposure
    plt.figure(figsize=(12, 6.5))
    plt.step(week_end, gross, where="post")
    plt.title("Gross crypto exposure by week")
    plt.xlabel("Completed week")
    plt.ylabel("BTC + ETH exposure")
    savefig("04_gross_exposure.png")

    # Chart 5 — BTC close and slow trend filter
    plt.figure(figsize=(12, 6.5))
    plt.plot(week_end, px["BTC"], label="BTC weekly close")
    plt.plot(week_end, sma40["BTC"], label="BTC SMA40w")
    plt.yscale("log")
    plt.title("BTC master gate — price vs SMA40w")
    plt.xlabel("Completed week")
    plt.ylabel("BTCUSDT, log scale")
    plt.legend()
    savefig("05_btc_master_gate.png")

    # Chart 6 — annual returns
    ax = annual.plot(kind="bar", figsize=(12, 6.5))
    ax.set_title("Calendar-year returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "06_annual_returns.png", dpi=170, bbox_inches="tight")
    plt.close()

    # Chart 7 — weekly net return distribution over time
    plt.figure(figsize=(12, 6.5))
    plt.bar(week_end, net, width=5)
    plt.title("CORE-PROD weekly net returns")
    plt.xlabel("Completed week")
    plt.ylabel("Net weekly return")
    savefig("07_weekly_returns.png")

    simple_hit_rate = float(replay["simple_call_hit"].mean())
    counts = {
        "weeks": int(len(replay)),
        "cash_weeks": int((~risk_on).sum()),
        "risk_on_weeks": int(risk_on.sum()),
        "cash_protected_btc_down_weeks": int(replay["cash_protected_btc_down_week"].sum()),
        "cash_missed_btc_up_weeks": int(replay["cash_missed_btc_up_week"].sum()),
        "risk_on_positive_weeks": int(replay["risk_on_positive_week"].sum()),
        "risk_on_negative_weeks": int(replay["risk_on_negative_week"].sum()),
        "simple_call_hit_rate": simple_hit_rate,
    }

    summary = {
        "model": "GCCC CORE-PROD v1.0",
        "replay": "strict weekly walk-forward with one-week execution lag",
        "period_week_end": [str(week_end.min().date()), str(week_end.max().date())],
        "config": FROZEN_CONFIG.to_dict(),
        "strategy": m_strategy,
        "btc_buy_hold": m_btc,
        "eth_buy_hold": m_eth,
        "counts": counts,
        "latest_signal_for_next_week": {
            "week_end": str(week_end[-1].date()),
            "regime": signal_regime.iloc[-1],
            "BTC": float(target["BTC"].iloc[-1]),
            "ETH": float(target["ETH"].iloc[-1]),
            "CASH": float(cash_tgt.iloc[-1]),
        },
        "note": "simple_call_hit_rate is an ex-post descriptive diagnostic, not a probability or institutional accuracy metric",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Report markdown
    annual_md = annual.applymap(lambda x: f"{100*x:.1f}%").to_markdown()
    trans_md = transitions.tail(30).to_markdown(index=False)
    report = f"""# GCCC CORE-PROD v1.0 — replay 2020 → 2026-08-30

## Protocol
- Completed Binance weekly closes only.
- `40w trend / 13w momentum / 12w vol / 25% vol target`.
- BTC is the master gate; ETH is subordinate.
- Signal observed at completed week `t` is executed for week `t+1`.
- Zero leverage; residual is cash.
- 25 bps one-way turnover cost.
- No future price is used by any signal.

## Results
| Metric | CORE-PROD | BTC B&H | ETH B&H |
|---|---:|---:|---:|
| CAGR | {pct(m_strategy['cagr'])} | {pct(m_btc['cagr'])} | {pct(m_eth['cagr'])} |
| Annualized vol | {pct(m_strategy['vol'])} | {pct(m_btc['vol'])} | {pct(m_eth['vol'])} |
| Sharpe | {m_strategy['sharpe']:.3f} | {m_btc['sharpe']:.3f} | {m_eth['sharpe']:.3f} |
| Max drawdown | {pct(m_strategy['max_dd'])} | {pct(m_btc['max_dd'])} | {pct(m_eth['max_dd'])} |
| Calmar | {m_strategy['calmar']:.3f} | {m_btc['calmar']:.3f} | {m_eth['calmar']:.3f} |

## Weekly decision diagnostics
- Weeks: **{counts['weeks']}**
- Cash weeks: **{counts['cash_weeks']}**
- Risk-on weeks: **{counts['risk_on_weeks']}**
- Cash weeks that avoided a BTC-negative week: **{counts['cash_protected_btc_down_weeks']}**
- Cash weeks while BTC rose (missed rally by this simple benchmark): **{counts['cash_missed_btc_up_weeks']}**
- Risk-on weeks with positive strategy return: **{counts['risk_on_positive_weeks']}**
- Risk-on weeks with negative strategy return: **{counts['risk_on_negative_weeks']}**
- Simple descriptive weekly call hit-rate: **{100*simple_hit_rate:.1f}%**

The hit-rate above is deliberately labelled **descriptive**; it is not a probability of forecasting accuracy.

## Annual returns
{annual_md}

## Most recent regime transitions
{trans_md}

## Latest completed-week signal
Week ending **{week_end[-1].date()}** produces next-week target:
- BTC **{100*target['BTC'].iloc[-1]:.2f}%**
- ETH **{100*target['ETH'].iloc[-1]:.2f}%**
- Cash **{100*cash_tgt.iloc[-1]:.2f}%**
- Regime **{signal_regime.iloc[-1]}**
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    # Self-contained-ish HTML with embedded images
    image_names = [
        "01_equity_replay.png", "02_drawdown.png", "03_weekly_allocation.png",
        "04_gross_exposure.png", "05_btc_master_gate.png", "06_annual_returns.png",
        "07_weekly_returns.png"
    ]
    imgs = []
    for name in image_names:
        b64 = base64.b64encode((OUT / name).read_bytes()).decode("ascii")
        imgs.append(f'<section><h2>{name}</h2><img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto"></section>')

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>GCCC CORE-PROD Replay 2020-2026</title>
<style>body{{font-family:Arial,sans-serif;max-width:1250px;margin:30px auto;padding:0 18px;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}section{{margin:32px 0}}.kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{border:1px solid #ddd;border-radius:10px;padding:14px}}@media(max-width:800px){{.kpi{{grid-template-columns:1fr 1fr}}}}</style></head><body>
<h1>GCCC CORE-PROD v1.0 — strict replay 2020 → 2026-08-30</h1>
<p>Walk-forward: only completed weekly data available at each date; one-week execution lag; 25 bps one-way costs; zero leverage.</p>
<div class='kpi'><div class='card'><b>CORE CAGR</b><br>{pct(m_strategy['cagr'])}</div><div class='card'><b>CORE Sharpe</b><br>{m_strategy['sharpe']:.3f}</div><div class='card'><b>CORE MaxDD</b><br>{pct(m_strategy['max_dd'])}</div><div class='card'><b>Simple weekly call hit</b><br>{100*simple_hit_rate:.1f}%</div></div>
<h2>Comparison</h2><table><tr><th>Metric</th><th>CORE-PROD</th><th>BTC B&H</th><th>ETH B&H</th></tr>
<tr><td>CAGR</td><td>{pct(m_strategy['cagr'])}</td><td>{pct(m_btc['cagr'])}</td><td>{pct(m_eth['cagr'])}</td></tr>
<tr><td>Vol</td><td>{pct(m_strategy['vol'])}</td><td>{pct(m_btc['vol'])}</td><td>{pct(m_eth['vol'])}</td></tr>
<tr><td>Sharpe</td><td>{m_strategy['sharpe']:.3f}</td><td>{m_btc['sharpe']:.3f}</td><td>{m_eth['sharpe']:.3f}</td></tr>
<tr><td>MaxDD</td><td>{pct(m_strategy['max_dd'])}</td><td>{pct(m_btc['max_dd'])}</td><td>{pct(m_eth['max_dd'])}</td></tr></table>
<h2>Decision diagnostics</h2><p>Cash weeks: {counts['cash_weeks']} · Risk-on weeks: {counts['risk_on_weeks']} · Cash-protected BTC down weeks: {counts['cash_protected_btc_down_weeks']} · Cash-missed BTC up weeks: {counts['cash_missed_btc_up_weeks']} · Risk-on positive/negative: {counts['risk_on_positive_weeks']}/{counts['risk_on_negative_weeks']}.</p>
{''.join(imgs)}
<h2>Latest target</h2><p>Week ending {week_end[-1].date()}: BTC {100*target['BTC'].iloc[-1]:.2f}% · ETH {100*target['ETH'].iloc[-1]:.2f}% · Cash {100*cash_tgt.iloc[-1]:.2f}%.</p>
<p><small>The simple weekly call hit-rate is an ex-post descriptive diagnostic, not a probability of future forecasting success.</small></p>
</body></html>"""
    (OUT / "gccc_core_prod_replay_2020_2026.html").write_text(html, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
