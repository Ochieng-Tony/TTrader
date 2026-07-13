# TTrader

A local market prediction and trade-suggestion system. Pulls price data, builds technical features, trains a gradient-boosted model with leakage-safe validation, and turns predictions into sized, risk-managed trade suggestions — all runnable offline on your own machine.

## Architecture

```
data/loader.py        Fetches & caches OHLCV data (yfinance, local Parquet cache)
features/engineer.py  Builds technical/statistical features + forward-return label
models/predictor.py   LightGBM classifier, walk-forward (time-ordered) validation
signals/generator.py  Converts model probabilities -> direction, confidence, size, stop/target
backtest/engine.py    Rolling backtest with periodic retraining + risk circuit breaker
main.py               CLI: `suggest` and `backtest` subcommands
```

Data flows one direction: raw prices → features → model → signal → risk-adjusted suggestion. The backtester re-runs that exact pipeline historically, retraining periodically on a rolling window, so the numbers it reports reflect how the system would actually have behaved live — not a model peeking at its own test set.

## Setup

```bash
cd ttrader
pip install -r requirements.txt
```

Requires Python 3.9+. No API keys needed — `yfinance` pulls free data directly from Yahoo Finance using your machine's normal internet connection.

## Usage

**Get a live trade suggestion:**
```bash
python main.py suggest AAPL
python main.py suggest AAPL MSFT NVDA          # multiple tickers
python main.py suggest AAPL --period 5y --horizon 10
```

**Backtest the strategy historically:**
```bash
python main.py backtest AAPL --period 5y
python main.py backtest AAPL --capital 50000 --retrain-every 30
```

Run `python main.py suggest --help` or `backtest --help` for all options.

## How each piece works

**Data layer** — downloads OHLCV bars via `yfinance` and caches them locally as Parquet, keyed by ticker/interval/period, so repeat runs don't re-hit the network. If no network is available, it falls back to synthetic data (geometric Brownian motion with GARCH-like volatility clustering) so the pipeline still runs end-to-end for testing/demo purposes — on your own machine with internet access, real data is used automatically.

**Feature engineering** — computes ~20 features per bar: RSI, MACD, ADX, Bollinger %B, ATR, moving-average ratios, volume z-score, OBV slope, and lagged returns. The label is the forward return over a configurable horizon (default 5 bars), bucketed into Up/Flat/Down.

**Model** — LightGBM multiclass classifier. Validation is **walk-forward**, not k-fold: each fold trains only on data before the test window and tests on the window immediately after, with the training window expanding over time. This avoids lookahead bias that would make a model look far better than it actually is.

**Signal generator** — takes the model's class probabilities for the most recent bar and turns them into a suggestion: direction, confidence, and a position size from a capped fractional-Kelly formula (capital at risk scales with edge, but is hard-capped at 25% regardless of confidence). A rule-based filter blocks signals that contradict a strong existing trend (e.g., a LONG signal when ADX shows a strong downtrend already in place).

**Risk manager / backtester** — every suggested trade carries an ATR-based stop-loss and take-profit. The backtester retrains the model periodically on a trailing window and walks forward bar-by-bar, applying those stops/targets and a portfolio-level max-drawdown circuit breaker that halts new trades if losses exceed a threshold. Reports total return, annualized Sharpe, max drawdown, win rate, and trade count.

## Disclaimer

This is a decision-support and educational tool, not a financial advisor. Markets are noisy and largely unpredictable; no model can reliably beat the market over time. Backtested performance does not guarantee future results — backtests are also prone to overfitting even with walk-forward validation. Treat any output as one input among many, never as investment advice, and never risk money you can't afford to lose. Claude is not a financial advisor.
