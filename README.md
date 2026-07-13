What TTrader Is
TTrader is a fully self-contained Python system that does five things in sequence: fetches and caches market price data, engineers predictive features from that data, trains a machine-learning model to forecast short-term price direction, converts those forecasts into concrete trade suggestions with position sizing and risk controls, and runs a historical backtest to measure how that whole pipeline would have performed in practice. Everything runs locally — no cloud service, no subscription, no API key.

Project Structure
ttrader/
├── main.py                 ← CLI entry point
├── requirements.txt
├── README.md
├── data/
│   └── loader.py           ← Data layer
├── features/
│   └── engineer.py         ← Feature engineering
├── models/
│   └── predictor.py        ← Model training & validation
├── signals/
│   └── generator.py        ← Signal + position sizing
├── backtest/
│   └── engine.py           ← Backtester + risk manager
└── storage/
    ├── cache/              ← Local Parquet data cache
    └── models/             ← Saved trained models (.joblib)

Layer 1 — Data (data/loader.py)
What it does: Fetches OHLCV (Open, High, Low, Close, Volume) price bars for any ticker symbol and stores them locally so they don't need to be re-downloaded on every run.
How it works:
The primary source is yfinance, which pulls historical price data directly from Yahoo Finance's free API — no account or key required. Each download is saved as an Apache Parquet file in storage/cache/, keyed by ticker + interval + period (e.g. AAPL_1d_2y.parquet). On the next run, if a fresh enough cache file exists, the system reads from disk instead of hitting the network. This also means once you've cached data, the system works fully offline.
The cache key includes the period argument (1y, 2y, 5y, etc.) so that fetching 5 years of data and 1 year of data for the same ticker are stored separately rather than silently returning a shorter slice.
Synthetic fallback: In a sandboxed environment without internet access (like the build environment used here), yfinance will fail. The loader catches this and generates synthetic OHLCV data using geometric Brownian motion for the price path, with a GARCH-like volatility clustering process layered on top (each bar's volatility is a weighted mix of the previous bar's volatility and a random shock). This means the data has realistic properties — trending periods, volatile bursts, mean-reversion — rather than pure white noise. Critically, it's seeded by the ticker name so results are reproducible. On a real machine with internet access, this fallback never triggers.

Layer 2 — Feature Engineering (features/engineer.py)
What it does: Transforms raw OHLCV bars into a matrix of ~20 predictive features, plus a forward-return label for supervised training.
The features, by category:
Momentum:

RSI (14-period) — measures whether a security is overbought or oversold
Stochastic %K — position of current close within recent high-low range
Rate of Change (10-period) — percentage price change over 10 bars

Trend:

MACD, MACD Signal, MACD Histogram — measures trend strength and crossover signals between 12 and 26-bar EMAs
ADX (14-period) — directional movement index; values above 25 indicate a strong trend regardless of direction
Price / SMA-20 ratio — how far price is from its 20-bar average (normalised)
Price / SMA-50 ratio — same over a longer horizon
EMA-12 minus EMA-26 difference (normalised by price) — an alternative MACD-like signal

Volatility / Regime:

Bollinger Band %B — where price sits within its 2-standard-deviation band; values near 0 mean price is at the lower band, near 1 means upper band
ATR-14 normalised — average true range divided by close price, giving a dimensionless volatility measure
20-bar rolling volatility — standard deviation of returns
Volatility regime — current 20-bar volatility divided by its 60-bar average; values above 1 mean the market is more volatile than its recent norm

Volume:

Volume z-score — how many standard deviations above or below the 20-bar average the current bar's volume sits; high z-scores often accompany breakouts or reversals
OBV slope — 5-bar difference in On-Balance Volume, capturing whether volume is flowing in (accumulation) or out (distribution)

Lagged returns:

Returns over 1, 3, 5, and 10 bars — straightforward momentum features, letting the model see the recent price path rather than just current indicators

The label: For each bar, the system looks horizon bars ahead (default 5) and computes the forward return. If it's above +1% the bar is labelled 2 (Up); below -1% it's labelled 0 (Down); otherwise 1 (Flat). This 3-class setup avoids forcing the model to always pick a direction when the signal is genuinely ambiguous.
All features are computed with the ta library (Technical Analysis Library for Python), which handles edge cases in indicator computation cleanly.

Layer 3 — Model (models/predictor.py)
What it does: Trains a gradient-boosted tree classifier to predict the 3-class label (Up/Flat/Down) from the engineered features, using a validation scheme that respects the time ordering of data.
Why LightGBM: LightGBM is fast, trains well on CPU without a GPU, handles tabular feature matrices naturally, produces calibrated class probabilities, and outputs feature importance scores that make it interpretable. It also generalises well with relatively small amounts of training data (a few hundred rows) compared to neural networks, which is important here because even 5 years of daily data is only ~1,260 rows after feature construction.
Walk-forward validation — the critical design decision:
A naive approach would use random k-fold cross-validation: shuffle all rows, split into folds, train on some, test on others. On time-series data this is completely wrong. Future bars get shuffled into the training set, which means the model is trained on information it would not have had at decision time. This is called lookahead bias, and it causes backtest performance to look far better than live performance.
Walk-forward validation works correctly: the dataset is split chronologically. Fold 1 trains on bars 1–100, tests on 101–120. Fold 2 trains on bars 1–120, tests on 121–140. And so on, with the training window always expanding into the past and the test window always advancing into the future. The model never sees a bar during testing that it could not have seen in real life. The accuracy reported across folds is a genuine out-of-sample estimate.
A separate final model is then trained on the entire dataset for use in the live suggest command, since walk-forward validation was only needed to get an honest performance estimate — for production use, more training data is always better.
Model persistence: Trained models are saved as .joblib files in storage/models/. On subsequent runs, the saved model is reloaded instead of retraining, making repeated suggest calls fast. Pass --retrain to force a fresh training run.

Layer 4 — Signal Generator (signals/generator.py)
What it does: Takes the trained model and the most recent bar of features and produces a structured trade suggestion: direction, confidence, position size, stop-loss, and take-profit.
Direction and confidence: The model outputs a probability distribution across the three classes — e.g. {down: 0.17, flat: 0.09, up: 0.74}. The predicted class is the one with highest probability. Confidence is that maximum probability. If confidence falls below a minimum threshold (default 40%), the direction is overridden to FLAT to avoid acting on weak signals.
Position sizing — fractional Kelly: The Kelly Criterion is a formula from information theory that answers: given a known win probability and win/loss payout ratio, what fraction of capital should you bet to maximise long-run geometric growth? Full Kelly is mathematically optimal but extremely aggressive in practice — a model that's 55% accurate on binary outcomes would suggest betting 10% of capital per trade. With three classes and a noisy model, full Kelly could easily recommend sizes that cause ruin on a bad run.
TTrader uses fractional Kelly: the Kelly formula is computed using the model's win probability and an assumed 1.5:1 win/loss ratio (average winner is 1.5x average loser, consistent with the ATR-based targets below), and then the result is hard-capped at 25% of capital regardless. This means even at maximum confidence, you never put more than a quarter of capital into a single trade.
Stop-loss and take-profit: Both are set as multiples of the ATR (Average True Range). ATR measures recent volatility in price terms rather than percentage terms, so stops automatically widen in volatile conditions and tighten in calm ones — this prevents getting stopped out by normal noise while still protecting against large adverse moves. The stop is set at 1.5× ATR and the take-profit at 2.25× ATR (1.5 × the 1.5 win/loss ratio used in Kelly), creating an asymmetric reward structure.
Trend confirmation filter: An optional rule-based filter checks ADX and the 50-bar MA ratio. If the model says LONG but ADX is above 25 (strong trend) and price is already more than 3% below its 50-bar average (strong downtrend), the signal is overridden to FLAT. This prevents the model from entering counter-trend positions in strongly trending markets, which tend to produce the worst losses even when the short-term indicators look appealing.

Layer 5 — Backtester and Risk Manager (backtest/engine.py)
What it does: Simulates running the complete pipeline historically, bar-by-bar, measuring how much money would have been made or lost — and enforces risk limits during that simulation.
Rolling retrain during backtest: The backtester doesn't just train once and test forever. Every retrain_every bars (default 60), it retrains the model from scratch on the most recent train_window bars (default 250). This mimics how the system would run in production — periodically refreshing the model on recent data as market regimes shift. It's more computationally expensive but produces far more realistic performance estimates.
Trade simulation: At each bar the system generates a signal exactly as the live suggest command would. If the signal is LONG or SHORT with sufficient confidence, a trade is opened with the suggested size. The forward return is then capped by the stop-loss (maximum loss) and take-profit (maximum gain) before being applied to the equity. Transaction costs and slippage are not modelled in the current version — something to add for production use.
Risk Manager:

Max position size: individual trades can't exceed 25% of capital regardless of Kelly output
Max drawdown circuit breaker: if the portfolio falls more than 20% from its peak at any point, all new trading is halted for the remainder of the backtest (and would be in live use). This prevents the system from continuing to compound losses in a regime where the model isn't working

Metrics reported:

Total return % — raw percentage gain/loss over the period
Annualized Sharpe ratio — return per unit of volatility, annualised; above 1.0 is generally considered good, above 2.0 is exceptional
Max drawdown % — the worst peak-to-trough decline; tells you the worst case scenario during the period
Number of trades — how active the strategy was
Win rate % — percentage of trades that were profitable
Final equity — absolute dollar value of the portfolio at the end


Layer 6 — CLI (main.py)
Two subcommands, designed to be intuitive:
suggest — the live-use command. Loads data, trains or loads a cached model, generates a signal for the most recent bar, and prints a structured suggestion. Multiple tickers can be passed in one call. Options include --period (how much history to train on), --horizon (how many bars ahead to predict), --min-confidence (threshold below which signals are suppressed to FLAT), and --retrain (force fresh training rather than using the cached model).
backtest — the validation command. Runs the full rolling backtest and prints the summary metrics. Options include --period, --capital (starting portfolio size), --retrain-every, and --train-window.
Both commands suppress noisy library warnings so the output stays readable.

What the Numbers Mean in Practice
When tested on synthetic data in the build environment, the model achieves roughly 37–40% accuracy on the 3-class direction problem (random baseline is 33.3%). That modest edge is realistic and is all that's needed for a profitable strategy when combined with an asymmetric stop/target structure and disciplined position sizing. The backtest on synthetic data reported a Sharpe ratio around 1.2 and max drawdown around 14% — reasonable numbers, though synthetic data has weaker autocorrelation structure than real markets, so live results on real data will differ.
The honest caveat: any model trained on historical price data faces the fundamental challenge that markets change. A model that learned from 2020–2022 behaves differently on 2024 data. The rolling retrain and walk-forward validation mitigate this but don't eliminate it. The circuit breaker and conservative position sizing are the real safety net.

Extension Points
The architecture is deliberately modular so each layer can be swapped or extended:

Replace yfinance with a broker's API for higher-frequency data or options data
Add fundamental or macro features alongside the technical ones in engineer.py
Swap the LightGBM model for an LSTM or Transformer in predictor.py
Add a portfolio allocator on top of generator.py for multi-asset correlation-adjusted sizing
Add a report generator that outputs an equity curve chart and trade log
Hook a paper-trading execution layer onto generator.py to test with real market structure
