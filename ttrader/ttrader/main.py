#!/usr/bin/env python3
"""
TTrader — local market prediction and trade suggestion engine.

Usage:
    python main.py suggest AAPL              # get a live trade suggestion
    python main.py suggest AAPL MSFT GOOG     # multiple tickers
    python main.py backtest AAPL              # validate the strategy historically
    python main.py backtest AAPL --period 5y --retrain-every 60
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

from data.loader import get_data
from features.engineer import build_features, FEATURE_COLUMNS
from models import predictor as model_module
from signals import generator as signal_module
from backtest import engine as backtest_engine


def cmd_suggest(args):
    for ticker in args.tickers:
        print(f"\n{'='*50}\n  {ticker}\n{'='*50}")
        df = get_data(ticker, period=args.period)
        feats = build_features(df, horizon=args.horizon)

        model = model_module.load_model(ticker)
        if model is None or args.retrain:
            model, fold_reports = model_module.train_walk_forward(feats, FEATURE_COLUMNS)
            avg_acc = sum(r["accuracy"] for r in fold_reports) / len(fold_reports)
            print(f"Model trained. Walk-forward OOS accuracy: {avg_acc:.1%} (random baseline: 33.3%)")
            model_module.save_model(model, ticker)
        else:
            print("Using cached model (pass --retrain to force retraining).")

        latest = feats.iloc[-1]
        signal = signal_module.generate_signal(latest, model, FEATURE_COLUMNS,
                                                 min_confidence=args.min_confidence)

        print(f"\nAs of {feats.index[-1].date()}:")
        print(f"  >> SUGGESTION: {signal['direction']}")
        print(f"     Confidence: {signal['confidence']:.1%}")
        print(f"     P(down)={signal['probabilities']['down']:.2f}  "
              f"P(flat)={signal['probabilities']['flat']:.2f}  "
              f"P(up)={signal['probabilities']['up']:.2f}")
        if signal["direction"] != "FLAT":
            print(f"     Suggested size: {signal['suggested_position_size_pct_of_capital']}% of capital")
            print(f"     Stop-loss: -{signal['stop_loss_pct']}%   Take-profit: +{signal['take_profit_pct']}%")
        if signal["notes"]:
            print(f"     Notes: {'; '.join(signal['notes'])}")

        print("\n  [Not financial advice — see DISCLAIMER in README]")


def cmd_backtest(args):
    for ticker in args.tickers:
        print(f"\n{'='*50}\n  Backtesting {ticker}\n{'='*50}")
        df = get_data(ticker, period=args.period)
        feats = build_features(df)

        equity, trades, risk_mgr = backtest_engine.run_backtest(
            feats, FEATURE_COLUMNS, model_module, signal_module,
            initial_capital=args.capital, retrain_every=args.retrain_every, train_window=args.train_window,
        )
        metrics = backtest_engine.compute_metrics(equity, trades)

        print(f"\nInitial capital: ${args.capital:,.2f}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        if risk_mgr.trading_halted:
            print("  [!] Risk circuit breaker triggered (max drawdown limit hit).")


def main():
    parser = argparse.ArgumentParser(description="TTrader: local market prediction & trade suggestions")
    sub = parser.add_subparsers(dest="command", required=True)

    p_suggest = sub.add_parser("suggest", help="Generate a live trade suggestion")
    p_suggest.add_argument("tickers", nargs="+", help="Ticker symbol(s), e.g. AAPL MSFT")
    p_suggest.add_argument("--period", default="2y", help="History to use (default 2y)")
    p_suggest.add_argument("--horizon", type=int, default=5, help="Prediction horizon in bars (default 5)")
    p_suggest.add_argument("--min-confidence", type=float, default=0.40, dest="min_confidence")
    p_suggest.add_argument("--retrain", action="store_true", help="Force retraining instead of using cache")
    p_suggest.set_defaults(func=cmd_suggest)

    p_bt = sub.add_parser("backtest", help="Backtest the strategy historically")
    p_bt.add_argument("tickers", nargs="+")
    p_bt.add_argument("--period", default="5y")
    p_bt.add_argument("--capital", type=float, default=10_000)
    p_bt.add_argument("--retrain-every", type=int, default=60, dest="retrain_every")
    p_bt.add_argument("--train-window", type=int, default=250, dest="train_window")
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
