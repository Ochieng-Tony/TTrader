"""
TTrader Backtester
Walks forward through history bar-by-bar, generating a signal at each step
using ONLY data available up to that point (no lookahead), applies the
risk manager's stop-loss/take-profit/position sizing, and reports
Sharpe ratio, max drawdown, win rate, and equity curve.
"""
import numpy as np
import pandas as pd


class RiskManager:
    def __init__(self, max_position_pct: float = 25.0, max_drawdown_pct: float = 20.0):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.trading_halted = False

    def check_drawdown(self, equity_curve: list) -> bool:
        """Circuit breaker: halts new trades if drawdown exceeds limit."""
        if len(equity_curve) < 2:
            return True
        peak = max(equity_curve)
        current = equity_curve[-1]
        dd = (peak - current) / peak * 100
        if dd >= self.max_drawdown_pct:
            self.trading_halted = True
        return not self.trading_halted

    def size_position(self, suggested_pct: float) -> float:
        return min(suggested_pct, self.max_position_pct)


def run_backtest(feature_df: pd.DataFrame, feature_cols: list, model_module, signal_module,
                  initial_capital: float = 10_000, retrain_every: int = 60, train_window: int = 250):
    """
    Rolling backtest: retrains the model periodically (retrain_every bars)
    using only the trailing `train_window` bars, then trades forward until
    the next retrain. This mimics how the system would run live.
    """
    risk_mgr = RiskManager()
    equity = [initial_capital]
    cash = initial_capital
    position = 0.0  # +pct long, -pct short (as fraction of capital)
    entry_price = None
    trades = []

    n = len(feature_df)
    model = None

    for i in range(train_window, n - 1):
        # Retrain periodically on trailing window only (no future data)
        if model is None or (i - train_window) % retrain_every == 0:
            train_slice = feature_df.iloc[max(0, i - train_window):i]
            if len(train_slice) > 50:
                model, _ = model_module.train_walk_forward(train_slice, feature_cols, n_splits=3)

        if model is None:
            equity.append(equity[-1])
            continue

        row = feature_df.iloc[i]
        signal = signal_module.generate_signal(row, model, feature_cols)

        price_now = (1 + row["fwd_return"]) ** 0 * 1.0  # placeholder; use actual close ratio below
        next_return = feature_df.iloc[i]["fwd_return"]  # forward return already computed in features (horizon bars)

        can_trade = risk_mgr.check_drawdown(equity)
        size_pct = risk_mgr.size_position(signal["suggested_position_size_pct_of_capital"]) / 100.0

        pnl_pct = 0.0
        if can_trade and signal["direction"] != "FLAT" and size_pct > 0:
            direction_mult = 1 if signal["direction"] == "LONG" else -1
            raw_move = next_return * direction_mult

            sl = signal["stop_loss_pct"] / 100.0
            tp = signal["take_profit_pct"] / 100.0
            capped_move = float(np.clip(raw_move, -sl, tp))

            pnl_pct = capped_move * size_pct
            trades.append({
                "date": feature_df.index[i],
                "direction": signal["direction"],
                "confidence": signal["confidence"],
                "size_pct": size_pct * 100,
                "pnl_pct_of_capital": pnl_pct * 100,
            })

        new_equity = equity[-1] * (1 + pnl_pct)
        equity.append(new_equity)

    equity_series = pd.Series(equity, index=feature_df.index[train_window - 1: train_window - 1 + len(equity)])
    return equity_series, pd.DataFrame(trades), risk_mgr


def compute_metrics(equity_series: pd.Series, trades_df: pd.DataFrame) -> dict:
    returns = equity_series.pct_change().dropna()
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    max_dd = drawdown.min()

    win_rate = (trades_df["pnl_pct_of_capital"] > 0).mean() * 100 if len(trades_df) else 0.0

    return {
        "total_return_pct": round(total_return, 2),
        "annualized_sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "num_trades": len(trades_df),
        "win_rate_pct": round(win_rate, 2),
        "final_equity": round(equity_series.iloc[-1], 2),
    }


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.loader import get_data
    from features.engineer import build_features, FEATURE_COLUMNS
    import models.predictor as model_module
    import signals.generator as signal_module

    df = get_data("AAPL", period="5y")
    feats = build_features(df)

    print("Running rolling backtest (this retrains periodically, may take a moment)...")
    equity, trades, risk_mgr = run_backtest(feats, FEATURE_COLUMNS, model_module, signal_module,
                                             initial_capital=10_000, retrain_every=60, train_window=250)

    metrics = compute_metrics(equity, trades)
    print("\n--- Backtest Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if risk_mgr.trading_halted:
        print("\n[!] Risk circuit breaker triggered during backtest (max drawdown exceeded).")

    print(f"\nSample trades:\n{trades.head(10)}")
