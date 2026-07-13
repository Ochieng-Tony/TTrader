"""
TTrader Signal Generator
Converts raw model class-probabilities into an actionable trade suggestion:
direction, confidence, suggested position size (capped fractional Kelly),
and rule-based confirmation filters to reduce false positives.
"""
import numpy as np
import pandas as pd

DIRECTION_MAP = {0: "SHORT", 1: "FLAT", 2: "LONG"}


def kelly_fraction(win_prob: float, win_loss_ratio: float = 1.5, cap: float = 0.25) -> float:
    """
    Fractional Kelly sizing: f* = p - (1-p)/b, where b = avg_win/avg_loss.
    Capped hard at `cap` (e.g. 25% of capital) to avoid over-betting on
    a model that is right ~40% of the time, not 90%.
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    f = win_prob - (1 - win_prob) / win_loss_ratio
    return float(np.clip(f, 0, cap))


def generate_signal(latest_features: pd.Series, model, feature_cols: list,
                     min_confidence: float = 0.40, require_trend_confirmation: bool = True) -> dict:
    """
    latest_features: a single row (most recent bar) of engineered features
    model: trained LightGBM classifier
    Returns a structured trade suggestion dict.
    """
    X = latest_features[feature_cols].to_frame().T
    probs = model.predict_proba(X)[0]  # [P(down), P(flat), P(up)]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    direction = DIRECTION_MAP[pred_class]

    # Rule-based confirmation filter: don't go long against a strong downtrend, vice versa
    confirmed = True
    reason = []
    if require_trend_confirmation and direction != "FLAT":
        adx = latest_features.get("adx_14", 0)
        sma_trend = latest_features.get("sma_50_ratio", 0)
        if direction == "LONG" and sma_trend < -0.03 and adx > 25:
            confirmed = False
            reason.append("Strong existing downtrend (ADX>25, price <50SMA) contradicts LONG signal")
        elif direction == "SHORT" and sma_trend > 0.03 and adx > 25:
            confirmed = False
            reason.append("Strong existing uptrend (ADX>25, price >50SMA) contradicts SHORT signal")

    if confidence < min_confidence:
        direction = "FLAT"
        reason.append(f"Confidence {confidence:.2f} below threshold {min_confidence}")

    if not confirmed:
        direction = "FLAT"

    win_loss_ratio = 1.5  # assume avg winner is 1.5x avg loser given ATR-based stop/target below
    size_fraction = kelly_fraction(confidence, win_loss_ratio) if direction != "FLAT" else 0.0

    atr = latest_features.get("atr_14", 0.02)  # already normalized by close in features
    stop_loss_pct = atr * 1.5
    take_profit_pct = atr * 1.5 * win_loss_ratio

    return {
        "direction": direction,
        "confidence": round(confidence, 3),
        "probabilities": {"down": round(float(probs[0]), 3), "flat": round(float(probs[1]), 3), "up": round(float(probs[2]), 3)},
        "suggested_position_size_pct_of_capital": round(size_fraction * 100, 2),
        "stop_loss_pct": round(stop_loss_pct * 100, 2),
        "take_profit_pct": round(take_profit_pct * 100, 2),
        "notes": reason,
    }


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.loader import get_data
    from features.engineer import build_features, FEATURE_COLUMNS
    from models.predictor import train_walk_forward

    df = get_data("AAPL", period="5y")
    feats = build_features(df)
    model, _ = train_walk_forward(feats, FEATURE_COLUMNS)

    latest = feats.iloc[-1]
    signal = generate_signal(latest, model, FEATURE_COLUMNS)
    print("Latest bar:", feats.index[-1].date())
    for k, v in signal.items():
        print(f"  {k}: {v}")
