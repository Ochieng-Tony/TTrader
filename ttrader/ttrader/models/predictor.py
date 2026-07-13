"""
TTrader Model Layer
Gradient-boosted tree classifier (LightGBM) predicting forward direction
(Up / Flat / Down). Uses walk-forward validation — never random k-fold —
because shuffling time series causes lookahead bias.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def walk_forward_splits(n_rows: int, n_splits: int = 5, min_train_size: int = 100):
    """
    Yields (train_idx, test_idx) where train always precedes test in time,
    and the train window grows with each split (expanding window).
    """
    fold_size = (n_rows - min_train_size) // n_splits
    for i in range(n_splits):
        train_end = min_train_size + i * fold_size
        test_end = train_end + fold_size
        if test_end > n_rows:
            break
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def train_walk_forward(feature_df: pd.DataFrame, feature_cols: list, n_splits: int = 5):
    """
    Trains across walk-forward folds, returns the final model (trained on all
    data) plus out-of-sample metrics from each fold for honest performance estimate.
    """
    X = feature_df[feature_cols]
    y = feature_df["label"].values

    fold_reports = []
    for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(len(feature_df), n_splits)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=3,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        fold_reports.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": acc,
        })

    # Final model trained on the full dataset for production use
    final_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=3, verbosity=-1,
    )
    final_model.fit(X, y)

    return final_model, fold_reports


def save_model(model, ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}_lgbm.joblib")
    joblib.dump(model, path)
    return path


def load_model(ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}_lgbm.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def feature_importance(model, feature_cols: list) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.loader import get_data
    from features.engineer import build_features, FEATURE_COLUMNS

    df = get_data("AAPL", period="5y")
    feats = build_features(df)
    model, reports = train_walk_forward(feats, FEATURE_COLUMNS, n_splits=5)

    print("Walk-forward fold results:")
    for r in reports:
        print(f"  Fold {r['fold']}: train={r['train_size']} test={r['test_size']} acc={r['accuracy']:.3f}")

    avg_acc = np.mean([r["accuracy"] for r in reports])
    print(f"\nAverage out-of-sample accuracy: {avg_acc:.3f} (baseline random = 0.333 for 3 classes)")
    print("\nTop features:")
    print(feature_importance(model, FEATURE_COLUMNS).head(8))

    save_model(model, "AAPL")
