from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
import shutil

import pandas as pd
from catboost import CatBoostClassifier


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost model from a CSV dataset.")
    parser.add_argument("--data-path", required=True, help="CSV file path.")
    parser.add_argument("--target-col", required=True, help="Target column name.")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g. META, BTCUSDT).")
    parser.add_argument("--timeframe", default="1m", help="Timeframe label.")
    parser.add_argument(
        "--feature-cols",
        default="",
        help="Comma-separated feature columns. If omitted, all non-target columns are used.",
    )
    parser.add_argument(
        "--version",
        default="",
        help="Version folder name. If omitted, generated from UTC timestamp.",
    )
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument(
        "--promote-live",
        action="store_true",
        help="Copy the trained model into models/live/catboost/<symbol>_<timeframe>.cbm",
    )
    return parser.parse_args()


def resolve_feature_cols(df: pd.DataFrame, target_col: str, raw_feature_cols: str) -> list[str]:
    if raw_feature_cols.strip():
        cols = [col.strip() for col in raw_feature_cols.split(",") if col.strip()]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return cols
    return [col for col in df.columns if col != target_col]


def split_by_time(df: pd.DataFrame, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1.")
    split_idx = int(len(df) * (1 - val_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Validation split leaves no data for train or validation.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column not found: {args.target_col}")

    feature_cols = resolve_feature_cols(df, args.target_col, args.feature_cols)
    df = df.dropna(subset=feature_cols + [args.target_col]).reset_index(drop=True)
    if len(df) < 50:
        raise ValueError("Not enough rows after dropna; need at least 50 rows.")

    train_df, val_df = split_by_time(df, args.val_size)

    x_train = train_df[feature_cols]
    y_train = train_df[args.target_col]
    x_val = val_df[feature_cols]
    y_val = val_df[args.target_col]

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        eval_metric="AUC",
        loss_function="Logloss",
        verbose=50,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

    version = args.version.strip() or datetime.now(UTC).strftime("v%Y-%m-%d_%H%M%S")
    training_dir = (
        REPO_ROOT
        / "models"
        / "training"
        / "catboost"
        / args.symbol
        / args.timeframe
        / version
    )
    training_dir.mkdir(parents=True, exist_ok=False)

    model_path = training_dir / "model.cbm"
    model.save_model(model_path)

    val_probs = model.predict_proba(x_val)[:, 1]
    metrics = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "val_probability_mean": float(val_probs.mean()),
        "best_iteration": int(model.get_best_iteration()),
        "best_score": model.get_best_score(),
    }
    metadata = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "target_col": args.target_col,
        "feature_cols": feature_cols,
        "data_path": str(data_path),
        "version": version,
    }
    train_config = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "val_size": args.val_size,
    }

    (training_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (training_dir / "feature_schema.json").write_text(json.dumps(metadata, indent=2))
    (training_dir / "train_config.json").write_text(json.dumps(train_config, indent=2))

    if args.promote_live:
        live_dir = REPO_ROOT / "models" / "live" / "catboost"
        live_dir.mkdir(parents=True, exist_ok=True)
        live_model_path = live_dir / f"{args.symbol}_{args.timeframe}.cbm"
        shutil.copy2(model_path, live_model_path)
        print(f"Promoted model to live path: {live_model_path}")

    print(f"Training artifact directory: {training_dir}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()

