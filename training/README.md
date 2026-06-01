# Training Workspace

This directory is where model-training code and related configs should live.

## Files

- `training/train_catboost.py`: CLI script to train CatBoost from a CSV dataset.
- `training/sample_train_catboost.ipynb`: notebook version of the same workflow.

## Example usage

```bash
python -m training.train_catboost \
  --data-path data/meta_train.csv \
  --target-col target \
  --symbol META \
  --timeframe 1m \
  --feature-cols "bb_score,ret_1m,ret_5m,rsi_14,atr_14m_pct"
```

Add `--promote-live` to also copy the model into `models/live/catboost/<symbol>_<timeframe>.cbm`.

Training artifacts are saved under:

- `models/training/catboost/<symbol>/<timeframe>/<version>/`

