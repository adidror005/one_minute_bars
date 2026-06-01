# Model Artifacts Layout

- `models/live/`: model artifacts used by live trading.
- `models/training/`: outputs from training runs (checkpoints, experiments, metrics).
- Training code lives in `training/`.

Recommended layout:

- `models/live/catboost/<symbol>/<timeframe>/<version>/...`
- `models/training/catboost/<symbol>/<timeframe>/<run_id>/...`

Keep version/run directories immutable and use config/registry pointers to select the active live model.
