# ============================================================
# MODEL FILTERS
# Default = AlwaysPassModelFilter
# Optional = CatBoostLiveModelFilter
# ============================================================

class AlwaysPassModelFilter:
    """
    Default no-op model filter.

    This means:
        ml_prob = 1.0
        ml_passed = True

    So the strategy behaves normally even if no CatBoost model is loaded.
    """

    def score(self, features):
        return {
            "enabled": False,
            "prob": 1.0,
            "passed": True,
            "missing_features": [],
        }
