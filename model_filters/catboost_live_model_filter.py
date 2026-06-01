class CatBoostLiveModelFilter:
    def __init__(
        self,
        model_path,
        feature_cols,
        prob_threshold=0.50,
        enabled=True,
    ):
        self.model_path = model_path
        self.feature_cols = list(feature_cols)
        self.prob_threshold = prob_threshold
        self.enabled = enabled

        self.model = None

        if self.enabled:
            self.load_model()

    def load_model(self):
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)

    def score(self, features):
        if not self.enabled:
            return {
                "enabled": False,
                "prob": 1.0,
                "passed": True,
                "missing_features": [],
            }

        missing = [c for c in self.feature_cols if c not in features]

        if missing:
            return {
                "enabled": True,
                "prob": None,
                "passed": False,
                "missing_features": missing,
            }

        row = {c: features[c] for c in self.feature_cols}

        X = pd.DataFrame([row], columns=self.feature_cols)
        X = X.replace([np.inf, -np.inf], np.nan)

        if X.isna().any(axis=None):
            bad_cols = X.columns[X.isna().any()].tolist()

            return {
                "enabled": True,
                "prob": None,
                "passed": False,
                "missing_features": bad_cols,
            }

        prob = float(self.model.predict_proba(X)[0, 1])

        return {
            "enabled": True,
            "prob": prob,
            "passed": prob >= self.prob_threshold,
            "missing_features": [],
        }
