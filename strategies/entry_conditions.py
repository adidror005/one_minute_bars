import operator

import pandas as pd


OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class EntryConditionEvaluator:
    def __init__(self, first_entry_rules, double_down_rules):
        self.first_entry_rules = list(first_entry_rules)
        self.double_down_rules = list(double_down_rules)

    @classmethod
    def from_config(cls, strategy_cfg, double_down_cfg):
        entry_cfg = strategy_cfg.get("entry_conditions", {})

        if entry_cfg.get("first_entry") or entry_cfg.get("double_down"):
            default_first, default_double_down = build_entry_strategy_rules(
                strategy_cfg=strategy_cfg,
                double_down_cfg=double_down_cfg,
            )
            first_entry_rules = entry_cfg.get("first_entry") or default_first
            double_down_rules = entry_cfg.get("double_down") or default_double_down
        else:
            first_entry_rules, double_down_rules = build_entry_strategy_rules(
                strategy_cfg=strategy_cfg,
                double_down_cfg=double_down_cfg,
            )

        return cls(
            first_entry_rules=first_entry_rules,
            double_down_rules=double_down_rules,
        )

    def evaluate(self, entry_type, features):
        rules = (
            self.first_entry_rules
            if entry_type == "first_entry"
            else self.double_down_rules
        )

        reasons = []
        debug = []

        for rule in rules:
            passed, reason, detail = self.evaluate_rule(rule, features)
            debug.append(detail)

            if not passed:
                reasons.append(reason)

        return reasons, debug

    def evaluate_rule(self, rule, features):
        field = rule["field"]
        op = rule.get("op", "<=")
        expected = rule.get("value")
        actual = features.get(field)

        detail = {
            "field": field,
            "op": op,
            "actual": actual,
            "expected": expected,
            "passed": False,
        }

        if field not in features:
            return False, f"entry_condition_missing_{field}", detail

        if actual is None or pd.isna(actual):
            return False, f"entry_condition_nan_{field}", detail

        if op not in OPS:
            return False, f"entry_condition_unsupported_op_{field}_{op}", detail

        passed = OPS[op](actual, expected)
        detail["passed"] = passed

        if not passed:
            return False, f"entry_condition_failed_{field}_{op}_{expected}", detail

        return True, None, detail

    def format_debug(self, debug):
        parts = []

        for item in debug:
            actual = item["actual"]
            if isinstance(actual, float):
                actual_text = f"{actual:.4f}"
            else:
                actual_text = str(actual)

            parts.append(
                f"{item['field']} {item['op']} {item['expected']} "
                f"actual={actual_text} passed={item['passed']}"
            )

        return "[" + "; ".join(parts) + "]"


def build_entry_strategy_rules(strategy_cfg, double_down_cfg):
    strategy = strategy_cfg.get("entry_strategy", {})
    name = strategy.get("name", "median_mad")

    if name == "median_mad":
        return median_mad_rules(strategy_cfg, double_down_cfg, strategy)

    if name == "standard_bb":
        return standard_bb_rules(strategy_cfg, double_down_cfg, strategy)

    if name == "always_true":
        return always_true_rules()

    raise ValueError(f"Unsupported entry_strategy.name: {name}")


def median_mad_rules(strategy_cfg, double_down_cfg, strategy):
    k = float(strategy.get("k", strategy_cfg.get("k", 2)))
    double_down_mult = float(strategy.get("double_down_mult", double_down_cfg.get("bb_mult", 3)))

    return (
        [{"field": "bb_score", "op": "<=", "value": -k}],
        [{"field": "bb_score", "op": "<=", "value": -(double_down_mult * k)}],
    )


def standard_bb_rules(strategy_cfg, double_down_cfg, strategy):
    z = float(strategy.get("z", strategy_cfg.get("k", 2)))
    double_down_mult = float(strategy.get("double_down_mult", double_down_cfg.get("bb_mult", 3)))

    return (
        [{"field": "standard_bb_z", "op": "<=", "value": -z}],
        [{"field": "standard_bb_z", "op": "<=", "value": -(double_down_mult * z)}],
    )


def always_true_rules():
    return ([], [])
