"""
Rule Engine
-----------
Evaluates a FlowFeatures object against a set of signature rules.
Rules are defined in config/rules.yaml — no code changes needed
to add or modify rules.

Each rule produces a RuleResult with:
  - matched   : bool
  - rule_id   : str
  - rule_name : str
  - severity  : LOW / MEDIUM / HIGH / CRITICAL
  - confidence: 0.0 – 1.0
  - details   : human-readable explanation
"""

import yaml
import os
from dataclasses import dataclass
from typing import List, Optional
from engine.feature_extractor import FlowFeatures


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    matched: bool
    rule_id: str
    rule_name: str
    severity: str           # LOW | MEDIUM | HIGH | CRITICAL
    confidence: float       # 0.0 – 1.0
    details: str
    category: str           # e.g. DoS, Probe, Brute-Force


@dataclass
class RuleEngineOutput:
    any_match: bool
    results: List[RuleResult]
    highest_severity: str
    max_confidence: float

    @property
    def matched_rules(self) -> List[RuleResult]:
        return [r for r in self.results if r.matched]


# ── Severity → numeric weight ─────────────────────────────────────────────────

SEVERITY_WEIGHT = {
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.75,
    "CRITICAL": 1.00,
}

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ── Rule Engine ───────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Usage
    -----
    engine = RuleEngine("config/rules.yaml")
    output = engine.evaluate(flow_features)
    """

    def __init__(self, rules_path: str = "config/rules.yaml"):
        self.rules_path = rules_path
        self.rules = self._load_rules()
        print(f"[RuleEngine] Loaded {len(self.rules)} rules from {rules_path}")

    def _load_rules(self) -> list:
        with open(self.rules_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("rules", [])

    def evaluate(self, ff: FlowFeatures) -> RuleEngineOutput:
        results = []
        for rule in self.rules:
            result = self._check_rule(rule, ff)
            results.append(result)

        matched = [r for r in results if r.matched]
        any_match = len(matched) > 0

        highest_severity = "LOW"
        max_confidence = 0.0
        for r in matched:
            if SEVERITY_ORDER.index(r.severity) > SEVERITY_ORDER.index(highest_severity):
                highest_severity = r.severity
            if r.confidence > max_confidence:
                max_confidence = r.confidence

        return RuleEngineOutput(
            any_match=any_match,
            results=results,
            highest_severity=highest_severity if any_match else "NONE",
            max_confidence=max_confidence,
        )

    def _check_rule(self, rule: dict, ff: FlowFeatures) -> RuleResult:
        """Evaluate a single rule against flow features."""
        rule_id   = rule.get("id", "unknown")
        rule_name = rule.get("name", "Unnamed Rule")
        severity  = rule.get("severity", "LOW")
        confidence= float(rule.get("confidence", 0.8))
        category  = rule.get("category", "General")
        conditions= rule.get("conditions", [])

        matched, details = self._evaluate_conditions(conditions, ff)

        return RuleResult(
            matched=matched,
            rule_id=rule_id,
            rule_name=rule_name,
            severity=severity,
            confidence=confidence if matched else 0.0,
            details=details if matched else "",
            category=category,
        )

    def _evaluate_conditions(self, conditions: list, ff: FlowFeatures):
        """All conditions must pass (AND logic)."""
        triggered = []
        raw = ff.raw

        for cond in conditions:
            field  = cond.get("field")
            op     = cond.get("op")
            value  = cond.get("value")

            actual = raw.get(field, 0)

            passed = self._compare(actual, op, value)
            if not passed:
                return False, ""
            triggered.append(f"{field} {op} {value} (actual: {actual})")

        return True, "; ".join(triggered)

    def _compare(self, actual, op: str, value) -> bool:
        try:
            actual = float(actual)
            value_f = float(value)
        except (TypeError, ValueError):
            # String comparison
            return str(actual).upper() == str(value).upper()

        ops = {
            ">":  actual > value_f,
            ">=": actual >= value_f,
            "<":  actual < value_f,
            "<=": actual <= value_f,
            "==": actual == value_f,
            "!=": actual != value_f,
        }
        return ops.get(op, False)
