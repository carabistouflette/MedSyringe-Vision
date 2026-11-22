"""Serialize landmark counts from the frozen detector comparison."""
from unittest.mock import patch

from scripts.experiments.parts import part_verifier_study


def integer_counts(audit):
    return {part: {metric: int(value) for metric, value in field.items()} for part, field in audit.items()}


def main():
    original = part_verifier_study.landmark_audit
    with patch.object(part_verifier_study, "landmark_audit",
                      side_effect=lambda rows, evidence: integer_counts(original(rows, evidence))):
        part_verifier_study.measure()


if __name__ == "__main__":
    main()
