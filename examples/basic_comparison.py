"""Compare two real checkpoints while preserving evidence boundaries."""

import sys

from mergelens import compare_models


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python examples/basic_comparison.py <reference> <candidate>")
        raise SystemExit(1)
    result = compare_models(sys.argv[1:])
    print(f"Static-risk heuristic: {result.mci.score}")
    print(f"Risk tier: {result.mci.risk_tier}")
    print(f"Validation status: {result.mci.validation_status}")
    print(f"Exact comparable tensors: {result.coverage[0].exact_shape_compatible_tensor_count}")
    for signal in result.metric_availability:
        print(f"{signal.metric}: {signal.status.value} - {signal.reason or 'available'}")


if __name__ == "__main__":
    main()
