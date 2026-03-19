"""Basic model comparison example.

Usage:
    python examples/basic_comparison.py model_a_path model_b_path
"""
import sys

from mergelens import compare_models


def main():
    if len(sys.argv) < 3:
        print("Usage: python basic_comparison.py <model_a> <model_b>")
        sys.exit(1)

    result = compare_models([sys.argv[1], sys.argv[2]])
    print(f"Merge Compatibility Index: {result.mci.score}/100")
    print(f"Verdict: {result.mci.verdict}")
    print(f"Conflict zones: {len(result.conflict_zones)}")

    if result.strategy:
        print(f"\nRecommended method: {result.strategy.method}")
        print(f"Confidence: {result.strategy.confidence:.0%}")

if __name__ == "__main__":
    main()
