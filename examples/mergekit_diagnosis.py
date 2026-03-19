"""Diagnose a MergeKit YAML config before merging.

Usage:
    python examples/mergekit_diagnosis.py path/to/merge.yaml
"""
import sys

from mergelens import diagnose_config


def main():
    if len(sys.argv) < 2:
        print("Usage: python mergekit_diagnosis.py <config.yaml>")
        sys.exit(1)

    result = diagnose_config(sys.argv[1])
    print(f"Overall interference: {result.overall_interference:.3f}")

    if result.recommendations:
        print("\nRecommendations:")
        for r in result.recommendations:
            print(f"  - {r}")

    if result.interference_scores:
        print(f"\nTop interference layers ({len(result.interference_scores)} total):")
        top = sorted(result.interference_scores, key=lambda s: s.score, reverse=True)[:5]
        for s in top:
            print(f"  {s.layer_name}: {s.score:.3f}")

if __name__ == "__main__":
    main()
