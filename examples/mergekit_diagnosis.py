"""Diagnose a MergeKit YAML config before merging.

Usage:
    python examples/mergekit_diagnosis.py path/to/merge.yaml
"""
import sys

from mergelens import diagnose


def main():
    if len(sys.argv) < 2:
        print("Usage: python mergekit_diagnosis.py <config.yaml>")
        sys.exit(1)

    result = diagnose(sys.argv[1])
    print(f"Config quality: {result.quality}")
    print(f"Interference score: {result.interference_score:.3f}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

if __name__ == "__main__":
    main()
