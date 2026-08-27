"""Describe the supported static subset of a MergeKit configuration."""

import sys

from mergelens import diagnose_config


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python examples/mergekit_diagnosis.py <config.yaml>")
        raise SystemExit(1)
    result = diagnose_config(sys.argv[1])
    print(f"Analysis status: {result.analysis_status}")
    print(f"Descriptive static proxy: {result.overall_interference:.3f}")
    print(f"Honoured: {', '.join(result.honored_features) or 'none'}")
    print(f"Ignored: {', '.join(result.ignored_features) or 'none'}")
    print(f"Unsupported: {', '.join(result.unsupported_features) or 'none'}")
    if result.interference_scores:
        print("Highest tensor inspection priorities:")
        for item in sorted(result.interference_scores, key=lambda value: value.score, reverse=True)[
            :5
        ]:
            print(f"  {item.tensor_name}: {item.score:.3f}")


if __name__ == "__main__":
    main()
