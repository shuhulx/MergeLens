"""MergeLens — Pre-merge diagnostic framework for LLM model merging.

Usage:
    from mergelens import compare_models, diagnose_config, generate_report

    result = compare_models(["model_a", "model_b"])
    print(result.mci.score)  # 0-100 compatibility score
"""

from importlib import metadata

try:
    __version__ = metadata.version("mergelens")
except metadata.PackageNotFoundError:
    __version__ = "0.1.5"

from mergelens.compare import compare_models
from mergelens.diagnose import diagnose_config
from mergelens.report import generate_report

__all__ = [
    "__version__",
    "compare_models",
    "diagnose_config",
    "generate_report",
]
