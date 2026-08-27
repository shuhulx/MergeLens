"""MergeLens public API for static checkpoint inspection."""

from __future__ import annotations

from typing import Any

from mergelens.__about__ import __version__
from mergelens.compare import compare_models
from mergelens.diagnose import diagnose_config


def generate_report(*args: Any, **kwargs: Any) -> str:
    """Generate an HTML report, importing optional dependencies only when used."""
    from mergelens.report import generate_report as _generate_report

    return _generate_report(*args, **kwargs)


__all__ = ["__version__", "compare_models", "diagnose_config", "generate_report"]
