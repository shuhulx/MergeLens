"""Linear CKA over explicitly aligned calibration activations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from mergelens.activations.extractor import ActivationSet
from mergelens.compare.metrics import cka_similarity


@dataclass(frozen=True)
class CKAComparison(Mapping[str, float]):
    """CKA scores with calibration and layer-alignment provenance."""

    scores: dict[str, float]
    calibration_id: str
    sample_count: int
    aligned_layers: tuple[str, ...]
    pooling_rule: str

    def __getitem__(self, key: str) -> float:
        return self.scores[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.scores)

    def __len__(self) -> int:
        return len(self.scores)


def compare_activations_cka(
    activations_a: ActivationSet,
    activations_b: ActivationSet,
) -> CKAComparison:
    """Compute CKA for exact-name layers over the same calibration samples.

    Feature widths may differ. Sample counts and calibration identities may not.
    """

    if activations_a.calibration_id != activations_b.calibration_id:
        raise ValueError(
            "Calibration identity mismatch; CKA requires activations from the same texts and rule."
        )
    if activations_a.sample_count != activations_b.sample_count:
        raise ValueError(
            f"Sample count mismatch: {activations_a.sample_count} vs "
            f"{activations_b.sample_count} samples"
        )
    if activations_a.pooling_rule != activations_b.pooling_rule:
        raise ValueError(
            f"Pooling rule mismatch: {activations_a.pooling_rule!r} vs "
            f"{activations_b.pooling_rule!r}"
        )

    common_layers = sorted(set(activations_a.activations) & set(activations_b.activations))
    scores: dict[str, float] = {}
    for layer in common_layers:
        first = activations_a.activations[layer]
        second = activations_b.activations[layer]
        if first.shape[0] != second.shape[0]:
            raise ValueError(
                f"Sample count mismatch for {layer}: {first.shape[0]} vs {second.shape[0]}"
            )
        scores[layer] = round(cka_similarity(first, second), 6)

    return CKAComparison(
        scores=scores,
        calibration_id=activations_a.calibration_id,
        sample_count=activations_a.sample_count,
        aligned_layers=tuple(common_layers),
        pooling_rule=activations_a.pooling_rule,
    )
