"""Interference scoring — measures how much models conflict during merging."""

from __future__ import annotations

import torch

from mergelens.compare.loader import ModelHandle, find_common_tensors
from mergelens.compare.metrics import cosine_similarity
from mergelens.models import InterferenceScore


def compute_interference(
    handles: list[ModelHandle],
    base_model: str | None = None,
    device: str = "cpu",
) -> list[InterferenceScore]:
    """Compute per-layer interference scores.

    Interference = 1 - cosine_sim(merged_layer, weighted_avg_of_sources)
    Measures how much the merge deviates from simple averaging.

    For pre-merge prediction: uses task vector overlap to predict problematic layers.
    """
    if len(handles) < 2:
        return []

    common = find_common_tensors(handles)
    scores = []

    for name in common:
        tensors = [h.get_tensor(name) for h in handles]

        # Compute weighted average (equal weights)
        avg = torch.stack([t.float() for t in tensors]).mean(dim=0)

        # Interference per source
        contributions = {}
        for _i, (h, t) in enumerate(zip(handles, tensors)):
            cos = cosine_similarity(t, avg)
            contributions[h.info.name] = round(cos, 4)

        # Task vector interference
        base_t = tensors[0].float()
        task_vecs = [t.float() - base_t for t in tensors[1:]]

        if len(task_vecs) >= 2 and task_vecs[0].numel() > 0:
            # Average pairwise task vector cosine similarity
            pair_count = 0
            tv_cos_sum = 0.0
            for i in range(len(task_vecs)):
                for j in range(i + 1, len(task_vecs)):
                    tv_cos_sum += cosine_similarity(task_vecs[i], task_vecs[j])
                    pair_count += 1
            tv_cos_avg = tv_cos_sum / pair_count
            # High similarity = low interference, negative = high interference
            interference = max(0.0, min(1.0, (1.0 - tv_cos_avg) / 2.0))
        else:
            # Single task vector: use magnitude as proxy
            interference = 0.0
            for tv in task_vecs:
                norm = torch.norm(tv).item()
                base_norm = torch.norm(base_t).item()
                if base_norm > 1e-10:
                    interference = max(interference, min(1.0, norm / base_norm))

        scores.append(
            InterferenceScore(
                layer_name=name,
                score=round(float(interference), 4),
                source_contributions=contributions,
            )
        )

    return scores
