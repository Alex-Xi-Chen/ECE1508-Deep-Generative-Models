from __future__ import annotations

from typing import Any

from musemotion.emotions import map_goemotion_labels_to_quadrant


def label_names_from_goemotions_features(features: Any) -> list[str]:
    label_feature = features["labels"].feature
    return list(label_feature.names)


def map_goemotions_example(example: dict[str, Any], label_names: list[str]) -> dict[str, Any] | None:
    original_labels = [label_names[index] for index in example["labels"]]
    quadrant = map_goemotion_labels_to_quadrant(original_labels)
    if quadrant is None:
        return None
    return {
        "text": example["text"],
        "label": quadrant.id,
        "quadrant": quadrant.name,
        "original_labels": original_labels,
    }
