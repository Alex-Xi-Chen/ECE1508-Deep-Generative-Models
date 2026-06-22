from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EmotionQuadrant:
    id: int
    name: str
    valence: str
    arousal: str
    description: str


EMOPIA_QUADRANTS: tuple[EmotionQuadrant, ...] = (
    EmotionQuadrant(0, "Q1", "high", "high", "positive and energetic"),
    EmotionQuadrant(1, "Q2", "low", "high", "negative and energetic"),
    EmotionQuadrant(2, "Q3", "low", "low", "negative and subdued"),
    EmotionQuadrant(3, "Q4", "high", "low", "positive and calm"),
)

_QUADRANTS_BY_NAME = {quadrant.name: quadrant for quadrant in EMOPIA_QUADRANTS}

GOEMOTIONS_TO_QUADRANT: dict[str, str] = {
    "admiration": "Q1",
    "amusement": "Q1",
    "anger": "Q2",
    "annoyance": "Q2",
    "approval": "Q1",
    "caring": "Q4",
    "confusion": "Q4",
    "curiosity": "Q4",
    "desire": "Q4",
    "disappointment": "Q3",
    "disapproval": "Q2",
    "disgust": "Q2",
    "embarrassment": "Q2",
    "excitement": "Q1",
    "fear": "Q2",
    "gratitude": "Q1",
    "grief": "Q3",
    "joy": "Q1",
    "love": "Q4",
    "nervousness": "Q2",
    "neutral": "Q4",
    "optimism": "Q1",
    "pride": "Q1",
    "realization": "Q4",
    "relief": "Q1",
    "remorse": "Q3",
    "sadness": "Q3",
    "surprise": "Q4",
}


def quadrant_id(name: str) -> int:
    try:
        return _QUADRANTS_BY_NAME[name].id
    except KeyError as exc:
        raise ValueError(f"Unknown EMOPIA quadrant: {name}") from exc


def quadrant_name(emotion_id: int) -> str:
    try:
        return EMOPIA_QUADRANTS[emotion_id].name
    except IndexError as exc:
        raise ValueError(f"Unknown EMOPIA emotion id: {emotion_id}") from exc


def map_goemotion_labels_to_quadrant(labels: Iterable[str]) -> EmotionQuadrant | None:
    votes = Counter()
    for raw_label in labels:
        label = str(raw_label).strip().lower()
        quadrant_name_value = GOEMOTIONS_TO_QUADRANT.get(label)
        if quadrant_name_value is not None:
            votes[quadrant_name_value] += 1

    if not votes:
        return None

    [(winner, count)] = votes.most_common(1)
    tied = [name for name, value in votes.items() if value == count]
    if len(tied) > 1:
        return None
    return _QUADRANTS_BY_NAME[winner]
