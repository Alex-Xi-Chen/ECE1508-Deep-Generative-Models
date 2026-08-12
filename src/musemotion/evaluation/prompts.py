"""The fixed prompt set for the end-to-end text-to-music measurement.

Forty sentences, ten per quadrant, each carrying the quadrant I intend it to express. The
set is committed as a constant and fixed before any results are inspected, so the end-to-end
number cannot drift by quietly swapping in sentences that happen to score well.

The intended labels are my own judgements about what these sentences express, not
GoEmotions annotations. That distinction matters when reading the results: a sentence the
classifier "gets wrong" may instead be a case where my intent and the GoEmotions mapping
genuinely disagree, which is a finding about the label mapping rather than about the model.

Sentences are deliberately plain first-person statements of feeling, matching how someone
would actually use the app, and they avoid naming the emotion words the classifier was
trained on wherever a natural phrasing allows it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from musemotion.emotions import EMOPIA_QUADRANTS, quadrant_id


@dataclass(frozen=True)
class EvaluationPrompt:
    """One sentence and the quadrant it is meant to express."""

    text: str
    intended_quadrant: str

    @property
    def intended_emotion_id(self) -> int:
        return quadrant_id(self.intended_quadrant)


# Q1: high valence, high arousal - positive and energetic.
_Q1 = (
    "I just got the job offer and I cannot stop grinning",
    "I am so excited about the trip next week",
    "Everything is finally coming together and I feel unstoppable",
    "I am thrilled with how the presentation went today",
    "I feel proud of what our team pulled off this month",
    "I am buzzing with energy and ready to take on anything",
    "This is the best news I have heard all year",
    "I feel optimistic and full of momentum right now",
    "I am delighted that my friends surprised me like that",
    "I feel fantastic and I want to celebrate with everyone",
)

# Q2: low valence, high arousal - negative and energetic.
_Q2 = (
    "I am furious about how they treated me in that meeting",
    "I feel panicked and my heart will not slow down",
    "I am so frustrated that I want to shout",
    "I feel anxious and on edge about tomorrow",
    "I am disgusted by what I just read",
    "I feel threatened and I cannot calm myself down",
    "I am irritated by every little thing today",
    "I feel scared and completely wound up",
    "I resent being blamed for something I did not do",
    "I am agitated and cannot sit still",
)

# Q3: low valence, low arousal - negative and subdued.
_Q3 = (
    "I feel hollow and I do not want to get out of bed",
    "I am quietly heartbroken about how it ended",
    "I feel like I have let everyone down",
    "I am tired of trying and nothing ever changes",
    "I miss them and the house feels empty",
    "I feel numb and disconnected from everything",
    "I am disappointed in myself tonight",
    "I feel a heavy sadness I cannot explain",
    "I regret the way I handled all of it",
    "I feel lonely and worn down",
)

# Q4: high valence, low arousal - positive and calm.
_Q4 = (
    "I feel calm and grateful today",
    "I am at peace sitting here in the quiet",
    "I feel warm and content with the people around me",
    "I am relieved that it is finally over",
    "I feel loved and safe tonight",
    "I am thankful for the small things this morning",
    "I feel settled and gently hopeful",
    "I care about them deeply and it feels steady",
    "I feel serene watching the rain outside",
    "I am quietly happy and unhurried",
)

EVALUATION_PROMPTS: tuple[EvaluationPrompt, ...] = tuple(
    EvaluationPrompt(text=text, intended_quadrant=quadrant)
    for quadrant, sentences in (("Q1", _Q1), ("Q2", _Q2), ("Q3", _Q3), ("Q4", _Q4))
    for text in sentences
)


def prompts_for_quadrant(quadrant: str) -> tuple[EvaluationPrompt, ...]:
    return tuple(prompt for prompt in EVALUATION_PROMPTS if prompt.intended_quadrant == quadrant)


def prompt_counts() -> dict[str, int]:
    return {
        quadrant.name: len(prompts_for_quadrant(quadrant.name)) for quadrant in EMOPIA_QUADRANTS
    }


def limited_prompts(limit: int | None = None) -> tuple[EvaluationPrompt, ...]:
    """A balanced subset, taking the same number from each quadrant.

    Used by the capped smoke configuration. Trimming per quadrant rather than truncating the
    flat list keeps the subset balanced, which the accuracy decomposition relies on.
    """
    if limit is None or limit >= len(EVALUATION_PROMPTS):
        return EVALUATION_PROMPTS
    per_quadrant = max(1, limit // len(EMOPIA_QUADRANTS))
    subset: list[EvaluationPrompt] = []
    for quadrant in EMOPIA_QUADRANTS:
        subset.extend(prompts_for_quadrant(quadrant.name)[:per_quadrant])
    return tuple(subset)


def as_records(prompts: Sequence[EvaluationPrompt] = EVALUATION_PROMPTS) -> list[dict[str, str]]:
    return [
        {"text": prompt.text, "intended_quadrant": prompt.intended_quadrant} for prompt in prompts
    ]
