"""Shared fixtures for the evaluation tests.

Several test modules need the same thing: a short run of notes with known pitches and a regular
rhythm. They had each grown their own builder with slightly different parameter names, which made
the tests read as though they were constructing different kinds of input when they were not.
"""
import pytest

from musemotion.music.tokenizer import MidiNote


def build_clip(pitches, step=0.25, velocity=80, start=0.0):
    """A clip with one note per pitch, each lasting exactly one ``step``.

    Regular by construction, so rhythm-derived features have known values: note density is
    ``1 / step``, mean duration is ``step``, and the inter-onset interval is ``step``.
    """
    return [
        MidiNote(
            pitch=int(pitch),
            start=start + index * step,
            end=start + (index + 1) * step,
            velocity=velocity,
        )
        for index, pitch in enumerate(pitches)
    ]


@pytest.fixture
def clip():
    """The clip builder, for tests that prefer a fixture to an import."""
    return build_clip
