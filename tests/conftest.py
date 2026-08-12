"""Shared fixtures and import setup for the evaluation tests.

Several test modules need the same thing: a short run of notes with known pitches and a regular
rhythm. They had each grown their own builder with slightly different parameter names, which made
the tests read as though they were constructing different kinds of input when they were not.

This module also puts ``scripts/`` on the import path, since it is not a package and is not on
pytest's ``pythonpath``. Nothing imported the figure code because of that, which is how a renamed
metrics field kept a live reader and broke the stage-attribution figure unnoticed.
"""
import sys
from pathlib import Path

import pytest

# Appended rather than inserted at the front: these module names are distinctive enough not to
# collide today, but putting a loose directory ahead of the standard library for the whole test
# session is a shadowing risk with no upside.
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

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
