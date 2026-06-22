from musemotion.emotions import (
    EMOPIA_QUADRANTS,
    map_goemotion_labels_to_quadrant,
    quadrant_id,
)


def test_quadrant_ids_are_stable():
    assert [q.name for q in EMOPIA_QUADRANTS] == ["Q1", "Q2", "Q3", "Q4"]
    assert quadrant_id("Q3") == 2


def test_single_goemotion_label_maps_to_expected_quadrant():
    result = map_goemotion_labels_to_quadrant(["joy"])

    assert result is not None
    assert result.name == "Q1"


def test_conflicting_goemotion_labels_are_dropped():
    assert map_goemotion_labels_to_quadrant(["joy", "grief"]) is None


def test_majority_goemotion_labels_pick_unique_quadrant():
    result = map_goemotion_labels_to_quadrant(["joy", "excitement", "sadness"])

    assert result is not None
    assert result.name == "Q1"
