"""Exercise every committed artifact through the loader that actually reads it.

The lesson these encode: when a schema changes, the files already on disk are the stale party.
Tests that pass against them prove nothing about the next run, and a reader left behind by a
rename keeps working on old artifacts while failing on everything produced afterwards - which is
exactly how the stage-attribution figure broke.

So these do not parse the files as plain JSON and check keys. They run them through the real
loaders and the real scorers, and the strongest of them recomputes a recorded number from
scratch: if the artifact, the loader, the feature extractor, or the metric drift apart, the
recomputation stops matching what was recorded.
"""
import json

import numpy as np
import pytest

from musemotion.config import REPO_ROOT, load_yaml_config, resolve_path
from musemotion.evaluation.probe import load_clip_splits, load_probe_metadata, load_probes
from musemotion.music.tokenizer import MusicTokenizer


def _paths():
    config = load_yaml_config("configs/evaluation.yaml")
    return (
        resolve_path(config["probes"]["directory"]),
        resolve_path(config["data"]["tokenized_dir"]),
        resolve_path(config["generator"]["tokenizer"]),
    )


@pytest.fixture(scope="module")
def committed():
    probe_dir, tokenized_dir, tokenizer_path = _paths()
    if not (probe_dir / "probe_metadata.json").exists() or not tokenized_dir.exists():
        pytest.skip("committed probe or tokenized artifacts not present")
    tokenizer = MusicTokenizer.load(tokenizer_path)
    metadata = load_probe_metadata(probe_dir)
    splits = load_clip_splits(tokenized_dir, tokenizer, int(metadata["note_budget"]))
    return {
        "probes": load_probes(probe_dir, device="cpu", tokenizer_path=tokenizer_path),
        "metadata": metadata,
        "splits": splits,
    }


def test_both_committed_probes_load_and_score(committed):
    probes = committed["probes"]
    assert set(probes) == {"feature", "neural"}

    clips = [row["notes"] for row in committed["splits"]["test"][:8]]
    for name, probe in probes.items():
        probabilities = probe.predict_proba_batch(clips)
        assert probabilities.shape == (len(clips), 4), name
        assert np.allclose(probabilities.sum(axis=1), 1.0), name


def test_committed_probes_reproduce_their_recorded_test_accuracy(committed):
    """The end-to-end contract: artifact, loader, scorer and recorded metric must still agree.

    This is the number every round-trip accuracy in a run is divided by, so a silent drift here
    would move every reported result without any of them looking wrong.
    """
    splits, metadata = committed["splits"], committed["metadata"]
    truth = np.asarray([row["emotion_id"] for row in splits["test"]], dtype="int64")
    clips = [row["notes"] for row in splits["test"]]

    for name, probe in committed["probes"].items():
        predicted = np.argmax(probe.predict_proba_batch(clips), axis=-1)
        recomputed = float((predicted == truth).mean())
        recorded = metadata["probes"][name]["test"]["accuracy"]
        assert recomputed == pytest.approx(recorded, abs=1e-9), (
            f"{name} probe scores {recomputed:.4f} on the committed test split but "
            f"probe_metadata.json records {recorded:.4f}"
        )


def test_committed_metadata_carries_every_field_its_readers_use(committed):
    """Fields consumed by the harness and the figures, checked against the shipped file."""
    metadata = committed["metadata"]

    assert metadata["note_budget"] > 0
    assert metadata["chance_accuracy"] == pytest.approx(0.25)
    assert metadata["majority_class_accuracy"] > metadata["chance_accuracy"]
    for name, values in metadata["probes"].items():
        assert values["test"]["accuracy"] > metadata["majority_class_accuracy"], name
        control = values["shuffled_label_control"]["test"]
        assert control["accuracy"] <= metadata["majority_class_accuracy"] + 0.02, name


def test_committed_tokenized_splits_load_at_the_probes_note_budget(committed):
    splits, budget = committed["splits"], committed["metadata"]["note_budget"]

    assert set(splits) == {"train", "validation", "test"}
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [862, 108, 108]
    for row in splits["train"][:20]:
        assert 0 < len(row["notes"]) <= budget
        # Re-encoded rather than truncated: BOS, whole four-token groups, EOS.
        assert len(row["token_ids"]) == len(row["notes"]) * 4 + 2
        assert 0 <= row["emotion_id"] < 4


def test_committed_source_paths_are_relative_not_machine_specific(committed):
    """Provenance is kept; the layout of the machine that produced it is not."""
    for row in committed["splits"]["train"][:50]:
        source = row["source"]
        assert source and not source.startswith(("/", "C:", "D:")), source
        assert "/content/" not in source


# README table label -> metrics.json system key. The README names rows for a reader; the harness
# names them for a machine.
README_ROWS = {
    "**real EMOPIA (ceiling)**": "real",
    "guidance 1.0 (CFG off)": "guidance=1",
    "guidance 2.0": "guidance=2",
    "**guidance 3.0 (shipped)**": "guidance=3",
    "guidance 5.0": "guidance=5",
    "end-to-end (text → BERT → gen)": "end_to_end",
}


def _readme_results_table():
    """Parse the results table out of the README into {label: [cells]}."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in README_ROWS:
            rows[cells[0]] = cells
    return rows


def test_the_readme_results_table_matches_the_recorded_run():
    """The headline numbers must be checkable, not transcribed.

    Without the run's metrics.json committed, every number in the README results table is
    reproducible from nothing in the repository and verified by nothing - which is how the probe
    documentation drifted to stale control values before anyone noticed. Run the notebook's
    save-results cell to record a run, and this becomes a real check.
    """
    recorded = REPO_ROOT / "models/real_training/evaluation/metrics.json"
    if not recorded.exists():
        pytest.skip(
            "no committed evaluation run; the README results table is unverified. "
            "Run the save-results cell in notebook Step 9a to record one."
        )
    metrics = json.loads(recorded.read_text(encoding="utf-8"))
    table = _readme_results_table()
    # Every expected row must have matched. Iterating only what parsed would let a relabelled row
    # drop out of the mapping and be checked by nothing while the test stayed green - the same
    # shape as the rename that disconnected the stage-attribution figure.
    assert set(table) == set(README_ROWS), (
        f"README rows did not match the expected labels: {set(README_ROWS) ^ set(table)}"
    )

    for label, cells in table.items():
        system = metrics["systems"][README_ROWS[label]]
        # Columns: label | n | round-trip | CI | arousal | valence | vs ceiling | ...
        assert int(cells[1]) == system["round_trip"]["overall"]["count"], label
        quoted = float(cells[2].strip("*"))
        assert quoted == pytest.approx(
            system["round_trip"]["overall"]["accuracy"], abs=5e-4
        ), f"{label}: README says {quoted}"
        for column, axis in ((4, "arousal"), (5, "valence")):
            assert float(cells[column]) == pytest.approx(
                system["round_trip"]["axes"][axis]["accuracy"], abs=5e-4
            ), f"{label} {axis}"
        # The interval column carries the "non-overlapping" argument, so both endpoints are
        # checked rather than left as prose. The separator is an en dash in the README.
        low, high = (float(part) for part in cells[3].replace("–", "-").split("-"))
        overall = system["round_trip"]["overall"]
        assert low == pytest.approx(overall["ci_low"], abs=5e-4), f"{label} ci_low"
        assert high == pytest.approx(overall["ci_high"], abs=5e-4), f"{label} ci_high"


def test_the_recorded_run_is_the_one_the_readme_describes():
    """A 6-clip smoke run must not be mistaken for the 200-clip result the README reports."""
    recorded = REPO_ROOT / "models/real_training/evaluation/metrics.json"
    if not recorded.exists():
        pytest.skip("no committed evaluation run")
    generation = json.loads(recorded.read_text(encoding="utf-8"))["run"]["config"]["generation"]

    assert generation["clips_per_quadrant"] >= 50, (
        "the committed run is too small to support the README's intervals"
    )
    assert [float(v) for v in generation["guidance_scales"]] == [1.0, 2.0, 3.0, 5.0]


def _valence_flow(confusion):
    """(positive -> negative, negative -> positive) counts from a confusion matrix."""
    from musemotion.emotions import EMOPIA_QUADRANTS

    high_valence = {q.id for q in EMOPIA_QUADRANTS if q.valence == "high"}
    positive_to_negative = sum(
        confusion[i][j] for i in range(4) for j in range(4)
        if i in high_valence and j not in high_valence
    )
    negative_to_positive = sum(
        confusion[i][j] for i in range(4) for j in range(4)
        if i not in high_valence and j in high_valence
    )
    return positive_to_negative, negative_to_positive


def test_the_directional_negativity_finding_holds_in_the_recorded_run():
    """Puts a test under the finding, not just under its precondition.

    "The generator renders calm as melancholy rather than content" rests on the valence errors
    being lopsided. The error *total* is already implied by the per-row assertions; the split
    between the two directions is the part the claim actually depends on, and it is derivable
    from the confusion matrix that is already in the payload.
    """
    recorded = REPO_ROOT / "models/real_training/evaluation/metrics.json"
    if not recorded.exists():
        pytest.skip("no committed evaluation run")
    metrics = json.loads(recorded.read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    confusion = metrics["systems"]["guidance=3"]["round_trip"]["confusion_matrix"]
    positive_to_negative, negative_to_positive = _valence_flow(confusion)

    assert positive_to_negative > negative_to_positive, (
        "the README claims valence errors run mostly positive-to-negative, but the recorded run "
        f"shows {positive_to_negative} against {negative_to_positive}"
    )
    # And the specific counts the README prints must be the ones in the run.
    assert f"{positive_to_negative} run positive" in readme or (
        f"**{positive_to_negative} run positive" in readme
    ), f"README does not quote {positive_to_negative} positive-to-negative errors"
    assert f"{negative_to_positive} the other way" in readme, (
        f"README does not quote {negative_to_positive} negative-to-positive errors"
    )


def test_the_non_overlapping_interval_claim_holds():
    """The README argues CFG works because two intervals do not overlap. Check that they don't."""
    recorded = REPO_ROOT / "models/real_training/evaluation/metrics.json"
    if not recorded.exists():
        pytest.skip("no committed evaluation run")
    systems = json.loads(recorded.read_text(encoding="utf-8"))["systems"]

    off = systems["guidance=1"]["round_trip"]["overall"]
    shipped = systems["guidance=3"]["round_trip"]["overall"]

    assert off["ci_high"] < shipped["ci_low"], (
        f"the README claims these intervals do not overlap, but guidance=1 reaches "
        f"{off['ci_high']:.3f} and guidance=3 starts at {shipped['ci_low']:.3f}"
    )
