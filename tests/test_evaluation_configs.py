"""Checks on the committed evaluation configs.

The failure these guard against is real and quiet: a config that points at a gitignored
`artifacts/` path works on the machine that generated it and fails on a fresh checkout or a
fresh Colab runtime, which is exactly where the notebook is meant to run.
"""
import json

import pytest
import yaml

from musemotion.config import REPO_ROOT, load_yaml_config, resolve_path

# Configs that must work on a fresh checkout, with only committed files present.
SELF_CONTAINED_CONFIGS = [
    "configs/evaluation.yaml",
    "configs/evaluation_smoke.yaml",
    "configs/probe.yaml",
    "configs/inference.yaml",
]
ALL_EVALUATION_CONFIGS = SELF_CONTAINED_CONFIGS


def test_configs_parse_and_are_mappings():
    for name in ALL_EVALUATION_CONFIGS:
        assert isinstance(load_yaml_config(name), dict)


def test_input_paths_are_committed_not_gitignored():
    """Every input a self-contained config reads must be outside the ignored trees."""
    ignored_roots = ("artifacts/", "data/", "checkpoints/", "output/")
    for name in SELF_CONTAINED_CONFIGS:
        config = load_yaml_config(name)
        candidates = [
            config.get("generator", {}).get("checkpoint"),
            config.get("generator", {}).get("tokenizer"),
            config.get("classifier", {}).get("model_dir"),
            config.get("probes", {}).get("directory"),
            config.get("data", {}).get("tokenized_dir"),
            config.get("data", {}).get("tokenizer"),
        ]
        for value in [item for item in candidates if item]:
            assert not str(value).startswith(ignored_roots), (
                f"{name} reads {value}, which is gitignored and absent on a fresh checkout"
            )


def test_output_paths_stay_inside_the_ignored_tree():
    """Runs must not write into the committed tree by default."""
    evaluation = load_yaml_config("configs/evaluation.yaml")
    assert evaluation["output"]["directory"].startswith("artifacts/")
    assert load_yaml_config("configs/probe.yaml")["training"]["output_dir"].startswith("artifacts/")


def test_tokenized_reference_data_is_present_and_labelled():
    tokenized = resolve_path(load_yaml_config("configs/evaluation.yaml")["data"]["tokenized_dir"])
    for split in ("train", "validation", "test"):
        path = tokenized / f"{split}.jsonl"
        assert path.exists(), f"missing committed split {path}"
    # CC BY 4.0 permits redistribution but requires credit, so the attribution must ship with it.
    readme = (tokenized / "README.md").read_text(encoding="utf-8")
    assert "Attribution 4.0" in readme
    assert "EMOPIA" in readme


def test_guidance_sweep_includes_the_no_guidance_baseline():
    """Without a guidance=1.0 row there is nothing to attribute the improvement to."""
    generation = load_yaml_config("configs/evaluation.yaml")["generation"]
    assert 1.0 in [float(value) for value in generation["guidance_scales"]]
    assert float(generation["guidance_scale"]) in [float(v) for v in generation["guidance_scales"]]


@pytest.mark.parametrize("name", ["configs/evaluation.yaml", "configs/evaluation_smoke.yaml"])
def test_note_budget_matches_the_token_cap(name):
    """Four tokens per note, in every evaluation config.

    A mismatch is silent and corrupting: real clips get cropped to note_budget while generated
    clips stop at max_tokens / 4, so every novelty, diversity, and fidelity comparison is then
    between clips of different lengths. This originally guarded only the full config, which is
    exactly why the smoke config drifted.
    """
    config = load_yaml_config(name)
    note_budget = int(config["data"]["note_budget"])
    assert int(config["generation"]["max_tokens"]) == note_budget * 4, name


def test_probe_note_budget_matches_the_evaluation_it_judges():
    """The probe must be trained on the same window the harness asks it about."""
    evaluation = load_yaml_config("configs/evaluation.yaml")
    probe = load_yaml_config("configs/probe.yaml")
    assert int(probe["data"]["note_budget"]) == int(evaluation["data"]["note_budget"])


def test_committed_probe_metadata_matches_the_probe_config():
    """The shipped artifacts must have been produced by the config that documents them."""
    probe_config = load_yaml_config("configs/probe.yaml")
    metadata_path = (
        resolve_path(load_yaml_config("configs/evaluation.yaml")["probes"]["directory"])
        / "probe_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("no committed probe metadata to check against")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["note_budget"] == int(probe_config["data"]["note_budget"])
    assert metadata["seed"] == int(probe_config["training"]["seed"])
    recorded_epochs = metadata["probes"]["neural"].get("epochs")
    assert recorded_epochs == int(probe_config["training"]["epochs"]), (
        f"committed probe ran {recorded_epochs} epochs but configs/probe.yaml says "
        f"{probe_config['training']['epochs']}"
    )


def test_notebook_written_configs_are_valid_yaml():
    """The %%writefile cells embed YAML; a typo there only surfaces at runtime in Colab."""
    import json

    notebook = json.loads(
        (REPO_ROOT / "notebooks/musemotion_colab.ipynb").read_text(encoding="utf-8")
    )
    written = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or not source.lstrip().startswith("%%writefile"):
            continue
        body = "\n".join(source.splitlines()[1:])
        assert isinstance(yaml.safe_load(body), dict), "a %%writefile config is not a YAML mapping"
        written += 1
    assert written >= 3


def test_probe_metadata_records_the_majority_class_baseline():
    """Accuracy on an imbalanced split must be read against the majority-class rate.

    The test split runs 20/24/30/34, so always answering Q4 scores 0.315 while learning nothing.
    A shuffled-label control judged against uniform 0.25 would look like it had found signal.
    """
    metadata_path = (
        resolve_path(load_yaml_config("configs/evaluation.yaml")["probes"]["directory"])
        / "probe_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("no committed probe metadata to check against")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    counts = metadata["split_quadrant_counts"]["test"]
    expected = max(counts.values()) / sum(counts.values())
    assert metadata["majority_class_accuracy"] == pytest.approx(expected)
    assert metadata["majority_class_accuracy"] > metadata["chance_accuracy"]


def test_committed_controls_do_not_beat_the_majority_class_baseline():
    """The control is the project's trust argument, so it is asserted, not just recorded."""
    metadata_path = (
        resolve_path(load_yaml_config("configs/evaluation.yaml")["probes"]["directory"])
        / "probe_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("no committed probe metadata to check against")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    majority = metadata["majority_class_accuracy"]

    for name, values in metadata["probes"].items():
        control = values.get("shuffled_label_control", {}).get("test")
        if not control:
            continue
        assert control["accuracy"] <= majority + 0.02, f"{name} control beats majority class"
        # Macro-F1 is not fooled by class frequency, so it must sit near uniform chance.
        assert control["macro_f1"] <= metadata["chance_accuracy"] + 0.05, name
        # And the real probe must clearly beat both baselines.
        assert values["test"]["accuracy"] > majority + 0.1, name


def test_the_full_evaluation_matches_the_committed_probes_note_budget():
    """The reportable config must ask the probes about the window they were trained on.

    A mismatch does not fail, it just depresses every accuracy and the ceiling together, so it
    would move every number without any of them looking wrong.
    """
    metadata_path = (
        resolve_path(load_yaml_config("configs/evaluation.yaml")["probes"]["directory"])
        / "probe_metadata.json"
    )
    if not metadata_path.exists():
        pytest.skip("no committed probe metadata to check against")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    evaluation = load_yaml_config("configs/evaluation.yaml")
    assert int(evaluation["data"]["note_budget"]) == int(metadata["note_budget"])
