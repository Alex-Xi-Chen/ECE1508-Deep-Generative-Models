from pathlib import Path

from musemotion.config import load_yaml_config, resolve_path


def test_load_yaml_config_returns_nested_dictionary(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: bert-base-uncased\n", encoding="utf-8")

    config = load_yaml_config(config_file)

    assert config["model"]["name"] == "bert-base-uncased"


def test_resolve_path_expands_relative_paths_from_repo_root():
    path = resolve_path("artifacts/demo")

    assert isinstance(path, Path)
    assert path.name == "demo"
    assert path.parent.name == "artifacts"
