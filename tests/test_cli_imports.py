import importlib


def test_cli_modules_import_without_side_effects():
    for module_name in [
        "musemotion.cli.train_classifier",
        "musemotion.cli.download_emopia",
        "musemotion.cli.prepare_emopia",
        "musemotion.cli.train_generator",
    ]:
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")
