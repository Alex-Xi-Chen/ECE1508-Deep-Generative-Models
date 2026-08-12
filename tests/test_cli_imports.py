import importlib


def test_cli_modules_import_without_side_effects():
    for module_name in [
        "musemotion.cli.train_classifier",
        "musemotion.cli.download_emopia",
        "musemotion.cli.prepare_emopia",
        "musemotion.cli.train_generator",
        "musemotion.cli.generate",
        "musemotion.cli.train_probe",
        "musemotion.cli.evaluate",
    ]:
        module = importlib.import_module(module_name)
        assert hasattr(module, "main")


def test_cli_parsers_accept_their_defaults():
    """Every CLI must parse an empty argument list, so the documented defaults are real."""
    for module_name, expected_config in [
        ("musemotion.cli.train_probe", "configs/probe.yaml"),
        ("musemotion.cli.evaluate", "configs/evaluation.yaml"),
    ]:
        module = importlib.import_module(module_name)
        args = module.build_parser().parse_args([])
        assert args.config == expected_config
