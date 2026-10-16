from pathlib import Path

import pytest

from labelaudit.config import Config, load_config, read_run_config, save_config
from labelaudit.noise import flip_target_matrix

DEMO = Path(__file__).parents[1] / "configs" / "agnews-demo.toml"


def test_demo_config_loads_and_has_valid_flip_targets():
    config = load_config(DEMO)
    assert config.noise.mode == "class_dependent"
    flip_target_matrix(4, config.noise.flip_targets)


def test_config_round_trips_through_run_directory(tmp_path):
    config = load_config(DEMO)
    save_config(config, tmp_path / "config.json")
    assert read_run_config(tmp_path / "config.json") == config


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        Config.from_dict({"audit": {"fold": 3}})
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_dict({"sed": 1})


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"noise": {"rate": 1.5}})
    with pytest.raises(ValueError):
        Config.from_dict({"audit": {"weight_a": 0, "weight_b": 0}})
