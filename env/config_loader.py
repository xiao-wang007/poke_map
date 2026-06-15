from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
