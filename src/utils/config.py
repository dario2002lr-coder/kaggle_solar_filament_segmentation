from pathlib import Path
import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]

def get_path(path: str) -> Path:
    return get_project_root() / path

def load_config() -> dict:
    """Load the project configuration from config.yaml."""
    project_root = get_project_root()
    config_path = project_root / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)