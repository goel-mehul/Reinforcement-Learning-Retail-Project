import yaml
from pathlib import Path
from typing import Any


class ConfigLoader:
    """
    Loads and provides access to YAML configuration files.
    Supports dot-notation access for nested keys.
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dot-notation access to nested config values.
        e.g. loader.get('demand.price_sensitivity') -> 2.5
        """
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __repr__(self) -> str:
        return f"ConfigLoader({self.config_path.name})"


def load_env_config(base_dir: Path = None) -> ConfigLoader:
    """Convenience function to load env_config.yaml."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    return ConfigLoader(base_dir / "config" / "env_config.yaml")


def load_agent_config(base_dir: Path = None) -> ConfigLoader:
    """Convenience function to load agent_config.yaml."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    return ConfigLoader(base_dir / "config" / "agent_config.yaml")