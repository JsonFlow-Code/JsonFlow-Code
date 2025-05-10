import json
import os
from typing import Any

class Config:
    """Configuration manager for JSONFlow."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = cls._load_config()
        return cls._instance

    @staticmethod
    def _load_config() -> dict:
        """Load configuration from file or environment."""
        config = {
            "generator": {
                "default_language": "python",
                "indent_level": 4
            },
            "logging": {
                "level": "INFO",
                "file": "jsonflow.log"
            }
        }
        config_file = os.environ.get("JSONFLOW_CONFIG", "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except json.JSONDecodeError as e:
                print(f"Error reading config file: {e}")
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value by key, supporting nested keys."""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                if not isinstance(value, dict):
                    return default
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
