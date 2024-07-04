import json
import os
import logging
from typing import Any, Dict

class Configuration:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.parameters = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)
        
        self._validate_config(config)
        return config.get('parameters', {})

    def _validate_config(self, config: Dict[str, Any]) -> None:
        if 'parameters' not in config:
            raise ValueError("Configuration must contain 'parameters' key.")

        parameters = config['parameters']

        if 'use_openai' not in parameters:
            raise ValueError("Configuration must contain 'use_openai' parameter.")
        
        if parameters['use_openai']:
            if '#api_key' not in parameters:
                raise ValueError("When 'use_openai' is true, '#api_key' must be provided.")
            if 'openai_model' not in parameters:
                raise ValueError("When 'use_openai' is true, 'openai_model' must be provided.")
        
        if not parameters['use_openai']:
            if 'model_name' not in parameters:
                raise ValueError("When 'use_openai' is false, 'model_name' must be provided.")

    def get(self, key: str, default: Any = None) -> Any:
        return self.parameters.get(key, default)

