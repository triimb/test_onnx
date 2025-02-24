import json
import os
from typing import Any, Dict, Optional

class ModelRegistry:
    """
    A simple registry for storing metadata about quantized ONNX models.
    The registry is stored as a JSON file.
    """

    def __init__(self, registry_file: str = "model_registry.json") -> None:
        """
        Initialize the registry by loading an existing JSON file or starting fresh.
        """
        self.registry_file = registry_file
        if os.path.exists(registry_file):
            with open(registry_file, "r") as f:
                self.registry: Dict[str, Any] = json.load(f)
        else:
            self.registry = {}

    def register_model(self, model_name: str, version: str, file_path: str, metadata: Dict[str, Any]) -> None:
        """
        Register a new quantized model with metadata.

        Parameters:
            model_name (str): A unique name for the model.
            version (str): Version string.
            file_path (str): File path to the quantized model.
            metadata (dict): Additional metadata (e.g., quantization parameters, performance metrics).
        """
        self.registry[model_name] = {
            "version": version,
            "file_path": file_path,
            "metadata": metadata
        }
        self._save_registry()

    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model metadata by name.

        Parameters:
            model_name (str): The model's unique name.

        Returns:
            dict or None: The model information if found, else None.
        """
        return self.registry.get(model_name)

    def _save_registry(self) -> None:
        """
        Save the current registry state to the JSON file.
        """
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
