"""CLI target: set NOISE_SETTINGS to a reviewed settings JSON file."""

import os
from pathlib import Path
from noise_recipe import Settings, build_pipeline

settings_path = Path(os.environ.get("NOISE_SETTINGS", Path(__file__).with_name("settings.json"))).resolve()
settings = Settings.model_validate_json(settings_path.read_text())
for array in settings.arrays.values():
    if not array.layout.is_absolute():
        array.layout = (settings_path.parent / array.layout).resolve()
recipe = build_pipeline(settings)
