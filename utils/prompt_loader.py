import yaml
from typing import Dict


def load_prompt(path: str) -> Dict:
    """Load a YAML prompt file into a dict (system_prompt, user_prompt, etc.)"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
