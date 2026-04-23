"""Shared utilities for the scripts layer."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.config import AttacksConfig

_CONFIGS_DIR = Path("configs")
_PROMPTS_DIR = Path("prompts")
_RESULTS_DIR = Path("results")
_PROMPTS_CACHE_DIR = _RESULTS_DIR / "prompts"

_VALID_CATEGORIES = ["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"]


def compute_config_hash(attacks_config: AttacksConfig) -> str:
    """Hash attacks.yaml and all deepteam custom_prompt_file paths.

    Including prompt template files ensures the cache is invalidated when
    a DeepTeam generator template changes, not just the YAML config.

    Args:
        attacks_config: Loaded AttacksConfig from configs/attacks.yaml.

    Returns:
        SHA-256 hex digest string.
    """
    hasher = hashlib.sha256()
    hasher.update((_CONFIGS_DIR / "attacks.yaml").read_bytes())
    for key in sorted(attacks_config.deepteam.categories):
        cat_cfg = attacks_config.deepteam.categories[key]
        if not cat_cfg.custom_prompt_file:
            continue
        prompt_file = Path(cat_cfg.custom_prompt_file)
        if prompt_file.exists():
            hasher.update(prompt_file.read_bytes())
    return hasher.hexdigest()


def build_run_id(config_hash: str) -> str:
    """Generate a unique run ID: timestamp + first 6 chars of the config hash.

    Args:
        config_hash: SHA-256 hex digest from compute_config_hash().

    Returns:
        String of the form '20260423-143022-a3f9c1'.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{config_hash[:6]}"
