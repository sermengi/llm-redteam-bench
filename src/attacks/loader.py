"""Attack prompt loader: aggregates manual, PyRIT, and DeepTeam prompts per category."""

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.config import AttacksConfig

logger = logging.getLogger(__name__)

_MANUAL_DIR = Path(__file__).resolve().parent / "manual"
_CATEGORY_FILE_MAP: dict[str, str] = {
    "LLM01": "llm01.txt",
    "LLM02": "llm02.txt",
    "LLM04": "llm04.txt",
    "LLM06": "llm06.txt",
    "LLM07": "llm07.txt",
    "LLM09": "llm09.txt",
}


@dataclass
class AttackPrompt:
    """A single attack prompt with its source and strategy metadata."""

    prompt: str
    attack_source: Literal["pyrit", "deepteam", "manual"]
    attack_strategy: Literal["direct_injection", "indirect_injection", "multi_turn_crescendo"]


class CachedPromptSet(BaseModel):
    """Serializable container for a generated prompt list with cache metadata."""

    category: str
    config_hash: str
    generated_at: str
    prompts: list[dict[str, str]]


def _load_manual(category: str, n: int) -> list[AttackPrompt]:
    """Load n manual prompts from the category text file.

    Args:
        category: OWASP category string (e.g. 'LLM01').
        n: Number of prompts to return (takes first n lines).

    Returns:
        List of AttackPrompt with source='manual', strategy='direct_injection'.
    """
    path = _MANUAL_DIR / _CATEGORY_FILE_MAP[category]
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [
        AttackPrompt(prompt=line, attack_source="manual", attack_strategy="direct_injection")
        for line in lines[:n]
    ]


def load_attacks(category: str, config: AttacksConfig) -> list[AttackPrompt]:
    """Load all attack prompts for a given category from all three sources.

    Sources and counts are driven by config. DeepTeam count is runtime-derived
    from the deepteam.categories config block. Total is logged, not pre-validated.

    Args:
        category: OWASP category string (e.g. 'LLM01').
        config: Loaded AttacksConfig from configs/attacks.yaml.

    Returns:
        Flat list of AttackPrompt from manual + pyrit + deepteam sources.
    """
    from src.attacks.deepteam_attacks import generate_deepteam_prompts
    from src.attacks.pyrit_attacks import generate_pyrit_prompts

    counts = config.prompts_per_category
    manual = _load_manual(category, counts.manual)
    seed_prompts = [a.prompt for a in manual]
    pyrit = generate_pyrit_prompts(
        category,
        seed_prompts=seed_prompts,
        n=counts.pyrit,
        converters=config.pyrit_converters,
    )
    deepteam = generate_deepteam_prompts(
        category,
        config=config.deepteam.categories[category],
        simulator_model=config.deepteam.simulator_model,
    )

    all_attacks = manual + pyrit + deepteam
    logger.info(
        "Loaded %d attacks for %s (manual=%d pyrit=%d deepteam=%d)",
        len(all_attacks),
        category,
        len(manual),
        len(pyrit),
        len(deepteam),
    )
    return all_attacks


def save_prompts(
    prompts: list[AttackPrompt],
    category: str,
    config_hash: str,
    cache_dir: Path,
) -> Path:
    """Serialize prompts to a JSON cache file.

    Args:
        prompts: Generated attack prompts to cache.
        category: OWASP category string (e.g. 'LLM01').
        config_hash: SHA-256 of attacks.yaml + prompt templates at generation time.
        cache_dir: Directory to write the cache file into.

    Returns:
        Path to the written cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = CachedPromptSet(
        category=category,
        config_hash=config_hash,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompts=[dataclasses.asdict(p) for p in prompts],
    )
    path = cache_dir / f"{category}.json"
    path.write_text(payload.model_dump_json(indent=2))
    logger.info("Cached %d prompts for %s → %s", len(prompts), category, path)
    return path


def load_cached_prompts(
    category: str,
    config_hash: str,
    cache_dir: Path,
) -> list[AttackPrompt] | None:
    """Load cached prompts if the file exists and the config hash matches.

    Args:
        category: OWASP category string (e.g. 'LLM01').
        config_hash: Current SHA-256 of attacks.yaml + prompt templates.
        cache_dir: Directory to look for the cache file.

    Returns:
        List of AttackPrompt on cache hit, None on miss or hash mismatch.
    """
    path = cache_dir / f"{category}.json"
    if not path.exists():
        return None
    data = CachedPromptSet.model_validate_json(path.read_text())
    if data.config_hash != config_hash:
        logger.info(
            "Cache stale for %s (stored=%s current=%s) — will regenerate",
            category,
            data.config_hash[:8],
            config_hash[:8],
        )
        return None
    prompts = [AttackPrompt(**p) for p in data.prompts]
    logger.info("Cache hit: loaded %d prompts for %s from %s", len(prompts), category, path)
    return prompts
