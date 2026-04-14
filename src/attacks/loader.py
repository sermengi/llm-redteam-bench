"""Attack prompt loader: aggregates manual, PyRIT, and DeepTeam prompts per category."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

    Sources and counts are driven by config.prompts_per_category.
    Strategy assignment: manual → direct_injection; PyRIT → split direct/indirect;
    DeepTeam → direct_injection.

    Args:
        category: OWASP category string (e.g. 'LLM01').
        config: Loaded AttacksConfig from configs/attacks.yaml.

    Returns:
        Flat list of AttackPrompt, length == config.prompts_per_category.total.
    """
    # Deferred to avoid ImportError while pyrit_attacks/deepteam_attacks are not yet created.
    from src.attacks.pyrit_attacks import generate_pyrit_prompts
    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    counts = config.prompts_per_category
    manual = _load_manual(category, counts.manual)
    seed_prompts = [a.prompt for a in manual]
    pyrit = generate_pyrit_prompts(category, seed_prompts=seed_prompts, n=counts.pyrit)
    deepteam = generate_deepteam_prompts(category, n=counts.deepteam)

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
