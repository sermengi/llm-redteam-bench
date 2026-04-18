"""Static template-based attack prompt generator.

Templates are sourced from DeepTeam's vulnerability taxonomy and stored in
deepteam_templates.yaml. Real DeepTeam LLM-based generation (via simulate_attacks)
is deferred — see project roadmap.
"""

import logging
from pathlib import Path

import yaml

from src.attacks.loader import AttackPrompt

logger = logging.getLogger(__name__)

_TEMPLATES_PATH = Path(__file__).resolve().parent / "deepteam_templates.yaml"
_TEMPLATES: dict[str, list[str]] = yaml.safe_load(_TEMPLATES_PATH.read_text())


def generate_template_prompts(category: str, n: int) -> list[AttackPrompt]:
    """Generate n attack prompts using static DeepTeam-taxonomy templates.

    All returned prompts use attack_strategy='direct_injection'.

    Args:
        category: OWASP category string (e.g. 'LLM01').
        n: Number of prompts to return.

    Returns:
        List of AttackPrompt with attack_source='template'.

    Raises:
        ValueError: If the category has no templates or fewer than n are defined.
    """
    if category not in _TEMPLATES:
        raise ValueError(
            f"No templates defined for category '{category}'. "
            f"Add entries to src/attacks/deepteam_templates.yaml."
        )
    templates = _TEMPLATES[category]
    if len(templates) < n:
        raise ValueError(
            f"generate_template_prompts: needed {n} templates for {category} "
            f"but only {len(templates)} are defined."
        )
    selected = templates[:n]
    logger.info("Generated %d template prompts for %s", len(selected), category)
    return [
        AttackPrompt(prompt=p, attack_source="template", attack_strategy="direct_injection")
        for p in selected
    ]
