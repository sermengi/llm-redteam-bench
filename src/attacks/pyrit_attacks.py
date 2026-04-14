"""PyRIT-based attack prompt generator (shallow: converter-only, no target/memory)."""

import asyncio
import logging
from typing import Literal

from pyrit.prompt_converter import Base64Converter, LeetspeakConverter, ROT13Converter

from src.attacks.loader import AttackPrompt

logger = logging.getLogger(__name__)

# Week 4 extension point: add CrescendoOrchestrator-based generator here
# for multi_turn_crescendo strategy. See design doc section "Week 4 Extension Point".

_CONVERTERS = [Base64Converter(), ROT13Converter(), LeetspeakConverter()]


def _convert_sync(text: str, converter) -> str:
    """Apply a PyRIT converter to a text string synchronously.

    Args:
        text: The seed prompt string to convert.
        converter: An instantiated PyRIT PromptConverter.

    Returns:
        The converted prompt string.
    """
    result = asyncio.run(converter.convert_async(prompt=text, input_type="text"))
    return result.output_text


def generate_pyrit_prompts(
    category: str,
    seed_prompts: list[str],
    n: int,
) -> list[AttackPrompt]:
    """Generate n attack prompts using PyRIT converters applied to seed_prompts.

    Applies each of 3 converters to each seed prompt, producing len(seed_prompts) * 3
    variants, then takes the first n. First ceil(n/2) are direct_injection, rest are
    indirect_injection.

    Args:
        category: OWASP category string (unused here, reserved for future filtering).
        seed_prompts: Base prompts to convert (typically the 5 manual prompts).
        n: Total number of PyRIT prompts to return.

    Returns:
        List of AttackPrompt with attack_source='pyrit'.
    """
    variants: list[str] = []
    for converter in _CONVERTERS:
        for seed in seed_prompts:
            try:
                variants.append(_convert_sync(seed, converter))
            except Exception:
                logger.warning(
                    "Converter %s failed on seed, skipping.", type(converter).__name__
                )

    variants = variants[:n]
    n_direct = (n + 1) // 2  # ceiling division — direct gets the extra if n is odd

    result: list[AttackPrompt] = []
    for i, prompt in enumerate(variants):
        strategy: Literal["direct_injection", "indirect_injection"] = (
            "direct_injection" if i < n_direct else "indirect_injection"
        )
        result.append(
            AttackPrompt(prompt=prompt, attack_source="pyrit", attack_strategy=strategy)
        )

    logger.info("Generated %d PyRIT prompts for %s", len(result), category)
    return result
