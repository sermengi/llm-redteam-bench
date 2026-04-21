"""DeepTeam attack prompt generator using AttackSimulator.simulate()."""

import logging
from typing import Literal

from deepteam.attacks.attack_simulator import AttackSimulator
from deepteam.attacks.single_turn import GrayBox, PromptInjection, SyntheticContextInjection
from deepteam.vulnerabilities import (
    IndirectInstruction,
    Misinformation,
    PIILeakage,
    PromptLeakage,
    Robustness,
    ShellInjection,
    SQLInjection,
    Toxicity,
)

from src.attacks.loader import AttackPrompt
from src.config import DeepTeamCategoryConfig

logger = logging.getLogger(__name__)

_VULNERABILITY_MAP: dict[str, type] = {
    "PromptLeakage": PromptLeakage,
    "IndirectInstruction": IndirectInstruction,
    "PIILeakage": PIILeakage,
    "Misinformation": Misinformation,
    "Robustness": Robustness,
    "Toxicity": Toxicity,
    "SQLInjection": SQLInjection,
    "ShellInjection": ShellInjection,
}

_ATTACK_METHOD_MAP: dict[str, type] = {
    "PromptInjection": PromptInjection,
    "SyntheticContextInjection": SyntheticContextInjection,
    "GrayBox": GrayBox,
}

_STRATEGY_MAP: dict[
    str, Literal["direct_injection", "indirect_injection", "multi_turn_crescendo"]
] = {
    "PromptInjection": "direct_injection",
    "SyntheticContextInjection": "indirect_injection",
    "GrayBox": "direct_injection",
}


def generate_deepteam_prompts(
    category: str,
    config: DeepTeamCategoryConfig,
    simulator_model: str,
) -> list[AttackPrompt]:
    """Generate attack prompts for a category using DeepTeam's AttackSimulator.

    Args:
        category: OWASP category string (e.g. 'LLM01').
        config: DeepTeam category config with vulnerability and attack method names.
        simulator_model: OpenAI model name for attack synthesis.

    Returns:
        List of AttackPrompt with attack_source='deepteam'.

    Raises:
        ValueError: If an unknown vulnerability or attack method name is in config.
        RuntimeError: If all generated test cases are filtered out.
    """
    unknown_vulns = [v for v in config.vulnerabilities if v not in _VULNERABILITY_MAP]
    if unknown_vulns:
        raise ValueError(
            f"Unknown vulnerability name(s): {unknown_vulns}. "
            f"Valid names: {list(_VULNERABILITY_MAP)}"
        )

    unknown_attacks = [a for a in config.attack_methods if a not in _ATTACK_METHOD_MAP]
    if unknown_attacks:
        raise ValueError(
            f"Unknown attack method name(s): {unknown_attacks}. "
            f"Valid names: {list(_ATTACK_METHOD_MAP)}"
        )

    vulnerabilities = [
        _VULNERABILITY_MAP[v](simulator_model=simulator_model) for v in config.vulnerabilities
    ]
    attack_methods = [_ATTACK_METHOD_MAP[a]() for a in config.attack_methods]

    simulator = AttackSimulator(purpose=category, max_concurrent=1)
    test_cases = simulator.simulate(
        attacks_per_vulnerability_type=config.attacks_per_vulnerability_type,
        vulnerabilities=vulnerabilities,
        attacks=attack_methods,
        ignore_errors=True,
    )

    result: list[AttackPrompt] = []
    for tc in test_cases:
        if tc.input is None:
            continue
        if tc.error is not None:
            logger.warning("DeepTeam skipped a test case for %s (error: %s)", category, tc.error)
            continue
        strategy = _STRATEGY_MAP.get(tc.attack_method or "", "direct_injection")
        result.append(
            AttackPrompt(
                prompt=tc.input,
                attack_source="deepteam",
                attack_strategy=strategy,
            )
        )

    if not result:
        raise RuntimeError(
            f"generate_deepteam_prompts: no valid prompts produced for {category}. "
            "Check vulnerability config and OPENAI_API_KEY."
        )

    logger.info("Generated %d DeepTeam prompts for %s", len(result), category)
    return result
