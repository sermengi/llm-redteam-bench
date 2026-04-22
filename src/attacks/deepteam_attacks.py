"""DeepTeam attack prompt generator — explicit two-phase workflow.

Phase 1: CustomVulnerability.simulate_attacks() generates baseline RTTestCase objects.
Phase 2: A single-turn technique's enhance() transforms each baseline prompt.
"""

import logging
from pathlib import Path
from typing import Literal

from deepteam.attacks.single_turn import (
    AuthorityEscalation,
    BaseSingleTurnAttack,
    ContextPoisoning,
    EmotionalManipulation,
    GoalRedirection,
    InputBypass,
    LinguisticConfusion,
    PermissionEscalation,
    PromptInjection,
    Roleplay,
    SyntheticContextInjection,
    SystemOverride,
)
from deepteam.test_case import RTTestCase
from deepteam.vulnerabilities import CustomVulnerability

from src.attacks.loader import AttackPrompt
from src.config import DeepTeamCategoryConfig

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_TECHNIQUE_MAP: dict[str, type] = {
    "SystemOverride": SystemOverride,
    "PromptInjection": PromptInjection,
    "Roleplay": Roleplay,
    "AuthorityEscalation": AuthorityEscalation,
    "EmotionalManipulation": EmotionalManipulation,
    "SyntheticContextInjection": SyntheticContextInjection,
    "PermissionEscalation": PermissionEscalation,
    "GoalRedirection": GoalRedirection,
    "LinguisticConfusion": LinguisticConfusion,
    "InputBypass": InputBypass,
    "ContextPoisoning": ContextPoisoning,
}

# Only SyntheticContextInjection produces indirect injections; everything else is direct.
_TECHNIQUE_STRATEGY_MAP: dict[str, Literal["indirect_injection"]] = {
    "SyntheticContextInjection": "indirect_injection",
}


def generate_baseline_attacks(
    vulnerability: CustomVulnerability,
    purpose: str,
    attacks_per_type: int,
) -> list[RTTestCase]:
    """Generate unenhanced baseline test cases from a CustomVulnerability.

    Args:
        vulnerability: Configured CustomVulnerability instance.
        purpose: Context string passed to simulate_attacks (e.g. 'LLM01').
        attacks_per_type: Number of baseline prompts to generate per vulnerability type.

    Returns:
        List of RTTestCase with .input populated. Exceptions propagate.
    """
    return vulnerability.simulate_attacks(
        purpose=purpose,
        attacks_per_vulnerability_type=attacks_per_type,
    )


def enhance_attacks(
    test_cases: list[RTTestCase],
    technique: BaseSingleTurnAttack,
    technique_name: str,
    strategy: Literal["direct_injection", "indirect_injection", "multi_turn_crescendo"],
    simulator_model: str,
    category: str,
    ignore_errors: bool = True,
) -> list[AttackPrompt]:
    """Enhance baseline prompts using a single-turn attack technique.

    Args:
        test_cases: Baseline RTTestCase objects from generate_baseline_attacks().
        technique: Instantiated single-turn attack.
        technique_name: Name string for logging.
        strategy: attack_strategy value to stamp on every resulting AttackPrompt.
        simulator_model: OpenAI model string passed to technique.enhance().
        category: OWASP category string for logging context.
        ignore_errors: If True, skip failed enhancements with a warning.

    Returns:
        List of AttackPrompt with attack_source='deepteam'.

    Raises:
        RuntimeError: If no valid enhanced prompts result.
    """
    result: list[AttackPrompt] = []
    for tc in test_cases:
        if tc.input is None:
            continue
        try:
            enhanced = technique.enhance(attack=tc.input, simulator_model=simulator_model)
            result.append(
                AttackPrompt(
                    prompt=enhanced,
                    attack_source="deepteam",
                    attack_strategy=strategy,
                )
            )
        except Exception as exc:
            if ignore_errors:
                logger.warning(
                    "Enhancement failed for %s test case with %s: %s",
                    category,
                    technique_name,
                    exc,
                )
            else:
                raise

    if not result:
        raise RuntimeError(
            f"enhance_attacks: no valid prompts produced for {category}. "
            "Check technique config and OPENAI_API_KEY."
        )

    return result


def generate_deepteam_prompts(
    category: str,
    config: DeepTeamCategoryConfig,
    simulator_model: str,
) -> list[AttackPrompt]:
    """Generate DeepTeam attack prompts for a category using the explicit two-phase workflow.

    Phase 1: Build a CustomVulnerability and call simulate_attacks() to get baselines.
    Phase 2: Call technique.enhance() on each baseline prompt.

    Args:
        category: OWASP category string (e.g. 'LLM01').
        config: DeepTeam category config from attacks.yaml.
        simulator_model: OpenAI model name for both phases.

    Returns:
        List of AttackPrompt with attack_source='deepteam'.

    Raises:
        ValueError: If config.technique is not in _TECHNIQUE_MAP.
        FileNotFoundError: If config.custom_prompt_file does not exist.
        RuntimeError: If all enhanced prompts are filtered out.
    """
    if config.technique not in _TECHNIQUE_MAP:
        raise ValueError(
            f"Unknown technique {config.technique!r}. "
            f"Valid techniques: {sorted(_TECHNIQUE_MAP)}"
        )

    prompt_path = _PROJECT_ROOT / config.custom_prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Custom prompt file not found: {prompt_path.resolve()}"
        )

    custom_prompt = prompt_path.read_text()
    strategy = _TECHNIQUE_STRATEGY_MAP.get(config.technique, "direct_injection")
    technique = _TECHNIQUE_MAP[config.technique]()

    vulnerability = CustomVulnerability(
        name=f"{category} Prompt Injection",
        criteria=f"The model must not comply with {category} adversarial inputs.",
        types=config.types,
        custom_prompt=custom_prompt,
        simulator_model=simulator_model,
    )

    test_cases = generate_baseline_attacks(
        vulnerability=vulnerability,
        purpose=category,
        attacks_per_type=config.attacks_per_type,
    )

    prompts = enhance_attacks(
        test_cases=test_cases,
        technique=technique,
        technique_name=config.technique,
        strategy=strategy,
        simulator_model=simulator_model,
        category=category,
    )

    logger.info(
        "Generated %d DeepTeam prompts for %s (technique=%s)",
        len(prompts),
        category,
        config.technique,
    )
    return prompts
