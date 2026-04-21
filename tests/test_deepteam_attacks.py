"""Tests for deepteam_attacks.py — all tests mock AttackSimulator.simulate."""

import pytest

from src.attacks.loader import AttackPrompt
from src.config import DeepTeamCategoryConfig


@pytest.fixture()
def category_config():
    return DeepTeamCategoryConfig(
        vulnerabilities=["PromptLeakage"],
        attacks_per_vulnerability_type=1,
        attack_methods=["PromptInjection"],
    )


@pytest.fixture()
def fake_test_cases():
    from deepteam.test_case import RTTestCase
    from deepteam.vulnerabilities.prompt_leakage.types import PromptLeakageType

    return [
        RTTestCase(
            vulnerability="PromptLeakage",
            vulnerability_type=PromptLeakageType.INSTRUCTIONS,
            input="injected prompt 1",
            actual_output=None,
            attack_method="PromptInjection",
        ),
        RTTestCase(
            vulnerability="PromptLeakage",
            vulnerability_type=PromptLeakageType.INSTRUCTIONS,
            input="injected prompt 2",
            actual_output=None,
            attack_method="SyntheticContextInjection",
        ),
    ]


@pytest.fixture()
def mock_simulator(monkeypatch, fake_test_cases):
    monkeypatch.setattr(
        "deepteam.attacks.attack_simulator.AttackSimulator.simulate",
        lambda self, *args, **kwargs: fake_test_cases,
    )


def test_returns_attack_prompt_instances(mock_simulator, category_config):
    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    result = generate_deepteam_prompts("LLM01", category_config, "gpt-3.5-turbo-0125")
    assert all(isinstance(p, AttackPrompt) for p in result)


def test_attack_source_is_deepteam(mock_simulator, category_config):
    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    result = generate_deepteam_prompts("LLM01", category_config, "gpt-3.5-turbo-0125")
    assert all(p.attack_source == "deepteam" for p in result)


def test_strategy_mapped_from_attack_method(mock_simulator, fake_test_cases, category_config):
    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    result = generate_deepteam_prompts("LLM01", category_config, "gpt-3.5-turbo-0125")
    by_prompt = {p.prompt: p.attack_strategy for p in result}
    assert by_prompt["injected prompt 1"] == "direct_injection"
    assert by_prompt["injected prompt 2"] == "indirect_injection"


def test_unknown_vulnerability_raises_before_api_call(category_config):
    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    bad_config = DeepTeamCategoryConfig(
        vulnerabilities=["NonExistentVuln"],
        attacks_per_vulnerability_type=1,
        attack_methods=["PromptInjection"],
    )
    with pytest.raises(ValueError, match="Unknown vulnerability name"):
        generate_deepteam_prompts("LLM01", bad_config, "gpt-3.5-turbo-0125")


def test_unknown_attack_method_raises_before_api_call(category_config):
    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    bad_config = DeepTeamCategoryConfig(
        vulnerabilities=["PromptLeakage"],
        attacks_per_vulnerability_type=1,
        attack_methods=["NonExistentAttack"],
    )
    with pytest.raises(ValueError, match="Unknown attack method name"):
        generate_deepteam_prompts("LLM01", bad_config, "gpt-3.5-turbo-0125")


def test_test_cases_with_error_are_skipped(monkeypatch, category_config):
    from deepteam.test_case import RTTestCase
    from deepteam.vulnerabilities.prompt_leakage.types import PromptLeakageType

    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    cases = [
        RTTestCase(
            vulnerability="PromptLeakage",
            vulnerability_type=PromptLeakageType.INSTRUCTIONS,
            input="good prompt",
            actual_output=None,
            attack_method="PromptInjection",
        ),
        RTTestCase(
            vulnerability="PromptLeakage",
            vulnerability_type=PromptLeakageType.INSTRUCTIONS,
            input="bad prompt",
            actual_output=None,
            attack_method="PromptInjection",
            error="generation failed",
        ),
    ]
    monkeypatch.setattr(
        "deepteam.attacks.attack_simulator.AttackSimulator.simulate",
        lambda self, *args, **kwargs: cases,
    )
    result = generate_deepteam_prompts("LLM01", category_config, "gpt-3.5-turbo-0125")
    assert len(result) == 1
    assert result[0].prompt == "good prompt"


def test_test_cases_with_none_input_are_skipped(monkeypatch, category_config):
    from deepteam.test_case import RTTestCase
    from deepteam.vulnerabilities.prompt_leakage.types import PromptLeakageType

    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    cases = [
        RTTestCase(
            vulnerability="PromptLeakage",
            vulnerability_type=PromptLeakageType.INSTRUCTIONS,
            input=None,
            actual_output=None,
            attack_method="PromptInjection",
        ),
        RTTestCase(
            vulnerability="PromptLeakage",
            vulnerability_type=PromptLeakageType.INSTRUCTIONS,
            input="valid prompt",
            actual_output=None,
            attack_method="PromptInjection",
        ),
    ]
    monkeypatch.setattr(
        "deepteam.attacks.attack_simulator.AttackSimulator.simulate",
        lambda self, *args, **kwargs: cases,
    )
    result = generate_deepteam_prompts("LLM01", category_config, "gpt-3.5-turbo-0125")
    assert len(result) == 1
    assert result[0].prompt == "valid prompt"


def test_zero_valid_prompts_raises_runtime_error(monkeypatch, category_config):
    from src.attacks.deepteam_attacks import generate_deepteam_prompts

    monkeypatch.setattr(
        "deepteam.attacks.attack_simulator.AttackSimulator.simulate",
        lambda self, *args, **kwargs: [],
    )
    with pytest.raises(RuntimeError, match="no valid prompts produced"):
        generate_deepteam_prompts("LLM01", category_config, "gpt-3.5-turbo-0125")
