"""Tests for deepteam_attacks.py — two-phase explicit workflow."""

from enum import Enum

import pytest

from src.attacks.loader import AttackPrompt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_cases(inputs: list[str]):
    """Build RTTestCase objects with a minimal fake vulnerability_type Enum."""
    from deepteam.test_case import RTTestCase

    FakeType = Enum("FakeType", {"DIRECT_OVERRIDE": "direct_override"})
    return [
        RTTestCase(
            vulnerability="LLM01 Prompt Injection",
            vulnerability_type=FakeType.DIRECT_OVERRIDE,
            input=text,
        )
        for text in inputs
    ]


# ---------------------------------------------------------------------------
# generate_baseline_attacks
# ---------------------------------------------------------------------------


def test_generate_baseline_attacks_returns_test_cases(monkeypatch):
    from deepteam.vulnerabilities import CustomVulnerability

    from src.attacks.deepteam_attacks import generate_baseline_attacks

    fake_cases = _make_test_cases(["baseline 1", "baseline 2", "baseline 3"])
    monkeypatch.setattr(
        CustomVulnerability,
        "simulate_attacks",
        lambda self, **kwargs: fake_cases,
    )

    vuln = CustomVulnerability(
        name="LLM01 Prompt Injection",
        criteria="test",
        types=["direct_override"],
        simulator_model="gpt-3.5-turbo-0125",
    )
    result = generate_baseline_attacks(vuln, purpose="LLM01", attacks_per_type=3)

    assert len(result) == 3
    assert result[0].input == "baseline 1"
    assert result[2].input == "baseline 3"


# ---------------------------------------------------------------------------
# enhance_attacks
# ---------------------------------------------------------------------------


def test_enhance_attacks_returns_attack_prompts(monkeypatch):
    from deepteam.attacks.single_turn import SystemOverride

    from src.attacks.deepteam_attacks import enhance_attacks

    test_cases = _make_test_cases(["baseline 1", "baseline 2"])
    monkeypatch.setattr(
        SystemOverride,
        "enhance",
        lambda self, attack, **kwargs: f"enhanced: {attack}",
    )

    result = enhance_attacks(
        test_cases=test_cases,
        technique=SystemOverride(),
        technique_name="SystemOverride",
        strategy="direct_injection",
        simulator_model="gpt-3.5-turbo-0125",
        category="LLM01",
    )

    assert len(result) == 2
    assert all(isinstance(p, AttackPrompt) for p in result)
    assert all(p.attack_source == "deepteam" for p in result)
    assert all(p.attack_strategy == "direct_injection" for p in result)
    assert result[0].prompt == "enhanced: baseline 1"
    assert result[1].prompt == "enhanced: baseline 2"


def test_enhance_attacks_skips_none_input(monkeypatch):
    from deepteam.attacks.single_turn import SystemOverride
    from deepteam.test_case import RTTestCase

    from src.attacks.deepteam_attacks import enhance_attacks

    FakeType = Enum("FakeType", {"DIRECT_OVERRIDE": "direct_override"})
    test_cases = [
        RTTestCase(
            vulnerability="LLM01",
            vulnerability_type=FakeType.DIRECT_OVERRIDE,
            input=None,
        ),
        RTTestCase(
            vulnerability="LLM01",
            vulnerability_type=FakeType.DIRECT_OVERRIDE,
            input="valid prompt",
        ),
    ]
    monkeypatch.setattr(
        SystemOverride,
        "enhance",
        lambda self, attack, **kwargs: f"enhanced: {attack}",
    )

    result = enhance_attacks(
        test_cases=test_cases,
        technique=SystemOverride(),
        technique_name="SystemOverride",
        strategy="direct_injection",
        simulator_model="gpt-3.5-turbo-0125",
        category="LLM01",
    )

    assert len(result) == 1
    assert result[0].prompt == "enhanced: valid prompt"


def test_enhance_attacks_raises_when_all_fail(monkeypatch):
    from deepteam.attacks.single_turn import SystemOverride

    from src.attacks.deepteam_attacks import enhance_attacks

    test_cases = _make_test_cases(["baseline 1"])

    def _always_fail(self, attack, **kwargs):
        raise RuntimeError("api error")

    monkeypatch.setattr(SystemOverride, "enhance", _always_fail)

    with pytest.raises(RuntimeError, match="no valid prompts"):
        enhance_attacks(
            test_cases=test_cases,
            technique=SystemOverride(),
            technique_name="SystemOverride",
            strategy="direct_injection",
            simulator_model="gpt-3.5-turbo-0125",
            category="LLM01",
        )


# ---------------------------------------------------------------------------
# generate_deepteam_prompts — integration (both phases mocked)
# ---------------------------------------------------------------------------


def test_generate_deepteam_prompts_full_pipeline(monkeypatch, tmp_path):
    from deepteam.attacks.single_turn import SystemOverride
    from deepteam.vulnerabilities import CustomVulnerability

    from src.attacks.deepteam_attacks import generate_deepteam_prompts
    from src.config import DeepTeamCategoryConfig

    prompt_file = tmp_path / "deepteam_llm01.txt"
    prompt_file.write_text("fake custom prompt template")

    config = DeepTeamCategoryConfig(
        types=["direct_override", "policy_bypass"],
        attacks_per_type=2,
        technique="SystemOverride",
        custom_prompt_file=str(prompt_file),
    )

    fake_cases = _make_test_cases(["baseline 1", "baseline 2"])
    monkeypatch.setattr(
        CustomVulnerability,
        "simulate_attacks",
        lambda self, **kwargs: fake_cases,
    )
    monkeypatch.setattr(
        SystemOverride,
        "enhance",
        lambda self, attack, **kwargs: f"enhanced: {attack}",
    )

    result = generate_deepteam_prompts("LLM01", config, "gpt-3.5-turbo-0125")

    assert len(result) == 2
    assert all(p.attack_source == "deepteam" for p in result)
    assert all(p.attack_strategy == "direct_injection" for p in result)
    assert result[0].prompt == "enhanced: baseline 1"


# ---------------------------------------------------------------------------
# Error cases — no mocks needed, validation fires before any API call
# ---------------------------------------------------------------------------


def test_unknown_technique_raises_value_error(tmp_path):
    from src.attacks.deepteam_attacks import generate_deepteam_prompts
    from src.config import DeepTeamCategoryConfig

    prompt_file = tmp_path / "deepteam_llm01.txt"
    prompt_file.write_text("fake")

    config = DeepTeamCategoryConfig(
        types=["direct_override"],
        attacks_per_type=1,
        technique="NonExistentTechnique",
        custom_prompt_file=str(prompt_file),
    )
    with pytest.raises(ValueError, match="Unknown technique"):
        generate_deepteam_prompts("LLM01", config, "gpt-3.5-turbo-0125")


def test_missing_prompt_file_raises_file_not_found_error():
    from src.attacks.deepteam_attacks import generate_deepteam_prompts
    from src.config import DeepTeamCategoryConfig

    config = DeepTeamCategoryConfig(
        types=["direct_override"],
        attacks_per_type=1,
        technique="SystemOverride",
        custom_prompt_file="prompts/this_file_does_not_exist.txt",
    )
    with pytest.raises(FileNotFoundError):
        generate_deepteam_prompts("LLM01", config, "gpt-3.5-turbo-0125")
