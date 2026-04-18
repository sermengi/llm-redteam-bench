"""Tests for template_attacks.py."""

import pytest

from src.attacks.loader import AttackPrompt

CATEGORIES = ["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"]


def test_yaml_loads_all_categories():
    from src.attacks.template_attacks import _TEMPLATES

    assert set(CATEGORIES).issubset(set(_TEMPLATES.keys()))


@pytest.mark.parametrize("category", CATEGORIES)
def test_yaml_has_ten_templates_per_category(category):
    from src.attacks.template_attacks import _TEMPLATES

    assert (
        len(_TEMPLATES[category]) >= 10
    ), f"{category} has only {len(_TEMPLATES[category])} templates, need >= 10"


@pytest.mark.parametrize("category", CATEGORIES)
def test_generate_returns_correct_count(category):
    from src.attacks.template_attacks import generate_template_prompts

    prompts = generate_template_prompts(category, n=10)
    assert len(prompts) == 10


@pytest.mark.parametrize("category", CATEGORIES)
def test_generate_returns_attack_prompt_instances(category):
    from src.attacks.template_attacks import generate_template_prompts

    prompts = generate_template_prompts(category, n=3)
    assert all(isinstance(p, AttackPrompt) for p in prompts)


@pytest.mark.parametrize("category", CATEGORIES)
def test_generate_sets_template_source(category):
    from src.attacks.template_attacks import generate_template_prompts

    prompts = generate_template_prompts(category, n=3)
    assert all(p.attack_source == "template" for p in prompts)


@pytest.mark.parametrize("category", CATEGORIES)
def test_generate_sets_direct_injection_strategy(category):
    from src.attacks.template_attacks import generate_template_prompts

    prompts = generate_template_prompts(category, n=3)
    assert all(p.attack_strategy == "direct_injection" for p in prompts)


def test_unsupported_category_raises():
    from src.attacks.template_attacks import generate_template_prompts

    with pytest.raises(ValueError, match="No templates defined"):
        generate_template_prompts("LLM99", n=5)


def test_too_few_templates_raises():
    from src.attacks.template_attacks import generate_template_prompts

    with pytest.raises(ValueError):
        generate_template_prompts("LLM01", n=100)
