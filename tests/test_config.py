from pathlib import Path

import pytest
from pydantic import ValidationError

CONFIGS_DIR = Path("configs")


def test_models_config_loads():
    from src.config import load_models_config

    config = load_models_config(CONFIGS_DIR / "models.yaml")
    assert len(config.models) == 3
    assert all(m.backend == "huggingface" for m in config.models)
    assert all(m.mock is True for m in config.models)


def test_attacks_config_loads():
    from src.config import load_attacks_config

    config = load_attacks_config(CONFIGS_DIR / "attacks.yaml")
    assert config.seed == 42
    assert set(config.categories) == {"LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"}


def test_judge_config_loads():
    from src.config import load_judge_config

    config = load_judge_config(CONFIGS_DIR / "judge.yaml")
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.0
    assert config.verdict_threshold == "borderline"


def test_model_config_missing_backend_raises():
    from src.config import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(name="some-model")  # backend is required


def test_attacks_config_default_pyrit_converters():
    from src.config import load_attacks_config

    config = load_attacks_config(CONFIGS_DIR / "attacks.yaml")
    assert config.pyrit_converters == ["base64"]


def test_attacks_config_invalid_converter_raises():
    from pydantic import ValidationError

    from src.config import (
        AttacksConfig,
        DeepTeamCategoryConfig,
        DeepTeamConfig,
        PromptsPerCategory,
    )

    with pytest.raises(ValidationError, match="not_a_real_converter"):
        AttacksConfig(
            seed=42,
            categories=["LLM01"],
            prompts_per_category=PromptsPerCategory(manual=5, pyrit=5),
            pyrit_converters=["not_a_real_converter"],
            deepteam=DeepTeamConfig(
                simulator_model="gpt-3.5-turbo-0125",
                categories={
                    "LLM01": DeepTeamCategoryConfig(
                        types=["direct_override"],
                        attacks_per_type=1,
                        technique="SystemOverride",
                        custom_prompt_file="prompts/deepteam_llm01.txt",
                    )
                },
            ),
        )


def test_attacks_config_pyrit_converters_default_when_omitted():
    from src.config import (
        AttacksConfig,
        DeepTeamCategoryConfig,
        DeepTeamConfig,
        PromptsPerCategory,
    )

    config = AttacksConfig(
        seed=42,
        categories=["LLM01"],
        prompts_per_category=PromptsPerCategory(manual=5, pyrit=5),
        deepteam=DeepTeamConfig(
            simulator_model="gpt-3.5-turbo-0125",
            categories={
                "LLM01": DeepTeamCategoryConfig(
                    types=["direct_override"],
                    attacks_per_type=1,
                    technique="SystemOverride",
                    custom_prompt_file="prompts/deepteam_llm01.txt",
                )
            },
        ),
    )
    assert config.pyrit_converters == ["base64", "rot13", "leetspeak"]


def test_attacks_config_empty_pyrit_converters_raises():
    from pydantic import ValidationError

    from src.config import (
        AttacksConfig,
        DeepTeamCategoryConfig,
        DeepTeamConfig,
        PromptsPerCategory,
    )

    with pytest.raises(ValidationError):
        AttacksConfig(
            seed=42,
            categories=["LLM01"],
            prompts_per_category=PromptsPerCategory(manual=5, pyrit=5),
            pyrit_converters=[],
            deepteam=DeepTeamConfig(
                simulator_model="gpt-3.5-turbo-0125",
                categories={
                    "LLM01": DeepTeamCategoryConfig(
                        types=["direct_override"],
                        attacks_per_type=1,
                        technique="SystemOverride",
                        custom_prompt_file="prompts/deepteam_llm01.txt",
                    )
                },
            ),
        )


def test_deepteam_category_config_parses():
    from src.config import DeepTeamCategoryConfig

    c = DeepTeamCategoryConfig(
        types=["direct_override", "policy_bypass"],
        attacks_per_type=3,
        technique="SystemOverride",
        custom_prompt_file="prompts/deepteam_llm01.txt",
    )
    assert c.types == ["direct_override", "policy_bypass"]
    assert c.attacks_per_type == 3
    assert c.technique == "SystemOverride"
    assert c.custom_prompt_file == "prompts/deepteam_llm01.txt"


def test_deepteam_config_parses():
    from src.config import DeepTeamCategoryConfig, DeepTeamConfig

    dc = DeepTeamConfig(
        simulator_model="gpt-3.5-turbo-0125",
        categories={
            "LLM01": DeepTeamCategoryConfig(
                types=["direct_override"],
                attacks_per_type=1,
                technique="SystemOverride",
                custom_prompt_file="prompts/deepteam_llm01.txt",
            )
        },
    )
    assert dc.simulator_model == "gpt-3.5-turbo-0125"
    assert "LLM01" in dc.categories


def test_prompts_per_category_has_no_total():
    from src.config import PromptsPerCategory

    p = PromptsPerCategory(manual=5, pyrit=5)
    assert p.manual == 5
    assert p.pyrit == 5
    assert not hasattr(p, "total")
    assert not hasattr(p, "template")


def test_attacks_config_loads_deepteam_section():
    from pathlib import Path

    from src.config import load_attacks_config

    config = load_attacks_config(Path("configs/attacks.yaml"))
    assert config.deepteam.simulator_model == "gpt-3.5-turbo-0125"
    assert set(config.deepteam.categories.keys()) == {"LLM01"}
    llm01 = config.deepteam.categories["LLM01"]
    assert "direct_override" in llm01.types
    assert llm01.attacks_per_type == 3
    assert llm01.technique == "SystemOverride"
    assert llm01.custom_prompt_file == "prompts/deepteam_llm01.txt"
