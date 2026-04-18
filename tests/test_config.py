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


def test_attacks_config_total_validated():
    from src.config import load_attacks_config

    config = load_attacks_config(CONFIGS_DIR / "attacks.yaml")
    p = config.prompts_per_category
    assert p.manual + p.pyrit + p.template == p.total


def test_judge_config_loads():
    from src.config import load_judge_config

    config = load_judge_config(CONFIGS_DIR / "judge.yaml")
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.0
    assert config.verdict_threshold == "borderline"


def test_prompts_per_category_invalid_total_raises():
    from src.config import PromptsPerCategory

    with pytest.raises(ValidationError, match="manual \\+ pyrit \\+ template must equal total"):
        PromptsPerCategory(manual=5, pyrit=15, template=10, total=999)


def test_model_config_missing_backend_raises():
    from src.config import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(name="some-model")  # backend is required
