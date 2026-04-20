"""Pydantic config models and YAML loaders for all three config files."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Configuration for a single target model."""

    name: str
    backend: str
    mock: bool = False


class ModelsConfig(BaseModel):
    """Top-level models.yaml schema."""

    models: list[ModelConfig]


class PromptsPerCategory(BaseModel):
    """Attack prompt counts per source. Validated to sum to total."""

    manual: int
    pyrit: int
    template: int
    total: int

    @model_validator(mode="after")
    def validate_total(self) -> "PromptsPerCategory":
        """Ensure manual + pyrit + template == total."""
        computed = self.manual + self.pyrit + self.template
        if computed != self.total:
            raise ValueError(
                f"manual + pyrit + template must equal total: "
                f"{self.manual} + {self.pyrit} + {self.template} = {computed} != {self.total}"
            )
        return self


# Valid PyRIT converter names. Extend this Literal and _CONVERTER_MAP in
# pyrit_attacks.py together when adding new converters.
PyritConverterName = Literal["base64", "rot13", "leetspeak"]


class AttacksConfig(BaseModel):
    """Top-level attacks.yaml schema."""

    seed: int
    categories: list[str]
    prompts_per_category: PromptsPerCategory
    pyrit_converters: list[PyritConverterName] = Field(
        default_factory=lambda: ["base64", "rot13", "leetspeak"],
        min_length=1,
    )


class JudgeConfig(BaseModel):
    """Top-level judge.yaml schema."""

    model: str
    prompt_version: str
    temperature: float
    verdict_threshold: Literal["safe", "borderline", "unsafe"]


def load_models_config(path: Path) -> ModelsConfig:
    """Load and validate models.yaml."""
    with open(path) as f:
        return ModelsConfig(**yaml.safe_load(f))


def load_attacks_config(path: Path) -> AttacksConfig:
    """Load and validate attacks.yaml."""
    with open(path) as f:
        return AttacksConfig(**yaml.safe_load(f))


def load_judge_config(path: Path) -> JudgeConfig:
    """Load and validate judge.yaml."""
    with open(path) as f:
        return JudgeConfig(**yaml.safe_load(f))


class SystemPromptsConfig(BaseModel):
    """Top-level system_prompts.yaml schema."""

    default: str
    categories: dict[str, str] = {}


def load_system_prompts_config(path: Path) -> SystemPromptsConfig:
    """Load and validate system_prompts.yaml."""
    with open(path) as f:
        return SystemPromptsConfig(**yaml.safe_load(f))
