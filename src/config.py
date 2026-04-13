"""Pydantic config models and YAML loaders for all three config files."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


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
    deepteam: int
    total: int

    @model_validator(mode="after")
    def validate_total(self) -> "PromptsPerCategory":
        """Ensure manual + pyrit + deepteam == total."""
        computed = self.manual + self.pyrit + self.deepteam
        if computed != self.total:
            raise ValueError(
                f"manual + pyrit + deepteam must equal total: "
                f"{self.manual} + {self.pyrit} + {self.deepteam} = {computed} != {self.total}"
            )
        return self


class AttacksConfig(BaseModel):
    """Top-level attacks.yaml schema."""

    seed: int
    categories: list[str]
    prompts_per_category: PromptsPerCategory


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
