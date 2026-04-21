from pathlib import Path

import pytest

from src.attacks.loader import AttackPrompt, load_attacks
from src.config import load_attacks_config

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── manual file sanity checks (unchanged) ───────────────────────────────────


def test_llm01_manual_prompts_file_exists():
    assert Path(REPO_ROOT / "src/attacks/manual/llm01.txt").exists()


def test_llm01_manual_prompts_count():
    lines = [
        line.strip()
        for line in Path(REPO_ROOT / "src/attacks/manual/llm01.txt").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 5, f"Expected 5 prompts, got {len(lines)}"


def test_llm01_manual_prompts_non_empty():
    lines = [
        line.strip()
        for line in Path(REPO_ROOT / "src/attacks/manual/llm01.txt").read_text().splitlines()
        if line.strip()
    ]
    assert all(len(line) > 10 for line in lines), "Every prompt must be >10 characters"


def test_llm06_manual_prompts_file_exists():
    assert Path(REPO_ROOT / "src/attacks/manual/llm06.txt").exists()


def test_llm06_manual_prompts_count():
    lines = [
        line.strip()
        for line in Path(REPO_ROOT / "src/attacks/manual/llm06.txt").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 5, f"Expected 5 prompts, got {len(lines)}"


def test_llm06_manual_prompts_non_empty():
    lines = [
        line.strip()
        for line in Path(REPO_ROOT / "src/attacks/manual/llm06.txt").read_text().splitlines()
        if line.strip()
    ]
    assert all(len(line) > 10 for line in lines), "Every prompt must be >10 characters"


# ── AttackPrompt dataclass ───────────────────────────────────────────────────


def test_attack_prompt_fields():
    ap = AttackPrompt(
        prompt="test prompt",
        attack_source="manual",
        attack_strategy="direct_injection",
    )
    assert ap.prompt == "test prompt"
    assert ap.attack_source == "manual"
    assert ap.attack_strategy == "direct_injection"


def test_attack_prompt_source_is_deepteam_literal():
    ap = AttackPrompt(
        prompt="dt prompt",
        attack_source="deepteam",
        attack_strategy="direct_injection",
    )
    assert ap.attack_source == "deepteam"


# ── mock fixture for deepteam ────────────────────────────────────────────────


@pytest.fixture()
def mock_deepteam(monkeypatch):
    """Patches generate_deepteam_prompts to avoid real OpenAI API calls."""
    fake = [
        AttackPrompt(prompt=f"dt{i}", attack_source="deepteam", attack_strategy="direct_injection")
        for i in range(5)
    ]
    monkeypatch.setattr(
        "src.attacks.deepteam_attacks.generate_deepteam_prompts",
        lambda *args, **kwargs: fake,
    )
    return fake


# ── load_attacks integration tests ──────────────────────────────────────────


def test_load_attacks_returns_list_of_attack_prompts(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    assert isinstance(attacks, list)
    assert all(isinstance(a, AttackPrompt) for a in attacks)


def test_load_attacks_sources_present(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    sources = {a.attack_source for a in attacks}
    assert sources == {"manual", "pyrit", "deepteam"}


def test_load_attacks_strategies_present(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    strategies = {a.attack_strategy for a in attacks}
    assert strategies == {"direct_injection", "indirect_injection"}


def test_load_attacks_manual_count(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    manual = [a for a in attacks if a.attack_source == "manual"]
    assert len(manual) == config.prompts_per_category.manual


def test_load_attacks_pyrit_count(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    pyrit_attacks = [a for a in attacks if a.attack_source == "pyrit"]
    assert len(pyrit_attacks) == config.prompts_per_category.pyrit


def test_load_attacks_deepteam_count(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    dt_attacks = [a for a in attacks if a.attack_source == "deepteam"]
    assert len(dt_attacks) == len(mock_deepteam)


def test_load_attacks_total_at_least_manual_plus_pyrit(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    assert len(attacks) >= config.prompts_per_category.manual + config.prompts_per_category.pyrit


def test_load_attacks_llm06(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM06", config)
    assert len(attacks) > 0
    assert all(isinstance(a, AttackPrompt) for a in attacks)


@pytest.mark.parametrize("category", ["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"])
def test_load_attacks_all_categories(mock_deepteam, category):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks(category, config)
    assert len(attacks) > 0
    assert all(isinstance(a, AttackPrompt) for a in attacks)


def test_load_attacks_single_converter_produces_correct_pyrit_count(mock_deepteam):
    from src.config import AttacksConfig, DeepTeamCategoryConfig, DeepTeamConfig, PromptsPerCategory

    config = AttacksConfig(
        seed=42,
        categories=["LLM01"],
        prompts_per_category=PromptsPerCategory(manual=5, pyrit=2),
        pyrit_converters=["base64"],
        deepteam=DeepTeamConfig(
            simulator_model="gpt-3.5-turbo-0125",
            categories={
                "LLM01": DeepTeamCategoryConfig(
                    vulnerabilities=["PromptLeakage"],
                    attacks_per_vulnerability_type=1,
                    attack_methods=["PromptInjection"],
                )
            },
        ),
    )
    attacks = load_attacks("LLM01", config)
    pyrit_attacks = [a for a in attacks if a.attack_source == "pyrit"]
    assert len(pyrit_attacks) == 2
