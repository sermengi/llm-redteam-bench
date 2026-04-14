from pathlib import Path

from src.attacks.loader import AttackPrompt, load_attacks
from src.config import load_attacks_config

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_attack_prompt_fields():
    ap = AttackPrompt(
        prompt="test prompt",
        attack_source="manual",
        attack_strategy="direct_injection",
    )
    assert ap.prompt == "test prompt"
    assert ap.attack_source == "manual"
    assert ap.attack_strategy == "direct_injection"


def test_load_attacks_returns_list_of_attack_prompts():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    assert isinstance(attacks, list)
    assert all(isinstance(a, AttackPrompt) for a in attacks)


def test_load_attacks_total_count():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    assert len(attacks) == config.prompts_per_category.total


def test_load_attacks_sources_present():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    sources = {a.attack_source for a in attacks}
    assert sources == {"manual", "pyrit", "deepteam"}


def test_load_attacks_strategies_present():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    strategies = {a.attack_strategy for a in attacks}
    assert strategies == {"direct_injection", "indirect_injection"}


def test_load_attacks_manual_count():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    manual = [a for a in attacks if a.attack_source == "manual"]
    assert len(manual) == config.prompts_per_category.manual


def test_load_attacks_pyrit_count():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    pyrit_attacks = [a for a in attacks if a.attack_source == "pyrit"]
    assert len(pyrit_attacks) == config.prompts_per_category.pyrit


def test_load_attacks_deepteam_count():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    deepteam_attacks = [a for a in attacks if a.attack_source == "deepteam"]
    assert len(deepteam_attacks) == config.prompts_per_category.deepteam


def test_load_attacks_llm06():
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM06", config)
    assert len(attacks) == config.prompts_per_category.total
