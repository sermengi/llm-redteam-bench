import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.attacks.loader import (
    AttackPrompt,
    load_attacks,
    load_cached_prompts,
    resolve_attacks,
    save_prompts,
)
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


def test_load_attacks_llm01(mock_deepteam):
    config = load_attacks_config(Path(REPO_ROOT / "configs/attacks.yaml"))
    attacks = load_attacks("LLM01", config)
    assert len(attacks) > 0
    assert all(isinstance(a, AttackPrompt) for a in attacks)


@pytest.mark.parametrize("category", ["LLM01"])
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
                    types=["direct_override"],
                    attacks_per_type=1,
                    technique="SystemOverride",
                    custom_prompt_file="prompts/deepteam_llm01.txt",
                )
            },
        ),
    )
    attacks = load_attacks("LLM01", config)
    pyrit_attacks = [a for a in attacks if a.attack_source == "pyrit"]
    assert len(pyrit_attacks) == 2


# ── save_prompts ─────────────────────────────────────────────────────────────


def test_save_prompts_creates_file(tmp_path):
    prompts = [
        AttackPrompt(prompt="p1", attack_source="manual", attack_strategy="direct_injection"),
        AttackPrompt(prompt="p2", attack_source="pyrit", attack_strategy="indirect_injection"),
    ]
    path = save_prompts(prompts, "LLM01", "abc123", tmp_path)
    assert path.exists()
    assert path == tmp_path / "LLM01.json"


def test_save_prompts_file_content(tmp_path):
    prompts = [
        AttackPrompt(prompt="hello", attack_source="manual", attack_strategy="direct_injection"),
    ]
    save_prompts(prompts, "LLM01", "deadbeef", tmp_path)
    data = json.loads((tmp_path / "LLM01.json").read_text())
    assert data["category"] == "LLM01"
    assert data["config_hash"] == "deadbeef"
    assert len(data["prompts"]) == 1
    assert data["prompts"][0]["prompt"] == "hello"


# ── load_cached_prompts ──────────────────────────────────────────────────────


def test_load_cached_prompts_returns_none_if_missing(tmp_path):
    result = load_cached_prompts("LLM01", "abc123", tmp_path)
    assert result is None


def test_load_cached_prompts_returns_none_on_hash_mismatch(tmp_path):
    prompts = [
        AttackPrompt(prompt="x", attack_source="manual", attack_strategy="direct_injection"),
    ]
    save_prompts(prompts, "LLM01", "hash-v1", tmp_path)
    result = load_cached_prompts("LLM01", "hash-v2", tmp_path)
    assert result is None


def test_load_cached_prompts_round_trip(tmp_path):
    original = [
        AttackPrompt(
            prompt="inject me", attack_source="pyrit", attack_strategy="indirect_injection"
        ),
        AttackPrompt(
            prompt="ignore sys", attack_source="manual", attack_strategy="direct_injection"
        ),
    ]
    save_prompts(original, "LLM06", "cafebabe", tmp_path)
    loaded = load_cached_prompts("LLM06", "cafebabe", tmp_path)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].prompt == "inject me"
    assert loaded[0].attack_source == "pyrit"
    assert loaded[0].attack_strategy == "indirect_injection"
    assert loaded[1].prompt == "ignore sys"


def test_load_cached_prompts_returns_none_on_corrupt_file(tmp_path):
    (tmp_path / "LLM01.json").write_text("this is not valid json {{{")
    result = load_cached_prompts("LLM01", "any-hash", tmp_path)
    assert result is None


# ── resolve_attacks ───────────────────────────────────────────────────────────


def test_resolve_attacks_generates_and_caches_on_miss(tmp_path, mock_deepteam):
    config = load_attacks_config(REPO_ROOT / "configs/attacks.yaml")
    prompts = resolve_attacks("LLM01", config, "hash-abc", tmp_path)
    assert len(prompts) > 0
    assert (tmp_path / "LLM01.json").exists()


def test_resolve_attacks_loads_from_cache_on_hit(tmp_path, mock_deepteam):
    config = load_attacks_config(REPO_ROOT / "configs/attacks.yaml")
    original = resolve_attacks("LLM01", config, "hash-xyz", tmp_path)
    with patch("src.attacks.loader.load_attacks") as mock_load:
        cached = resolve_attacks("LLM01", config, "hash-xyz", tmp_path)
    mock_load.assert_not_called()
    assert len(cached) == len(original)


def test_resolve_attacks_regenerates_on_stale_hash(tmp_path, mock_deepteam):
    config = load_attacks_config(REPO_ROOT / "configs/attacks.yaml")
    resolve_attacks("LLM01", config, "hash-v1", tmp_path)
    with patch("src.attacks.loader.load_attacks", wraps=load_attacks) as mock_load:
        resolve_attacks("LLM01", config, "hash-v2", tmp_path)
    mock_load.assert_called_once()
    cached_hash = json.loads((tmp_path / "LLM01.json").read_text())["config_hash"]
    assert cached_hash == "hash-v2"
