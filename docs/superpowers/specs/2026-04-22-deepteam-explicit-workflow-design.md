# DeepTeam Explicit Workflow Design

**Date:** 2026-04-22
**Replaces:** `2026-04-21-real-deepteam-integration-design.md`
**Scope:** `src/attacks/deepteam_attacks.py`, `configs/attacks.yaml`, `src/config.py`, `prompts/deepteam_llm01.txt`, `tests/test_deepteam_attacks.py`

---

## Problem

The current `deepteam_attacks.py` delegates entirely to `AttackSimulator.simulate()`,
which runs baseline generation and technique enhancement as a single opaque pass with
random technique sampling. This makes the two phases invisible, hard to test in
isolation, and doesn't allow explicit technique-to-vulnerability assignment.

---

## Goal

Replace the current implementation with an explicit two-phase workflow:

1. Generate baseline attacks using `CustomVulnerability.simulate_attacks()`
2. Enhance each baseline prompt with a single, deterministically-assigned technique

Start with LLM01 only. All other categories' deepteam config is removed.

---

## Config (`configs/attacks.yaml`)

The `deepteam` block is simplified to a single LLM01 entry. `vulnerabilities`,
`attack_methods`, and `attacks_per_vulnerability_type` are removed. New fields:
`types`, `attacks_per_type`, `technique`, and `custom_prompt_file`.

```yaml
deepteam:
  simulator_model: gpt-3.5-turbo-0125
  categories:
    LLM01:
      types:
        - direct_override
        - policy_bypass
        - role_hijacking
        - hidden_instruction_following
        - instruction_priority_confusion
      attacks_per_type: 3
      technique: SystemOverride
      custom_prompt_file: prompts/deepteam_llm01.txt
```

One technique is applied uniformly to all types within a category. The technique name
is resolved to a class at runtime via `_TECHNIQUE_MAP`.

---

## Pydantic Model (`src/config.py`)

`DeepTeamCategoryConfig` is rewritten:

```python
class DeepTeamCategoryConfig(BaseModel):
    types: list[str]
    attacks_per_type: int
    technique: str           # resolved to single-turn attack class at runtime
    custom_prompt_file: str  # path relative to project root
```

`DeepTeamConfig` is unchanged except its `categories` values use the new model.

---

## Custom Prompt File (`prompts/deepteam_llm01.txt`)

The `custom_prompt` from the notebook is extracted to a versioned file. This mirrors the
existing convention for `prompts/judge_v1.txt`. The file is loaded once inside
`generate_deepteam_prompts()` and passed to `CustomVulnerability(custom_prompt=...)`.

---

## Code Structure (`src/attacks/deepteam_attacks.py`)

The file is rewritten around three functions. The public entry point signature is
unchanged so `loader.py` requires no edits.

### `_TECHNIQUE_MAP`

Replaces `_VULNERABILITY_MAP`, `_ATTACK_METHOD_MAP`, `_STRATEGY_MAP`, and
`_SYNTHETIC_CONTEXT_TARGET` (all deleted). Maps config strings to single-turn attack
classes:

```python
_TECHNIQUE_MAP: dict[str, type] = {
    "SystemOverride": SystemOverride,
    "PromptInjection": PromptInjection,
    "Roleplay": Roleplay,
    # other single-turn attacks as needed
}
```

### `_TECHNIQUE_STRATEGY_MAP`

A minimal two-entry dict that maps technique names to `attack_strategy` literals:

```python
_TECHNIQUE_STRATEGY_MAP = {
    "SyntheticContextInjection": "indirect_injection",
    # all others default to "direct_injection"
}
```

### Functions

```python
def generate_baseline_attacks(
    vulnerability: CustomVulnerability,
    purpose: str,
    attacks_per_type: int,
) -> list[RTTestCase]:
    """Call simulate_attacks() and return unenhanced test cases.
    Exceptions propagate — no silent swallowing at this layer."""

def enhance_attacks(
    test_cases: list[RTTestCase],
    technique: BaseSingleTurnAttack,
    technique_name: str,
    category: str,
    ignore_errors: bool = True,
) -> list[AttackPrompt]:
    """Call technique.enhance() per test case. Skip and warn on failure.
    Raise RuntimeError if result is empty."""

def generate_deepteam_prompts(
    category: str,
    config: DeepTeamCategoryConfig,
    simulator_model: str,
) -> list[AttackPrompt]:
    """Public entry point. Validates config, loads prompt file, builds
    CustomVulnerability, calls the two phases in sequence."""
```

---

## Error Handling

All validation happens in `generate_deepteam_prompts()` before any API calls:

- **Unknown technique**: `ValueError` with valid options listed
- **Missing prompt file**: `FileNotFoundError` with resolved absolute path

Inside `enhance_attacks()`:
- Each `technique.enhance()` call is wrapped in try/except
- Failed cases are skipped with a `logger.warning`
- If all cases fail, raise `RuntimeError`

`generate_baseline_attacks()` lets `simulate_attacks()` exceptions propagate — failures
at this layer are loud by design.

No retries, no fallbacks.

---

## Testing (`tests/test_deepteam_attacks.py`)

Five tests, all using monkeypatching (no real OpenAI API calls):

| Test | What is mocked | What is verified |
|---|---|---|
| `test_generate_baseline_attacks` | `CustomVulnerability.simulate_attacks` | Count, `vulnerability_type`, `input` fields |
| `test_enhance_attacks` | `technique.enhance` (returns `"enhanced: " + input`) | `attack_source="deepteam"`, correct `attack_strategy`, enhanced text |
| `test_generate_deepteam_prompts` | Both phases | Full pipeline: config read, prompt file loaded, both phases called, correct count and metadata |
| `test_unknown_technique_raises` | — | `ValueError` on bad technique name |
| `test_missing_prompt_file_raises` | — | `FileNotFoundError` on bad path |

`tests/test_loader.py` is unchanged — it already mocks `generate_deepteam_prompts` at
the module boundary.

---

## Files Changed

| File | Action |
|---|---|
| `configs/attacks.yaml` | Rewrite deepteam block — LLM01 only, new schema |
| `src/config.py` | Rewrite `DeepTeamCategoryConfig` |
| `src/attacks/deepteam_attacks.py` | Full rewrite |
| `prompts/deepteam_llm01.txt` | New file — custom prompt from notebook |
| `tests/test_deepteam_attacks.py` | Full rewrite |

## Files Unchanged

| File | Reason |
|---|---|
| `src/attacks/loader.py` | Public interface of `generate_deepteam_prompts` is unchanged |
| `tests/test_loader.py` | Already mocks at the right boundary |
| `prompts/judge_v1.txt` | Unrelated |
