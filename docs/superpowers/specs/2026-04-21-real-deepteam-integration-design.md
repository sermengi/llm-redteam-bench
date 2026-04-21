# Real DeepTeam Integration Design

**Date:** 2026-04-21
**Status:** Approved
**Scope:** Replace static `template_attacks.py` with live DeepTeam attack generation via `AttackSimulator.simulate()`

---

## Context

The `template` attack source was a placeholder: a static YAML file of hand-written prompts tagged as `attack_source="template"`. The real DeepTeam package (`deepteam==1.0.6`) was installed but unused. This spec covers replacing the placeholder with genuine DeepTeam-generated attacks.

The central design decision: use `AttackSimulator.simulate()` directly instead of the `red_team()` public API. `red_team()` bundles generation + model inference + evaluation into one call, which would waste gpt-4o evaluation API calls on results we discard. `AttackSimulator.simulate()` is the isolated generation step — no model callback, no evaluation, only prompt synthesis via the configured simulator LLM.

---

## Architecture Overview

The change is a targeted module swap. `load_attacks()` in `loader.py` aggregates three sources; DeepTeam becomes the real third source. Nothing upstream (pipeline, recorder, judge) changes — `AttackPrompt` objects flow through identically.

**Files changed:**

| Action | File |
|---|---|
| Delete | `src/attacks/template_attacks.py` |
| Delete | `src/attacks/deepteam_templates.yaml` |
| Create | `src/attacks/deepteam_attacks.py` |
| Update | `src/attacks/loader.py` — swap import; rename log label |
| Update | `src/config.py` — update `PromptsPerCategory`; add `DeepTeamConfig` models |
| Update | `configs/attacks.yaml` — add deepteam mapping section; remove `template`/`total` |
| Update | `src/attacks/loader.py` — `AttackPrompt.attack_source` Literal rename |
| Replace | `tests/test_template_attacks.py` → `tests/test_deepteam_attacks.py` |
| Update | `tests/test_loader.py` — rename `"template"` → `"deepteam"` in assertions |
| Update | `pyproject.toml` — pin `deepteam==1.0.6` |

---

## Config & Schema

### `configs/attacks.yaml`

```yaml
seed: 42
categories:
  - LLM01
  - LLM02
  - LLM04
  - LLM06
  - LLM07
  - LLM09

prompts_per_category:
  manual: 5
  pyrit: 5
  # deepteam count is implicit — derived from deepteam.categories config
  # total is computed at runtime and logged

pyrit_converters:
  - base64

deepteam:
  simulator_model: gpt-3.5-turbo-0125
  categories:
    LLM01:
      vulnerabilities: [PromptLeakage, IndirectInstruction]
      attacks_per_vulnerability_type: 1
      attack_methods: [PromptInjection, SyntheticContextInjection]
    LLM02:
      vulnerabilities: [Toxicity]
      attacks_per_vulnerability_type: 3
      attack_methods: [PromptInjection]
    LLM04:
      vulnerabilities: [Robustness]
      attacks_per_vulnerability_type: 5
      attack_methods: [PromptInjection]
    LLM06:
      vulnerabilities: [PIILeakage]
      attacks_per_vulnerability_type: 2
      attack_methods: [PromptInjection, GrayBox]
    LLM07:
      vulnerabilities: [SQLInjection, ShellInjection]
      attacks_per_vulnerability_type: 1
      attack_methods: [PromptInjection]
    LLM09:
      vulnerabilities: [Misinformation]
      attacks_per_vulnerability_type: 3
      attack_methods: [PromptInjection]
```

`attacks_per_vulnerability_type` counts per sub-type (not per vulnerability class). Each vulnerability class exposes N built-in sub-types; the total generated = sum(sub-types per class) × attacks_per_vulnerability_type. The config owner is responsible for ensuring this product is nonzero.

### Pydantic models (`src/config.py`)

```python
class PromptsPerCategory(BaseModel):
    manual: int
    pyrit: int
    # removed: deepteam, total — both are now runtime-derived

class DeepTeamCategoryConfig(BaseModel):
    vulnerabilities: list[str]               # class names, resolved at runtime
    attacks_per_vulnerability_type: int
    attack_methods: list[str]                # class names, resolved at runtime

class DeepTeamConfig(BaseModel):
    simulator_model: str = "gpt-3.5-turbo-0125"
    categories: dict[str, DeepTeamCategoryConfig]

class AttacksConfig(BaseModel):
    seed: int
    categories: list[str]
    prompts_per_category: PromptsPerCategory
    pyrit_converters: list[str] | None = None
    deepteam: DeepTeamConfig
```

### `AttackPrompt` Literal update (`src/attacks/loader.py`)

```python
attack_source: Literal["pyrit", "deepteam", "manual"]  # "template" removed
```

---

## `deepteam_attacks.py` Module

### Lookup maps (module-level)

```python
_VULNERABILITY_MAP: dict[str, type] = {
    "PromptLeakage": PromptLeakage,
    "IndirectInstruction": IndirectInstruction,
    "PIILeakage": PIILeakage,
    "Misinformation": Misinformation,
    "Robustness": Robustness,
    "Toxicity": Toxicity,
    "SQLInjection": SQLInjection,
    "ShellInjection": ShellInjection,
}

_ATTACK_METHOD_MAP: dict[str, type] = {
    "PromptInjection": PromptInjection,
    "SyntheticContextInjection": SyntheticContextInjection,
    "GrayBox": GrayBox,
}

_STRATEGY_MAP: dict[str, str] = {
    "PromptInjection": "direct_injection",
    "SyntheticContextInjection": "indirect_injection",
    "GrayBox": "direct_injection",
    # default for unmapped: "direct_injection"
}
```

String names from config are resolved through these maps at call time. Unknown names raise `ValueError` immediately, before any API call is made.

### Public function

```python
def generate_deepteam_prompts(
    category: str,
    config: DeepTeamCategoryConfig,
    simulator_model: str,
) -> list[AttackPrompt]:
```

**Internal flow:**

1. Resolve vulnerability names → instantiate each with `simulator_model=simulator_model`
2. Resolve attack method names → instantiate each
3. Call `AttackSimulator(async_mode=False).simulate(vulnerabilities, attacks, attacks_per_vulnerability_type, ignore_errors=True)`
4. For each `RTTestCase`: skip if `.input is None` or `.error is not None` (log warning for skipped)
5. Map `.attack_method` → strategy via `_STRATEGY_MAP` (default `"direct_injection"`)
6. If zero valid prompts result: raise `RuntimeError`
7. Return `list[AttackPrompt]` with `attack_source="deepteam"`

### `loader.py` changes

Replace `generate_template_prompts` call with:

```python
from src.attacks.deepteam_attacks import generate_deepteam_prompts

deepteam_prompts = generate_deepteam_prompts(
    category,
    config=config.deepteam.categories[category],
    simulator_model=config.deepteam.simulator_model,
)
all_attacks = manual + pyrit + deepteam_prompts
logger.info(
    "Loaded %d attacks for %s (manual=%d pyrit=%d deepteam=%d)",
    len(all_attacks), category, len(manual), len(pyrit), len(deepteam_prompts),
)
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Unknown vulnerability/attack name in config | `ValueError` before any API call |
| `OPENAI_API_KEY` not set | Propagates naturally from deepteam — not caught |
| Single test case generation failure | `ignore_errors=True` — skip + log warning |
| Zero valid prompts produced | `RuntimeError` — config or API problem, must be loud |
| `RTTestCase.input is None` | Skip silently |

No retries, no fallback to static templates.

---

## Testing

`tests/test_deepteam_attacks.py` replaces `tests/test_template_attacks.py`. All tests mock `AttackSimulator.simulate` to avoid real API calls.

**Test cases:**
- All returned prompts have `attack_source == "deepteam"`
- Strategy correctly mapped from `RTTestCase.attack_method`
- Unknown vulnerability name raises `ValueError`
- Unknown attack method name raises `ValueError`
- Test cases with `.error` set are skipped
- Test cases with `.input is None` are skipped
- Zero valid prompts after filtering raises `RuntimeError`

**`tests/test_loader.py` updates:**
- `"template"` → `"deepteam"` in all source assertions
- `test_load_attacks_sources_present` asserts `{"manual", "pyrit", "deepteam"}`
- Remove `test_load_attacks_total_count` (no fixed total) — add check that `len(all) >= manual + pyrit`
- Mock `generate_deepteam_prompts` (not `AttackSimulator`) in loader tests — loader tests verify the loader boundary, not deepteam internals

---

## Constraints

- `deepteam==1.0.6` pinned in `pyproject.toml` — `AttackSimulator` is an internal class; pin prevents silent breakage on upgrades
- `OPENAI_API_KEY` must be set in environment — no default, no fallback
- `async_mode=False` used throughout — avoids event loop conflicts with PyRIT's own async handling in the same process
