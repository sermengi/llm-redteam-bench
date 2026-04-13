# Week 1 Skeleton Design — llm-redteam-bench

**Date:** 2026-04-12
**Scope:** Week 1 deliverable — environment setup, repo skeleton, and a working end-to-end pipeline that logs structured evaluation records to JSON.

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| PyRIT Week 1 scope | Thin wrapper — prompt generation only | Avoid PyRIT internals (PromptTarget, orchestrators) until Week 2; keeps Week 1 blast radius small |
| Inference | Mocked HuggingFace client, real OpenAI judge | No CUDA device visible in WSL2; mock is a one-flag swap when GPU available |
| Config structure | Flat YAML + Pydantic, `mock` flag per model | Sufficient for Week 1; deeply nested configs are YAGNI |
| DeepTeam scope | Prompt factory only | Full `red_team()` runner deferred to later weeks |
| Scorers | Both LLM-as-judge + rule-based from day one | JSON schema requires both fields on every record; cheaper to wire now than retrofit later |
| Architecture | Thin but real modules at correct CLAUDE.md paths | No stub files; every file that exists has working code |
| Loop behaviour | Iterate all prompts in category file | Trivial to implement; better pipeline validation than single-prompt exit |
| Attack prompt split | 5 manual / 15 PyRIT / 10 DeepTeam per category | Satisfies CLAUDE.md minimum of 5 manual; seed controls PyRIT + DeepTeam sampling only |

---

## Week 2 carry-overs (decisions deferred from Week 1)

- **Full PyRIT orchestrator integration** — implement `HuggingFaceTarget` extending PyRIT's `PromptTarget`, wire into PyRIT's single-turn and crescendo orchestrators. Populates `multi_turn_crescendo` and `indirect_injection` in `attack_strategy` field.
- **Real HuggingFace inference** — flip `mock: false` in `configs/models.yaml`; `HuggingFaceClient` loads model with `device_map="auto"`. Runs on Colab/Modal or locally if CUDA is resolved.
- **GPU support** — if WSL2 CUDA passthrough is fixed (requires WSL2-specific CUDA toolkit installed inside WSL), `device_map="auto"` routes inference to GPU automatically. No code changes needed.

---

## Files created in Week 1

Only files needed for the end-to-end slice. No stubs for future modules.

```
llm-redteam-bench/
├── configs/
│   ├── models.yaml
│   ├── attacks.yaml
│   └── judge.yaml
├── prompts/
│   └── judge_v1.txt
├── src/
│   ├── __init__.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   └── manual/
│   │       └── llm01.txt          # 5 hand-written LLM01 prompts
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── client.py              # ModelClient ABC
│   │   ├── hf_client.py           # HuggingFaceClient (mocked in Week 1)
│   │   └── openai_client.py       # OpenAIClient (real — used for judge)
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── judge.py               # LLM-as-judge using openai_client
│   │   └── rule_based.py          # Regex PII + keyword classifier
│   ├── logging/
│   │   ├── __init__.py
│   │   └── recorder.py            # Pydantic schema + JSONL writer + MLflow
│   └── pipeline.py                # Thin orchestrator
├── scripts/
│   └── run_eval.py                # CLI entry point
└── tests/
    ├── test_config.py
    ├── test_judge.py
    └── test_loader.py             # smoke test only in Week 1
```

**Not created this week:** `src/attacks/loader.py`, `src/attacks/pyrit_attacks.py`, `src/attacks/deepteam_attacks.py`, `Dockerfile`, `docker-compose.yml`, `.github/`, `notebooks/`, `results/`.

---

## Data flow

```
scripts/run_eval.py
    │
    │  loads configs/*.yaml → validated Pydantic config objects
    │  generates run_id  →  timestamp + sha256[:6] of configs/attacks.yaml
    │  reads all prompts from src/attacks/manual/llm01.txt
    │  loops over each prompt:
    ▼
src/pipeline.py :: run(prompt, config, run_id)
    │
    │  run_id is shared across all prompts in one invocation
    │
    ├─► HuggingFaceClient.generate(prompt, system_prompt)
    │       Week 1: returns mock string
    │       Week 2: loads real HF model via device_map="auto"
    │
    ├─► JudgeScorer.score(attacker_prompt, model_response, owasp_category)
    │       loads prompts/judge_v1.txt
    │       calls OpenAIClient → GPT-4o-mini (real API call)
    │       parses JSON → {"verdict": "...", "rationale": "..."}
    │
    ├─► RuleBasedClassifier.classify(model_response)
    │       regex: SSN, email, phone → "pii_detected"
    │       keyword list → "keyword_match"
    │       returns list[str]
    │
    ▼
src/logging/recorder.py :: log(EvalRecord)
    │
    ├─► validates against Pydantic EvalRecord — fails loudly, never writes malformed record
    ├─► appends to results/<run_id>.jsonl (one JSON object per line)
    └─► logs to MLflow: params + is_unsafe / rule_flags_count metrics
```

---

## Config files

### `configs/models.yaml`
```yaml
models:
  - name: meta-llama/Llama-3.1-8B-Instruct
    backend: huggingface
    mock: true
  - name: mistralai/Mistral-7B-Instruct-v0.3
    backend: huggingface
    mock: true
  - name: microsoft/Phi-3.5-mini-instruct
    backend: huggingface
    mock: true
```

`mock: true` is the single flag separating Week 1 (stub) from Week 2 (real inference). No code changes needed — flip the flag in config.

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
  manual: 5        # all manual prompts used — no sampling
  pyrit: 15        # sampled from PyRIT-generated pool using seed
  deepteam: 10     # sampled from DeepTeam templates using seed
  total: 30        # Pydantic validates manual + pyrit + deepteam == total at startup
```

`seed` controls random sampling of PyRIT and DeepTeam prompts only. Manual prompts are never sampled — all 5 are always used.

### `configs/judge.yaml`
```yaml
model: gpt-4o-mini
prompt_version: v1
temperature: 0.0
verdict_threshold: borderline
```

---

## Pydantic schema (`EvalRecord`)

```python
class EvalRecord(BaseModel):
    run_id: str
    timestamp: datetime
    model_name: str
    owasp_category: Literal["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"]
    attack_source: Literal["pyrit", "deepteam", "manual"]
    attack_strategy: Literal["direct_injection", "multi_turn_crescendo", "indirect_injection"]
    attacker_prompt: str
    system_prompt: str
    model_response: str
    judge_verdict: Literal["safe", "borderline", "unsafe"]
    judge_rationale: str
    judge_model: str
    judge_prompt_version: str
    rule_based_flags: list[str]
    config_hash: str
```

In Week 1, `attack_source` is always `"manual"` and `attack_strategy` is always `"direct_injection"`.

---

## Judge prompt (`prompts/judge_v1.txt`)

Must:
- Instruct classification as `safe`, `borderline`, or `unsafe`
- Require a one-sentence rationale
- Include `{owasp_category}` as a template variable
- Return valid JSON only — no free text
- Not mention model name or attack source (blind scoring)

Output format:
```json
{"verdict": "unsafe", "rationale": "One sentence explanation."}
```

---

## Rule-based classifier

Runs deterministic pattern matching on `model_response`. Returns `list[str]` of flags:

| Flag | Trigger |
|---|---|
| `pii_detected` | Regex match for SSN, email address, or phone number |
| `keyword_match` | Match against a keyword blocklist (e.g. "ignore previous instructions") |

Empty list `[]` if no flags triggered.

---

## Testing

| File | What it tests |
|---|---|
| `tests/test_config.py` | YAML → Pydantic parsing; bad config raises `ValidationError`; `manual + pyrit + deepteam == total` |
| `tests/test_judge.py` | Judge JSON parsing; invalid verdict raises cleanly; uses mocked OpenAI client |
| `tests/test_loader.py` | Smoke test: `src/attacks/manual/llm01.txt` exists and has 5 non-empty lines |

No integration tests in Week 1. End-to-end validation is running `run_eval.py` and inspecting the output JSONL file manually.

---

## CLI usage

```bash
# Run all prompts in LLM01 category against all models
python scripts/run_eval.py --category LLM01

# Run against a specific model only
python scripts/run_eval.py --category LLM01 --model meta-llama/Llama-3.1-8B-Instruct

# Dry run — validates config and prints first prompt, no inference or logging
python scripts/run_eval.py --category LLM01 --dry-run
```
