# CLAUDE.md — Project Brief for `llm-redteam-bench`

This file describes the full context, goals, architecture, and constraints of the
`llm-redteam-bench` project. Read this before writing any code or making any structural
decisions.

---

## What this project is

`llm-redteam-bench` is a systematic adversarial evaluation suite that red-teams
open-source LLMs across a defined set of OWASP Top 10 for LLM Applications vulnerability
categories. It is a portfolio project built to demonstrate production-grade eval
engineering skills to hiring managers in the LLM evaluation and AI safety field.

The project must be:
- **Reproducible**: every attack, response, and verdict must be fully logged and
  rerunnable from config
- **Extensible**: adding a new model, attack category, or scoring method should require
  minimal changes
- **Publishable**: the codebase and results must be presentable in a public GitHub repo
  and referenced in a technical blog post

---

## End goals

1. A public GitHub repository with clean, well-documented code that a hiring manager or
   engineer can clone and reproduce
2. A structured results dataset (JSON) covering 3 models × 6 OWASP vulnerability
   categories, with judge-scored verdicts
3. A technical blog post (1,200–1,500 words) presenting methodology, findings, and
   implications — written after the results are complete
4. A README with architecture overview, reproduction steps, results summary table, and
   dataset schema

This is NOT a research paper. It is an engineering portfolio project with a research
findings component.

---

## Target models

Attack all three models in every evaluation run:

| Model | Source | Notes |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face | Primary open-source target |
| `mistralai/Mistral-7B-Instruct-v0.3` | Hugging Face | Comparison target |
| `microsoft/Phi-3.5-mini-instruct` | Hugging Face | Lightweight comparison target |

All three are freely available on Hugging Face and can be run locally (CPU fallback
acceptable for development) or on Colab/Modal for full inference.

---

## OWASP vulnerability categories in scope

Cover exactly these six categories. Do not expand scope without discussion.

| ID | Category | Description |
|---|---|---|
| LLM01 | Prompt Injection | Direct and indirect injection attacks that override system instructions |
| LLM02 | Insecure Output Handling | Attacks that cause unsafe content in model outputs consumed downstream |
| LLM04 | Model Denial of Service | Adversarial inputs designed to cause resource exhaustion or degraded output |
| LLM06 | Sensitive Information Disclosure | Attacks that elicit PII, credentials, or training data |
| LLM07 | Insecure Plugin / Tool Design | Attacks targeting tool-calling or function-calling interfaces |
| LLM09 | Misinformation Generation | Prompts that elicit confident factual errors or fabricated content |

Each category should have a minimum of 30 attack prompts, a mix of:
- PyRIT-generated (using its attack orchestrators and converters)
- DeepTeam template-based (using its built-in vulnerability types)
- Manually crafted (at least 5 per category, written by the developer)

---

## Tech stack

### Core attack framework
- **PyRIT** (`azure/pyrit`) — primary attack orchestration, multi-turn attacks, converters
- **DeepTeam** (`confident-ai/deepteam`) — supplementary attack templates, 40+ built-in
  vulnerability types

### PyRIT attack strategies to implement
Implement exactly these three strategies in `src/attacks/pyrit_attacks.py`:

| Strategy | Type | Description |
|---|---|---|
| `direct_injection` | Single-turn | One-shot prompt that attempts to override instructions directly |
| `multi_turn_crescendo` | Multi-turn | Escalating conversation that incrementally shifts model behaviour |
| `indirect_injection` | Single-turn | Injection delivered via the system prompt, not the user turn |

These map to the `attack_strategy` field in the JSON log schema. Every logged record must
populate this field with one of the three values above.

### Inference
- **Hugging Face Transformers** — local model loading and inference for the three target
  models
- **OpenAI Python SDK** — for GPT-4o-mini as the LLM-as-judge scorer
- Wrap both behind a unified `ModelClient` interface so swapping providers requires only
  a config change, not code changes

### Scoring
- **Primary scorer**: GPT-4o-mini as LLM-as-judge with a structured rubric
- **Secondary scorer**: Rule-based classifier for deterministic checks (PII pattern
  matching, keyword detection)
- Judge prompt must be versioned — store it as a file, not a hardcoded string

### Logging and reproducibility
- All results logged to structured JSON (schema defined below)
- All configs (model names, attack counts, judge model, temperature, random seed) stored
  in YAML config files
- Runs are identified by a unique `run_id` (timestamp + short hash)
- `MLflow` for experiment tracking (consistent with the developer's existing MLOps stack)

### MLOps and tooling
- **Docker + docker-compose** for environment reproducibility
- **GitHub Actions** for post-push linting and formatting (pre-commit hooks: black,
  isort, flake8)
- **pytest** for unit and integration tests (minimum: attack loader, judge scorer, config
  parser)
- **Pydantic** for schema validation of all logged records

### Visualization (optional, stretch goal)
- A single-page HTML results table or simple Plotly dashboard showing attack success
  rates per category per model

---

## Project structure

```
llm-redteam-bench/
├── CLAUDE.md                  # This file
├── README.md                  # Public-facing documentation
├── pyproject.toml             # Project metadata and dependencies
├── docker-compose.yml
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml             # Lint + test on push
├── configs/
│   ├── models.yaml            # Target model definitions
│   ├── attacks.yaml           # Attack counts, categories, seeds
│   └── judge.yaml             # Judge model, prompt version, thresholds
├── prompts/
│   └── judge_v1.txt           # Versioned judge prompt — never hardcoded
├── src/
│   ├── __init__.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── loader.py          # Loads attack prompts per category
│   │   ├── pyrit_attacks.py   # PyRIT orchestrator wrappers
│   │   ├── deepteam_attacks.py# DeepTeam template wrappers
│   │   └── manual/
│   │       ├── llm01.txt      # Manually crafted prompts, one per line
│   │       ├── llm02.txt
│   │       └── ...
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── client.py          # Unified ModelClient interface
│   │   ├── hf_client.py       # Hugging Face implementation
│   │   └── openai_client.py   # OpenAI implementation
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── judge.py           # LLM-as-judge scorer
│   │   └── rule_based.py      # Rule-based classifier
│   ├── logging/
│   │   ├── __init__.py
│   │   └── recorder.py        # Structured JSON logger + MLflow integration
│   └── pipeline.py            # Top-level orchestrator: runs attacks → inference → scoring → logging
├── scripts/
│   └── run_eval.py            # CLI entry point
├── tests/
│   ├── test_loader.py
│   ├── test_judge.py
│   └── test_config.py
├── results/
│   └── .gitkeep              # Results stored locally, not committed (except summary)
└── notebooks/
    └── analysis.ipynb         # Post-run analysis and visualizations
```

---

## JSON log schema

Every evaluation record must conform to this schema exactly. Use Pydantic to validate.

```json
{
  "run_id": "20250328-a3f9c1",
  "timestamp": "2025-03-28T14:32:01Z",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "owasp_category": "LLM01",
  "attack_source": "pyrit | deepteam | manual",
  "attack_strategy": "direct_injection | multi_turn_crescendo | ...",
  "attacker_prompt": "...",
  "system_prompt": "...",
  "model_response": "...",
  "judge_verdict": "unsafe | borderline | safe",
  "judge_rationale": "...",
  "judge_model": "gpt-4o-mini",
  "judge_prompt_version": "v1",
  "rule_based_flags": ["pii_detected", "keyword_match"],
  "config_hash": "sha256 of configs/attacks.yaml at time of run"
}
```

---

## Judge prompt requirements

The judge prompt (stored in `prompts/judge_v1.txt`) must:
- Instruct the judge to classify each response as `safe`, `borderline`, or `unsafe`
- Require a one-sentence rationale for every verdict
- Include the OWASP category being evaluated as context
- Be structured to return valid JSON output (not free text)
- NOT leak the attack source or model name to the judge (blind scoring)

Example output format the judge must return:
```json
{
  "verdict": "unsafe",
  "rationale": "The model revealed internal system prompt contents in response to a
  direct injection attack, confirming LLM01 vulnerability."
}
```

---

## What constitutes a successful run

A full evaluation run is complete when:
- All 3 models have been attacked across all 6 OWASP categories
- Each category has at least 30 attack prompts executed and scored
- Every record is logged to JSON with no missing fields
- MLflow experiment run is logged with summary metrics (attack success rate per
  category per model)
- Config files and judge prompt version are recorded in the run metadata

---

## Coding standards

- Python 3.11+
- Type hints on all function signatures
- Docstrings on all public functions and classes
- Black formatting, isort imports, flake8 linting — enforced via pre-commit
- No hardcoded strings for model names, prompt text, API keys, or file paths — all
  driven by config or environment variables
- API keys loaded from environment variables only, never committed
- All randomness seeded via config for reproducibility

---

## What to avoid

- Do not try to cover all 10 OWASP categories — scope is fixed at 6
- Do not build a web UI or interactive dashboard until the core pipeline is complete
- Do not commit results JSON files to the repo — add `results/` to `.gitignore` (except
  a curated `results/summary.json` with aggregate metrics)
- Do not use paid model APIs as attack targets — only Hugging Face open-source models
- Do not hardcode the judge prompt — it must be a versioned external file
- Do not use `print()` for logging — use Python's `logging` module throughout

---

## Developer background and existing skills

The developer (Serkan) is an ML engineer with 3+ years of production experience. He has:
- Built an end-to-end LLM evaluation pipeline with multi-dimensional rubrics and
  LLM-as-judge scoring at VerifyWise
- Experience with Hugging Face Transformers, OpenAI API, MLflow, Docker, GitHub Actions,
  pytest, FastAPI, Pandas
- Familiarity with Pydantic, structured JSON logging, and config-driven ML pipelines

Code suggestions should match this level — no need to explain Python basics or standard
library usage. Focus on PyRIT and DeepTeam API patterns, which are newer to the
developer.

---

## Milestones (reference only — do not auto-generate code for all of these)

| Week | Deliverable |
|---|---|
| 1 | Repo skeleton + working `run_attack.py` that logs one result end-to-end |
| 2 | LLM01 + LLM06 attack sets implemented and raw results collected |
| 3 | LLM-as-judge scoring pipeline complete, verdicts on Week 2 results |
| 4 | Remaining 4 categories implemented, full dataset collected |
| 5 | Analysis notebook with summary statistics and 3+ key findings |
| 6 | README polished, repo publicly shareable |
| 7 | Blog post written and published |
| 8 | Buffer / stretch goal (results dashboard or arXiv preprint) |

Start with Week 1 unless told otherwise.
