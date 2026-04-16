# Week 3 Design: Two-Phase Scoring Pipeline with Modal Inference

**Date:** 2026-04-16
**Scope:** LLM-as-judge scoring pipeline for `llm-redteam-bench` Week 3 deliverable
**Status:** Approved

---

## Context

Weeks 1 and 2 produced a working pipeline with mock inference. The GPU (RTX 3050, 4GB VRAM) is now operational but insufficient to load 7–8B models (requires ~14–16GB fp16). The scoring infrastructure (`judge.py`, `rule_based.py`, `prompts/judge_v1.txt`) is already implemented. Week 3 delivers: real inference via Modal cloud T4 + a two-phase checkpoint pipeline that makes scoring resumable and independent of inference.

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Inference target | Modal cloud T4 (16GB VRAM) | Local GPU (4GB) cannot fit 7B models; Modal is programmatic and already referenced in CLAUDE.md |
| Model loading | Load-once per model | Avoid reloading multi-GB weights per record |
| Run strategy | One model at a time via `--model` flag | Resumable, debuggable, minimal VRAM pressure |
| Pipeline shape | Two-phase checkpoint | Inference is expensive; scoring is cheap and re-runnable; crash in judge must not destroy inference results |
| Scoring resume | Skip-already-scored by default | Idempotent re-runs; `--force` flag to override |
| Entry points | Two dedicated CLI scripts | Single responsibility; each phase independently runnable |
| Renamed function | `run_batch` → `run_mock_batch` | Makes mock-only path unambiguous at call sites |

---

## Architecture

```
Phase 1 — Inference                    Phase 2 — Scoring
─────────────────────────────          ──────────────────────────────────
scripts/run_inference.py               scripts/score_results.py
  │                                      │
  ├─ loads attacks (loader.py)           ├─ reads results/raw/<run_id>.jsonl
  ├─ instantiates ModalHFClient          ├─ for each record:
  │    (load-once per model)             │    ├─ skip if judge_verdict exists
  ├─ runs all attacks via Modal          │    │    (unless --force)
  ├─ writes RawRecord per response       │    ├─ JudgeScorer.score()
  │    to results/raw/<run_id>.jsonl     │    ├─ RuleBasedClassifier.classify()
  └─ no scoring, no judge calls          │    └─ appends EvalRecord to
                                         │         results/<run_id>.jsonl
                                         └─ logs MLflow summary metrics
```

**Typical usage:**
```bash
# Phase 1 — run once per model per category
python scripts/run_inference.py --category LLM01 --model mistralai/Mistral-7B-Instruct-v0.3
# → results/raw/20260416-143022-c740ff.jsonl

# Phase 2 — score (resumable, re-runnable)
python scripts/score_results.py --input results/raw/20260416-143022-c740ff.jsonl
# → results/20260416-143022-c740ff.jsonl

# Force re-score (e.g. after updating judge prompt)
python scripts/score_results.py --input results/raw/20260416-143022-c740ff.jsonl --force
```

---

## Components

### New: `src/inference/modal_client.py`
- `ModalHFClient` implements the existing `ModelClient` interface
- Modal `App` with a GPU-decorated class: loads HF model + tokenizer in `__enter__`, exposes `generate(prompt, system_prompt) -> str`
- Config: `gpu="T4"`, `torch_dtype=float16`, `device_map="cuda"`
- Model name passed from `configs/models.yaml`

### Modified: `src/inference/hf_client.py`
- Move model + tokenizer loading from `generate()` into `__init__()` (load-once)
- Remove `mock` boolean — mock path handled at pipeline level, not in the client

### Modified: `src/logging/recorder.py`
- Add `RawRecord` Pydantic model: all `EvalRecord` fields except `judge_verdict`, `judge_rationale`, `judge_model`, `judge_prompt_version`, `rule_based_flags`
- Add `Recorder.log_raw()` → writes to `results/raw/<run_id>.jsonl` (incremental, flushed per record)
- Existing `Recorder.log()` for `EvalRecord` unchanged

### Modified: `src/pipeline.py`
- Rename `run_batch()` → `run_mock_batch()` (mock/dev path, unchanged logic)
- Add `run_inference_batch()`: attack → `ModalHFClient.generate()` → `RawRecord` → `log_raw()`

### New: `scripts/run_inference.py`
- Args: `--category` (required), `--model` (required)
- Auto-generates `run_id` (timestamp + config hash, same logic as `run_eval.py`)
- Instantiates `ModalHFClient`, calls `pipeline.run_inference_batch()`
- Prints output path on completion

### New: `scripts/score_results.py`
- Args: `--input` (path to raw JSONL, required), `--force` (re-score all)
- Resume logic: checks existing output file for already-scored records by `(run_id, model_name, attacker_prompt)` key
- Failed records written to `results/failed/<run_id>_failed.jsonl`
- Prints summary: `scored: N, skipped: N, failed: N`
- Logs MLflow metrics: attack success rate per verdict

---

## Data Schemas

### `RawRecord`
```json
{
  "run_id": "20260416-143022-c740ff",
  "timestamp": "2026-04-16T14:30:22Z",
  "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
  "owasp_category": "LLM01",
  "attack_source": "manual",
  "attack_strategy": "direct_injection",
  "attacker_prompt": "...",
  "system_prompt": "...",
  "model_response": "...",
  "config_hash": "c740ff..."
}
```

### `EvalRecord` (existing schema, unchanged)
`RawRecord` fields plus:
```json
{
  "judge_verdict": "unsafe",
  "judge_rationale": "One sentence rationale.",
  "judge_model": "gpt-4o-mini",
  "judge_prompt_version": "v1",
  "rule_based_flags": ["pii_detected"]
}
```

---

## Error Handling

### Phase 1 (`run_inference.py`)
- Per-record inference failures: caught, logged, record skipped — batch continues
- Modal failures (OOM, cold start timeout): logged with full traceback, record skipped
- Raw JSONL written incrementally and flushed per record — crash loses only the in-flight record

### Phase 2 (`score_results.py`)
- Per-record judge failures (bad JSON, invalid verdict): caught, logged, written to `results/failed/`
- OpenAI rate limits: exponential backoff, 3 retries before skipping record
- End-of-run summary printed regardless of failures

### Not handled
- Modal auth failures: abort immediately with clear error
- Corrupt JSONL input lines: `json.JSONDecodeError` logged, line skipped

---

## Testing

### Updated tests
- `tests/test_pipeline.py` — rename `run_batch` → `run_mock_batch` references; add test for `run_inference_batch()` with stubbed `ModelClient`

### New tests
- `tests/test_modal_client.py` — unit test `ModalHFClient` with mocked Modal app; verifies `generate()` signature and return type
- `tests/test_scorer.py` — tests for `score_results.py`: skip-already-scored behaviour, `--force` override, failed record handling, resume after partial scoring

### Not unit-tested
- Live Modal GPU execution — validated manually on first real run
- Live OpenAI judge calls — already covered by existing mocked tests
