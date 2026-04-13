"""Pydantic EvalRecord schema and Recorder for JSONL logging + MLflow metrics."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import mlflow
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = ["safe", "borderline", "unsafe"]


class EvalRecord(BaseModel):
    """Schema for a single adversarial evaluation result. Every field is required."""

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


class Recorder:
    """Appends validated EvalRecord instances to a JSONL file and logs metrics to MLflow.

    Each run_id maps to one JSONL file: results/<run_id>.jsonl.
    Records are appended one per line. A malformed record is logged and skipped — never written.
    """

    def __init__(self, results_dir: Path, verdict_threshold: str) -> None:
        """Initialise the recorder.

        Args:
            results_dir: Directory where JSONL result files are written.
            verdict_threshold: Minimum verdict severity counted as attack success.
                'borderline' means borderline + unsafe = is_unsafe=1.
                'unsafe' means only unsafe = is_unsafe=1.
        """
        self.results_dir = results_dir
        self.verdict_threshold = verdict_threshold
        results_dir.mkdir(parents=True, exist_ok=True)

    def log(self, record: EvalRecord) -> None:
        """Validate, write, and log a single evaluation record.

        Args:
            record: A fully populated EvalRecord instance.
        """
        out_path = self.results_dir / f"{record.run_id}.jsonl"
        with open(out_path, "a") as f:
            f.write(record.model_dump_json() + "\n")

        logger.info(
            "Logged record run_id=%s model=%s category=%s verdict=%s",
            record.run_id,
            record.model_name,
            record.owasp_category,
            record.judge_verdict,
        )

        is_unsafe = int(
            _SEVERITY_ORDER.index(record.judge_verdict)
            >= _SEVERITY_ORDER.index(self.verdict_threshold)
        )
        mlflow.log_metric("is_unsafe", is_unsafe)
        mlflow.log_metric("rule_flags_count", len(record.rule_based_flags))
