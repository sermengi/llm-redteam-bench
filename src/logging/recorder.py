"""Pydantic EvalRecord schema and Recorder for JSONL logging + MLflow metrics."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import mlflow
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = ["safe", "borderline", "unsafe"]


class RawRecord(BaseModel):
    """Schema for a raw inference result — no judge fields."""

    run_id: str
    timestamp: datetime
    model_name: str
    owasp_category: Literal["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"]
    attack_source: Literal["pyrit", "deepteam", "manual"]
    attack_strategy: Literal["direct_injection", "multi_turn_crescendo", "indirect_injection"]
    attacker_prompt: str
    system_prompt: str
    model_response: str
    config_hash: str


class EvalRecord(RawRecord):
    """Schema for a fully scored evaluation result. Extends RawRecord with judge fields."""

    judge_verdict: Literal["safe", "borderline", "unsafe"]
    judge_rationale: str
    judge_model: str
    judge_prompt_version: str
    rule_based_flags: list[str]


class Recorder:
    """Appends validated records to JSONL files and logs metrics to MLflow.

    Raw records go to results/raw/<run_id>.jsonl.
    Scored records go to results/<run_id>.jsonl.
    """

    def __init__(self, results_dir: Path, verdict_threshold: str) -> None:
        """Initialise the recorder.

        Args:
            results_dir: Directory where JSONL result files are written.
            verdict_threshold: Minimum verdict severity counted as attack success.

        Raises:
            ValueError: If verdict_threshold is not one of the valid severity values.
        """
        if verdict_threshold not in _SEVERITY_ORDER:
            raise ValueError(
                f"verdict_threshold must be one of {_SEVERITY_ORDER!r}, got {verdict_threshold!r}"
            )
        self.results_dir = results_dir
        self.verdict_threshold = verdict_threshold
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "raw").mkdir(parents=True, exist_ok=True)

    def log_raw(self, record: RawRecord) -> None:
        """Write a raw inference record to results/raw/<run_id>.jsonl.

        Args:
            record: A RawRecord instance (no judge fields).
        """
        raw_dir = self.results_dir / "raw"
        out_path = raw_dir / f"{record.run_id}.jsonl"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        logger.info(
            "Logged raw record run_id=%s model=%s category=%s",
            record.run_id,
            record.model_name,
            record.owasp_category,
        )

    def log(self, record: EvalRecord) -> None:
        """Validate, write, and log a single evaluation record.

        Args:
            record: A fully populated EvalRecord instance.
        """
        out_path = self.results_dir / f"{record.run_id}.jsonl"
        with open(out_path, "a", encoding="utf-8") as f:
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
        if mlflow.active_run():
            mlflow.log_metric("is_unsafe", is_unsafe)
            mlflow.log_metric("rule_flags_count", len(record.rule_based_flags))
