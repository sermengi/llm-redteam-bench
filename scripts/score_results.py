#!/usr/bin/env python3
"""Phase 2 CLI: score raw inference JSONL with LLM-as-judge and rule-based classifier.

Usage:
    python scripts/score_results.py --input results/raw/20260416-143022-c740ff.jsonl
    python scripts/score_results.py --input results/raw/20260416-143022-c740ff.jsonl --force
"""

import argparse
import json
import logging
import time
from pathlib import Path

import mlflow

from src.config import load_judge_config
from src.logging.recorder import EvalRecord, RawRecord
from src.scoring.judge import JudgeScorer
from src.scoring.rule_based import RuleBasedClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path("configs")
_PROMPTS_DIR = Path("prompts")
_RESULTS_DIR = Path("results")


def _load_scored_keys(output_path: Path) -> set[tuple[str, str, str]]:
    """Load (run_id, model_name, attacker_prompt) keys from an existing scored JSONL.

    Args:
        output_path: Path to the scored output JSONL file.

    Returns:
        Set of (run_id, model_name, attacker_prompt) tuples already scored.
    """
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            keys.add((r["run_id"], r["model_name"], r["attacker_prompt"]))
        except (json.JSONDecodeError, KeyError):
            pass
    return keys


def _score_with_retry(
    judge: JudgeScorer,
    record: RawRecord,
    max_retries: int = 3,
) -> dict[str, str]:
    """Call judge.score() with exponential backoff on rate limit errors.

    Args:
        judge: Configured JudgeScorer instance.
        record: The RawRecord to score.
        max_retries: Maximum number of attempts before giving up.

    Returns:
        Dict with 'verdict' and 'rationale' keys.

    Raises:
        RuntimeError: If all retries are exhausted on rate limit errors.
        Exception: Any non-rate-limit exception is re-raised immediately.
    """
    for attempt in range(max_retries):
        try:
            return judge.score(
                attacker_prompt=record.attacker_prompt,
                model_response=record.model_response,
                owasp_category=record.owasp_category,
            )
        except Exception as exc:
            if "rate_limit" not in str(exc).lower():
                raise
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning(
                    "Rate limit hit (attempt %d/%d), retrying in %ds ...",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
    raise RuntimeError(f"Exhausted retries ({max_retries}) due to rate limiting.")


def score_file(
    input_path: Path,
    output_path: Path,
    judge: JudgeScorer,
    classifier: RuleBasedClassifier,
    force: bool = False,
) -> tuple[int, int, int]:
    """Score a raw JSONL file with LLM-as-judge and rule-based classifier.

    Args:
        input_path: Path to raw inference JSONL (results/raw/<run_id>.jsonl).
        output_path: Path to write scored EvalRecord JSONL.
        judge: Configured JudgeScorer instance.
        classifier: RuleBasedClassifier instance.
        force: If True, re-score records already present in output_path.

    Returns:
        Tuple of (scored, skipped, failed) counts.
    """
    failed_dir = output_path.parent / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    failed_path = failed_dir / f"{output_path.stem}_failed.jsonl"

    scored_keys = set() if force else _load_scored_keys(output_path)
    logger.info(
        "Loaded %d already-scored keys from %s (force=%s)",
        len(scored_keys),
        output_path,
        force,
    )

    raw_lines = [ln for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    scored = skipped = failed = 0

    for line in raw_lines:
        try:
            raw = RawRecord(**json.loads(line))
        except Exception as exc:
            logger.warning("Skipping corrupt JSONL line: %s", exc)
            continue

        key = (raw.run_id, raw.model_name, raw.attacker_prompt)
        if key in scored_keys:
            skipped += 1
            continue

        try:
            judge_result = _score_with_retry(judge, raw)
            flags = classifier.classify(raw.model_response)
            record = EvalRecord(
                **raw.model_dump(),
                judge_verdict=judge_result["verdict"],
                judge_rationale=judge_result["rationale"],
                judge_model=judge.client.model,
                judge_prompt_version=judge.prompt_version,
                rule_based_flags=flags,
            )
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            scored += 1
            logger.info(
                "Scored record %d: verdict=%s model=%s",
                scored,
                record.judge_verdict,
                record.model_name,
            )
        except Exception as exc:
            logger.error("Scoring failed for record: %s", exc)
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            failed += 1

    return scored, skipped, failed


def main() -> None:
    """Parse CLI arguments and run the scoring phase."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Score raw inference JSONL with LLM-as-judge."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw inference JSONL (results/raw/<run_id>.jsonl).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score all records, even if already scored.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    run_id = input_path.stem
    output_path = _RESULTS_DIR / f"{run_id}.jsonl"

    judge_config = load_judge_config(_CONFIGS_DIR / "judge.yaml")
    judge = JudgeScorer.from_config(judge_config, _PROMPTS_DIR)
    classifier = RuleBasedClassifier()

    with mlflow.start_run(run_name=f"score-{run_id}"):
        mlflow.log_param("input_file", str(input_path))
        mlflow.log_param("force_rescore", args.force)
        mlflow.log_param("judge_model", judge_config.model)

        scored, skipped, failed = score_file(
            input_path=input_path,
            output_path=output_path,
            judge=judge,
            classifier=classifier,
            force=args.force,
        )

        if output_path.exists() and scored > 0:
            all_lines = [
                ln for ln in output_path.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            total_scored = len(all_lines)
            unsafe_count = sum(
                1
                for ln in all_lines
                if json.loads(ln).get("judge_verdict") in ("unsafe", "borderline")
            )
            mlflow.log_metric("attack_success_rate", unsafe_count / total_scored)

    summary = f"scored: {scored}, skipped: {skipped}, failed: {failed}"
    logger.info(summary)
    print(summary)
    print(f"Output: {output_path}")
    failed_path = _RESULTS_DIR / "failed" / f"{run_id}_failed.jsonl"
    if failed:
        print(f"Failed records: {failed_path}")


if __name__ == "__main__":
    main()
