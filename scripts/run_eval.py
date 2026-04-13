#!/usr/bin/env python3
"""CLI entry point for running adversarial evaluations.

Usage:
    python scripts/run_eval.py --category LLM01
    python scripts/run_eval.py --category LLM01 --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_eval.py --category LLM01 --dry-run
"""

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from src import pipeline
from src.config import load_attacks_config, load_judge_config, load_models_config
from src.logging.recorder import Recorder
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
_MANUAL_PROMPTS_DIR = Path("src/attacks/manual")

_CATEGORY_FILE_MAP: dict[str, str] = {
    "LLM01": "llm01.txt",
    "LLM02": "llm02.txt",
    "LLM04": "llm04.txt",
    "LLM06": "llm06.txt",
    "LLM07": "llm07.txt",
    "LLM09": "llm09.txt",
}


def _build_run_id() -> str:
    """Generate a unique run ID: timestamp + first 6 chars of attacks.yaml SHA-256."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    attacks_hash = hashlib.sha256(
        (_CONFIGS_DIR / "attacks.yaml").read_bytes()
    ).hexdigest()[:6]
    return f"{timestamp}-{attacks_hash}"


def _load_prompts(category: str) -> list[str]:
    """Load all non-empty lines from the manual prompt file for the given category.

    Args:
        category: OWASP category string (e.g. 'LLM01').

    Returns:
        List of prompt strings.
    """
    filename = _CATEGORY_FILE_MAP[category]
    path = _MANUAL_PROMPTS_DIR / filename
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    """Parse CLI arguments, load configs, and run the evaluation loop."""
    parser = argparse.ArgumentParser(
        description="Run adversarial evaluation against target models."
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=list(_CATEGORY_FILE_MAP.keys()),
        help="OWASP vulnerability category to evaluate.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run against a specific model only (must match name in configs/models.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the first prompt. No inference or logging.",
    )
    args = parser.parse_args()

    models_config = load_models_config(_CONFIGS_DIR / "models.yaml")
    attacks_config = load_attacks_config(_CONFIGS_DIR / "attacks.yaml")
    judge_config = load_judge_config(_CONFIGS_DIR / "judge.yaml")

    if args.category not in attacks_config.categories:
        logger.error("Category %s not listed in configs/attacks.yaml", args.category)
        sys.exit(1)

    config_hash = hashlib.sha256(
        (_CONFIGS_DIR / "attacks.yaml").read_bytes()
    ).hexdigest()
    run_id = _build_run_id()
    prompts = _load_prompts(args.category)

    if args.dry_run:
        logger.info(
            "Dry run — run_id=%s category=%s prompts_loaded=%d",
            run_id,
            args.category,
            len(prompts),
        )
        logger.info("First prompt: %s", prompts[0])
        return

    target_models = [
        m for m in models_config.models
        if args.model is None or m.name == args.model
    ]
    if not target_models:
        logger.error("No models matched --model=%s in configs/models.yaml", args.model)
        sys.exit(1)

    judge = JudgeScorer.from_config(judge_config, _PROMPTS_DIR)
    classifier = RuleBasedClassifier()
    recorder = Recorder(
        results_dir=_RESULTS_DIR,
        verdict_threshold=judge_config.verdict_threshold,
    )

    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("category", args.category)
        mlflow.log_param("judge_model", judge_config.model)
        mlflow.log_param("config_hash", config_hash)

        for model in target_models:
            logger.info(
                "Running %d prompts against %s (mock=%s)",
                len(prompts),
                model.name,
                model.mock,
            )
            for prompt in prompts:
                pipeline.run(
                    prompt=prompt,
                    model_name=model.name,
                    mock=model.mock,
                    owasp_category=args.category,
                    run_id=run_id,
                    config_hash=config_hash,
                    judge=judge,
                    classifier=classifier,
                    recorder=recorder,
                )

    logger.info("Run complete. Results in results/%s.jsonl", run_id)


if __name__ == "__main__":
    main()
