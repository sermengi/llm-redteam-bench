#!/usr/bin/env python3
"""CLI entry point for running adversarial evaluations.

Usage:
    python scripts/run_eval.py --category LLM01
    python scripts/run_eval.py --category LLM06 --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_eval.py --category LLM01 --dry-run
"""

import argparse
import hashlib
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from src import pipeline
from src.attacks.loader import load_attacks
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

_VALID_CATEGORIES = ["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"]


def _build_run_id(config_hash: str) -> str:
    """Generate a unique run ID: timestamp + first 6 chars of attacks.yaml SHA-256."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{config_hash[:6]}"


def main() -> None:
    """Parse CLI arguments, load configs, and run the evaluation batch."""
    parser = argparse.ArgumentParser(
        description="Run adversarial evaluation against target models."
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=_VALID_CATEGORIES,
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
        help="Load and print attack prompts. No inference or logging.",
    )
    args = parser.parse_args()

    models_config = load_models_config(_CONFIGS_DIR / "models.yaml")
    attacks_config = load_attacks_config(_CONFIGS_DIR / "attacks.yaml")
    judge_config = load_judge_config(_CONFIGS_DIR / "judge.yaml")

    random.seed(attacks_config.seed)

    if args.category not in attacks_config.categories:
        logger.error("Category %s not listed in configs/attacks.yaml", args.category)
        sys.exit(1)

    config_hash = hashlib.sha256((_CONFIGS_DIR / "attacks.yaml").read_bytes()).hexdigest()
    run_id = _build_run_id(config_hash)
    attacks = load_attacks(args.category, attacks_config)

    if args.dry_run:
        logger.info(
            "Dry run — run_id=%s category=%s attacks_loaded=%d",
            run_id,
            args.category,
            len(attacks),
        )
        for i, attack in enumerate(attacks[:3]):
            logger.info(
                "Attack[%d] source=%s strategy=%s prompt=%s",
                i,
                attack.attack_source,
                attack.attack_strategy,
                attack.prompt[:80],
            )
        return

    target_models = [m for m in models_config.models if args.model is None or m.name == args.model]
    if not target_models:
        logger.error("No models matched --model=%s in configs/models.yaml", args.model)
        sys.exit(1)

    judge = JudgeScorer.from_config(judge_config, _PROMPTS_DIR)
    classifier = RuleBasedClassifier()
    recorder = Recorder(results_dir=_RESULTS_DIR, verdict_threshold=judge_config.verdict_threshold)

    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("category", args.category)
        mlflow.log_param("judge_model", judge_config.model)
        mlflow.log_param("config_hash", config_hash)
        mlflow.log_param("attack_count", len(attacks))

        records: list = []
        for model in target_models:
            batch = pipeline.run_batch(
                attacks=attacks,
                model_names=[model.name],
                owasp_category=args.category,
                run_id=run_id,
                config_hash=config_hash,
                mock=model.mock,
                judge=judge,
                classifier=classifier,
                recorder=recorder,
            )
            records.extend(batch)

    logger.info(
        "Run complete. %d records logged to results/%s.jsonl", len(records), run_id
    )


if __name__ == "__main__":
    main()
