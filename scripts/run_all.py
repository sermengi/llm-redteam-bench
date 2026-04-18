#!/usr/bin/env python3
"""Full-run orchestrator: all models × all categories in one command.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --categories LLM01 LLM07
    python scripts/run_all.py --models meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_all.py --skip-scoring
"""

import argparse
import hashlib
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from scripts.score_results import score_file
from src import pipeline
from src.attacks.loader import load_attacks
from src.config import (
    load_attacks_config,
    load_judge_config,
    load_models_config,
    load_system_prompts_config,
)
from src.inference.modal_client import ModalHFClient
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


def _build_run_id(config_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{config_hash[:6]}"


def main() -> None:
    """Parse CLI arguments and run the full evaluation."""
    parser = argparse.ArgumentParser(
        description="Full evaluation: all models × all categories, one run_id."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Subset of OWASP categories to run (default: all from attacks.yaml).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names to run (default: all from models.yaml).",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Run inference only; skip LLM-as-judge scoring pass.",
    )
    args = parser.parse_args()

    attacks_config = load_attacks_config(_CONFIGS_DIR / "attacks.yaml")
    models_config = load_models_config(_CONFIGS_DIR / "models.yaml")
    judge_config = load_judge_config(_CONFIGS_DIR / "judge.yaml")
    system_prompts = load_system_prompts_config(_CONFIGS_DIR / "system_prompts.yaml")

    random.seed(attacks_config.seed)
    config_hash = hashlib.sha256((_CONFIGS_DIR / "attacks.yaml").read_bytes()).hexdigest()
    run_id = _build_run_id(config_hash)

    categories = args.categories or attacks_config.categories
    model_names = args.models or [m.name for m in models_config.models]

    logger.info(
        "Starting full run run_id=%s models=%d categories=%d",
        run_id,
        len(model_names),
        len(categories),
    )

    judge = JudgeScorer.from_config(judge_config, _PROMPTS_DIR)
    classifier = RuleBasedClassifier()
    recorder = Recorder(results_dir=_RESULTS_DIR, verdict_threshold=judge_config.verdict_threshold)

    summary: list[dict[str, object]] = []

    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("models", ",".join(model_names))
        mlflow.log_param("categories", ",".join(categories))
        mlflow.log_param("config_hash", config_hash)

        for model_name in model_names:
            with ModalHFClient(model_name=model_name) as client:
                for category in categories:
                    attacks = load_attacks(category, attacks_config)
                    logged = 0
                    try:
                        raw_records = pipeline.run_inference_batch(
                            attacks=attacks,
                            model_name=model_name,
                            client=client,
                            owasp_category=category,
                            run_id=run_id,
                            config_hash=config_hash,
                            recorder=recorder,
                            system_prompts=system_prompts,
                        )
                        logged = len(raw_records)
                    except Exception as exc:
                        logger.error(
                            "Inference failed model=%s category=%s error=%s",
                            model_name,
                            category,
                            exc,
                        )

                    summary.append(
                        {
                            "model": model_name,
                            "category": category,
                            "logged": logged,
                            "scored": 0,
                        }
                    )
                    logger.info(
                        "Inference done model=%s category=%s logged=%d",
                        model_name,
                        category,
                        logged,
                    )

        if not args.skip_scoring:
            raw_path = _RESULTS_DIR / "raw" / f"{run_id}.jsonl"
            output_path = _RESULTS_DIR / f"{run_id}.jsonl"
            if raw_path.exists():
                logger.info("Starting scoring pass for run_id=%s", run_id)
                scored, skipped, failed = score_file(
                    input_path=raw_path,
                    output_path=output_path,
                    judge=judge,
                    classifier=classifier,
                )
                logger.info(
                    "Scoring complete scored=%d skipped=%d failed=%d",
                    scored,
                    skipped,
                    failed,
                )
                mlflow.log_metric("total_scored", scored)
                mlflow.log_metric("total_failed", failed)
                per_combo = scored // max(len(summary), 1)
                for row in summary:
                    row["scored"] = per_combo
            else:
                logger.warning("No raw output file found at %s — skipping scoring.", raw_path)

    col_model = 45
    col_cat = 8
    col_num = 8
    print(
        f"\n{'Model':<{col_model}} {'Category':<{col_cat}} "
        f"{'Logged':>{col_num}} {'Scored':>{col_num}}"
    )
    print("-" * (col_model + col_cat + col_num * 2 + 3))
    for row in summary:
        print(
            f"{row['model']:<{col_model}} {row['category']:<{col_cat}} "
            f"{row['logged']:>{col_num}} {row['scored']:>{col_num}}"
        )
    print(f"\nrun_id: {run_id}")
    print(f"Results: {_RESULTS_DIR / (run_id + '.jsonl')}")


if __name__ == "__main__":
    main()
