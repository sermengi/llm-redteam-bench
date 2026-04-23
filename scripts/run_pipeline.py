#!/usr/bin/env python3
"""Phase 2: Run inference over cached prompts for all models × categories.

Requires prompts to be cached first via generate_prompts.py.
Errors loudly if any requested category has no valid cache.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --categories LLM01 LLM07
    python scripts/run_pipeline.py --models meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_pipeline.py --skip-scoring
"""

import argparse
import logging
import random
import sys

import mlflow

from scripts.score_results import score_file
from src import pipeline
from src.attacks.loader import load_cached_prompts
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
from src.scripts_utils import (
    _CONFIGS_DIR,
    _PROMPTS_CACHE_DIR,
    _PROMPTS_DIR,
    _RESULTS_DIR,
    _VALID_CATEGORIES,
    build_run_id,
    compute_config_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run inference over cached prompts, then optionally score the results."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Inference from cache across all models × categories."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=_VALID_CATEGORIES,
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
        help="Run inference only; skip the LLM-as-judge scoring pass.",
    )
    args = parser.parse_args()

    attacks_config = load_attacks_config(_CONFIGS_DIR / "attacks.yaml")
    models_config = load_models_config(_CONFIGS_DIR / "models.yaml")
    judge_config = load_judge_config(_CONFIGS_DIR / "judge.yaml")
    system_prompts = load_system_prompts_config(_CONFIGS_DIR / "system_prompts.yaml")

    random.seed(attacks_config.seed)
    config_hash = compute_config_hash(attacks_config)
    run_id = build_run_id(config_hash)

    categories = args.categories or list(attacks_config.categories)
    model_names = args.models or [m.name for m in models_config.models]

    # Validate cache exists for all requested categories before touching any model.
    missing = [
        c for c in categories if load_cached_prompts(c, config_hash, _PROMPTS_CACHE_DIR) is None
    ]
    if missing:
        logger.error(
            "No valid prompt cache for: %s. Run generate_prompts.py first.",
            missing,
        )
        sys.exit(1)

    logger.info(
        "Starting inference run_id=%s models=%d categories=%d",
        run_id,
        len(model_names),
        len(categories),
    )

    recorder = Recorder(
        results_dir=_RESULTS_DIR,
        verdict_threshold=judge_config.verdict_threshold,
    )
    summary: list[dict[str, object]] = []

    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("models", ",".join(model_names))
        mlflow.log_param("categories", ",".join(categories))
        mlflow.log_param("config_hash", config_hash)

        for model_name in model_names:
            with ModalHFClient(model_name=model_name) as client:
                for category in categories:
                    attacks = load_cached_prompts(category, config_hash, _PROMPTS_CACHE_DIR)
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
                        {"model": model_name, "category": category, "logged": logged, "scored": 0}
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
                judge = JudgeScorer.from_config(judge_config, _PROMPTS_DIR)
                classifier = RuleBasedClassifier()
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
                logger.warning("No raw output found at %s — skipping scoring.", raw_path)

    col_model, col_cat, col_num = 45, 8, 8
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
