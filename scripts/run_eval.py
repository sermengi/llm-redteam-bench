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
from src.attacks.loader import load_attacks, resolve_attacks
from src.config import (
    load_attacks_config,
    load_judge_config,
    load_models_config,
    load_system_prompts_config,
)
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
_PROMPTS_CACHE_DIR = _RESULTS_DIR / "prompts"

_VALID_CATEGORIES = ["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"]


def _compute_config_hash(attacks_config) -> str:
    """Hash attacks.yaml and all deepteam custom_prompt_file paths.

    Including prompt template files ensures the cache is invalidated when
    a DeepTeam generator template changes, not just the YAML config.
    """
    hasher = hashlib.sha256()
    hasher.update((_CONFIGS_DIR / "attacks.yaml").read_bytes())
    for cat_cfg in attacks_config.deepteam.categories.values():
        prompt_file = Path(cat_cfg.custom_prompt_file)
        if prompt_file.exists():
            hasher.update(prompt_file.read_bytes())
    return hasher.hexdigest()


def _build_run_id(config_hash: str) -> str:
    """Generate a unique run ID: timestamp + first 6 chars of attacks.yaml SHA-256."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{config_hash[:6]}"


def _print_dry_run_summary(
    attacks: list,
    category: str,
    run_id: str,
    config_hash: str,
    cache_path: Path,
) -> None:
    """Print a formatted prompt summary table and cache info to stdout."""
    from collections import defaultdict

    by_source: dict[str, list] = defaultdict(list)
    for a in attacks:
        by_source[a.attack_source].append(a)

    print(f"=== Dry Run: {category}  run_id={run_id} ===")
    print()
    print(f"  {'source':<12}{'strategy':<26}{'count':>5}  sample")
    print(f"  {'─' * 12}{'─' * 26}{'─' * 5}  {'─' * 50}")
    for source in ("manual", "pyrit", "deepteam"):
        group = by_source.get(source, [])
        if not group:
            continue
        sample = group[0].prompt[:50].replace("\n", " ")
        strategy = group[0].attack_strategy
        print(f"  {source:<12}{strategy:<26}{len(group):>5}  {sample}...")
    print()
    print(f"  Total: {len(attacks)} prompts  |  Config hash: {config_hash[:8]}  |  Saved: {cache_path}")
    print()


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
    system_prompts = load_system_prompts_config(_CONFIGS_DIR / "system_prompts.yaml")

    random.seed(attacks_config.seed)

    if args.category not in attacks_config.categories:
        logger.error("Category %s not listed in configs/attacks.yaml", args.category)
        sys.exit(1)

    config_hash = _compute_config_hash(attacks_config)
    run_id = _build_run_id(config_hash)
    attacks = resolve_attacks(args.category, attacks_config, config_hash, _PROMPTS_CACHE_DIR)

    if args.dry_run:
        _print_dry_run_summary(
            attacks=attacks,
            category=args.category,
            run_id=run_id,
            config_hash=config_hash,
            cache_path=_PROMPTS_CACHE_DIR / f"{args.category}.json",
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

        records = pipeline.run_mock_batch(
            attacks=attacks,
            model_names=[m.name for m in target_models],
            owasp_category=args.category,
            run_id=run_id,
            config_hash=config_hash,
            judge=judge,
            classifier=classifier,
            recorder=recorder,
            system_prompts=system_prompts,
        )

    logger.info("Run complete. %d records logged to results/%s.jsonl", len(records), run_id)


if __name__ == "__main__":
    main()
