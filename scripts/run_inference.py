#!/usr/bin/env python3
"""Phase 1 CLI: run attack prompts through Modal inference, write raw JSONL.

Usage:
    python scripts/run_inference.py --category LLM01 --model mistralai/Mistral-7B-Instruct-v0.3
    python scripts/run_inference.py --category LLM06 --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import hashlib
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import pipeline
from src.attacks.loader import load_attacks
from src.config import load_attacks_config, load_models_config
from src.inference.modal_client import ModalHFClient
from src.logging.recorder import Recorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path("configs")
_RESULTS_DIR = Path("results")
_VALID_CATEGORIES = ["LLM01", "LLM02", "LLM04", "LLM06", "LLM07", "LLM09"]


def _build_run_id(config_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{config_hash[:6]}"


def main() -> None:
    """Parse CLI arguments and run the inference batch."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Run adversarial attacks via Modal and write raw responses."
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=_VALID_CATEGORIES,
        help="OWASP vulnerability category to evaluate.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model identifier (must match a name in configs/models.yaml).",
    )
    args = parser.parse_args()

    attacks_config = load_attacks_config(_CONFIGS_DIR / "attacks.yaml")
    models_config = load_models_config(_CONFIGS_DIR / "models.yaml")

    valid_model_names = [m.name for m in models_config.models]
    if args.model not in valid_model_names:
        logger.error(
            "Model '%s' not found in configs/models.yaml. Valid models: %s",
            args.model,
            valid_model_names,
        )
        sys.exit(1)

    if args.category not in attacks_config.categories:
        logger.error("Category %s not listed in configs/attacks.yaml", args.category)
        sys.exit(1)

    random.seed(attacks_config.seed)
    config_hash = hashlib.sha256((_CONFIGS_DIR / "attacks.yaml").read_bytes()).hexdigest()
    run_id = _build_run_id(config_hash)

    logger.info(
        "Starting inference run_id=%s category=%s model=%s",
        run_id,
        args.category,
        args.model,
    )

    attacks = load_attacks(args.category, attacks_config)
    client = ModalHFClient(model_name=args.model)
    recorder = Recorder(results_dir=_RESULTS_DIR, verdict_threshold="borderline")

    records = pipeline.run_inference_batch(
        attacks=attacks,
        model_name=args.model,
        client=client,
        owasp_category=args.category,
        run_id=run_id,
        config_hash=config_hash,
        recorder=recorder,
    )

    raw_path = _RESULTS_DIR / "raw" / f"{run_id}.jsonl"
    logger.info(
        "Inference complete. %d/%d records written to %s",
        len(records),
        len(attacks),
        raw_path,
    )
    print(raw_path)


if __name__ == "__main__":
    main()
