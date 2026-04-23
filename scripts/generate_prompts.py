#!/usr/bin/env python3
"""Phase 1: Generate and cache attack prompts per OWASP category.

This is the only script that calls load_attacks(). All downstream scripts
(run_pipeline.py) read exclusively from the prompt cache.

Usage:
    python scripts/generate_prompts.py                           # all categories
    python scripts/generate_prompts.py --categories LLM01 LLM06
    python scripts/generate_prompts.py --categories LLM01 --dry-run
"""

import argparse
import logging
import random
import sys
from collections import defaultdict

from src.attacks.loader import resolve_attacks
from src.config import load_attacks_config
from src.scripts_utils import (
    _CONFIGS_DIR,
    _PROMPTS_CACHE_DIR,
    _VALID_CATEGORIES,
    build_run_id,
    compute_config_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _print_dry_run_summary(attacks, category, run_id, config_hash, cache_path) -> None:
    """Print a formatted prompt summary table and cache path to stdout."""
    by_source: dict[str, list] = defaultdict(list)
    for a in attacks:
        by_source[a.attack_source].append(a)

    print(f"=== Dry Run: {category}  run_id={run_id} ===")
    print()
    print(f"  {'source':<12}{'sample_strategy':<26}{'count':>5}  sample")
    print(f"  {'─' * 12}{'─' * 26}{'─' * 5}  {'─' * 50}")
    for source in ("manual", "pyrit", "deepteam"):
        group = by_source.get(source, [])
        if not group:
            continue
        sample = group[0].prompt[:50].replace("\n", " ")
        strategy = group[0].attack_strategy
        print(f"  {source:<12}{strategy:<26}{len(group):>5}  {sample}...")
    print()
    print(
        f"  Total: {len(attacks)} prompts  |  "
        f"Config hash: {config_hash[:8]}  |  "
        f"Saved: {cache_path}"
    )
    print()


def main() -> None:
    """Generate and cache attack prompts for the requested categories."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate and cache attack prompts."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=_VALID_CATEGORIES,
        help="OWASP categories to generate (default: all from attacks.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt summary after generation. Cache is still written.",
    )
    args = parser.parse_args()

    attacks_config = load_attacks_config(_CONFIGS_DIR / "attacks.yaml")
    random.seed(attacks_config.seed)

    config_hash = compute_config_hash(attacks_config)
    run_id = build_run_id(config_hash)

    categories = args.categories or list(attacks_config.categories)
    invalid = [c for c in categories if c not in attacks_config.categories]
    if invalid:
        logger.error("Categories not listed in attacks.yaml: %s", invalid)
        sys.exit(1)

    for category in categories:
        attacks = resolve_attacks(category, attacks_config, config_hash, _PROMPTS_CACHE_DIR)
        if args.dry_run:
            _print_dry_run_summary(
                attacks=attacks,
                category=category,
                run_id=run_id,
                config_hash=config_hash,
                cache_path=_PROMPTS_CACHE_DIR / f"{category}.json",
            )

    if not args.dry_run:
        logger.info("Prompt generation complete. Cache dir: %s", _PROMPTS_CACHE_DIR)


if __name__ == "__main__":
    main()
