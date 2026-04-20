"""PyRIT-based attack prompt generator (shallow: converter-only, no target/memory)."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from threading import Thread
from typing import Any, Coroutine, Literal

from pyrit.prompt_converter import (
    Base64Converter,
    LeetspeakConverter,
    PromptConverter,
    ROT13Converter,
)

from src.attacks.loader import AttackPrompt

logger = logging.getLogger(__name__)

# multi_turn_crescendo strategy is reserved in the schema but not yet enacted.
# A real multi-turn crescendo loop is deferred to a future milestone.

# Shared instances are safe because these converters are stateless.
# If adding stateful converters later, instantiate per-call instead.
_CONVERTER_MAP: dict[str, PromptConverter] = {
    "base64": Base64Converter(),
    "rot13": ROT13Converter(),
    "leetspeak": LeetspeakConverter(),
}


def _run_coro_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine in sync code, including when an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Jupyter/pytest-asyncio case: run in a dedicated thread with its own loop.
    result_future: Future[object] = Future()

    def _runner() -> None:
        try:
            result_future.set_result(asyncio.run(coro))
        except Exception as exc:  # pragma: no cover - exception path is re-raised below
            result_future.set_exception(exc)

    thread = Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    return result_future.result()


def _convert_sync(text: str, converter: "PromptConverter") -> str:
    """Apply a PyRIT converter to a text string synchronously.

    Handles both plain sync contexts and running event-loop contexts (e.g.
    Jupyter notebooks) by running converter coroutines in a dedicated thread
    when needed.

    Args:
        text: The seed prompt string to convert.
        converter: An instantiated PyRIT PromptConverter.

    Returns:
        The converted prompt string.
    """
    result = _run_coro_sync(converter.convert_async(prompt=text, input_type="text"))
    return result.output_text


def generate_pyrit_prompts(
    category: str,
    seed_prompts: list[str],
    n: int,
    converters: list[str] | None = None,
) -> list[AttackPrompt]:
    """Generate n attack prompts using PyRIT converters applied to seed_prompts.

    Applies each active converter to each seed prompt, then takes the first n.
    First ceil(n/2) are direct_injection, rest are indirect_injection.

    Args:
        category: OWASP category string (unused here, reserved for future filtering).
        seed_prompts: Base prompts to convert (typically the 5 manual prompts).
        n: Total number of PyRIT prompts to return.
        converters: Converter names to use (subset of 'base64', 'rot13', 'leetspeak').
                    None means use all three.

    Returns:
        List of AttackPrompt with attack_source='pyrit'.
    """
    active_names = converters if converters is not None else list(_CONVERTER_MAP)
    active_converters = [_CONVERTER_MAP[name] for name in active_names]

    variants: list[str] = []
    for converter in active_converters:
        for seed in seed_prompts:
            try:
                variants.append(_convert_sync(seed, converter))
            except Exception:
                logger.warning(
                    "Converter %s failed on seed, skipping.",
                    type(converter).__name__,
                    exc_info=True,
                )

    variants = variants[:n]
    if len(variants) < n:
        raise RuntimeError(
            f"generate_pyrit_prompts: needed {n} variants but only produced "
            f"{len(variants)}. Check converter failure warnings in the logs."
        )
    actual_n = len(variants)
    n_direct = (actual_n + 1) // 2  # ceiling division — direct gets the extra if n is odd

    result: list[AttackPrompt] = []
    for i, prompt in enumerate(variants):
        strategy: Literal["direct_injection", "indirect_injection"] = (
            "direct_injection" if i < n_direct else "indirect_injection"
        )
        result.append(AttackPrompt(prompt=prompt, attack_source="pyrit", attack_strategy=strategy))

    logger.info("Generated %d PyRIT prompts for %s", len(result), category)
    return result
