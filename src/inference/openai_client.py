"""OpenAI API inference client — used for the LLM-as-judge scorer."""

import logging
import os

from openai import OpenAI

from src.inference.client import ModelClient

logger = logging.getLogger(__name__)


class OpenAIClient(ModelClient):
    """Inference client for OpenAI models.

    Reads OPENAI_API_KEY from the environment. Never pass the key as an argument.
    """

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        """Initialise the OpenAI client.

        Args:
            model: OpenAI model identifier (e.g. 'gpt-4o-mini').
            temperature: Sampling temperature. Use 0.0 for deterministic judge scoring.
        """
        self.model = model
        self.temperature = temperature
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running the evaluation pipeline."
            )
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Call the OpenAI chat completions API.

        Args:
            prompt: The user-turn message.
            system_prompt: Optional system-turn message. Omitted if empty string.

        Returns:
            Model response content as a string.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug("Calling OpenAI model=%s", self.model)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"OpenAI model {self.model} returned no text content.")
        return content
