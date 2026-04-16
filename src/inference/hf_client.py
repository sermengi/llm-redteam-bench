"""HuggingFace Transformers inference client."""

import logging

from transformers import pipeline

from src.inference.client import ModelClient

logger = logging.getLogger(__name__)


class HuggingFaceClient(ModelClient):
    """Inference client for HuggingFace models. Loads model weights once on init."""

    def __init__(self, model_name: str) -> None:
        """Load the model and tokenizer into memory.

        Args:
            model_name: HuggingFace model identifier (e.g. 'meta-llama/Llama-3.1-8B-Instruct').
        """
        self.model_name = model_name
        logger.info("Loading model %s ...", model_name)
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="float16",
            device_map="auto",
        )
        logger.info("Model %s loaded.", model_name)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Run inference and return the model's response.

        Args:
            prompt: The user-turn input.
            system_prompt: The system instruction. Omitted from messages if empty.

        Returns:
            Model response string.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        outputs = self._pipe(messages, max_new_tokens=512, return_full_text=False)
        return outputs[0]["generated_text"]
