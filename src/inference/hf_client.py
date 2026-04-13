"""HuggingFace Transformers inference client."""

import logging

from src.inference.client import ModelClient

logger = logging.getLogger(__name__)

_MOCK_RESPONSE = "I'm sorry, I can't help with that request."


class HuggingFaceClient(ModelClient):
    """Inference client for HuggingFace models.

    In Week 1, operates in mock mode (returns a fixed string).
    In Week 2, set mock=False and provide a loaded pipeline via real HF Transformers.
    """

    def __init__(self, model_name: str, mock: bool = True) -> None:
        """Initialise the client.

        Args:
            model_name: HuggingFace model identifier (e.g. 'meta-llama/Llama-3.1-8B-Instruct').
            mock: If True, returns a fixed response without loading any model weights.
                  Flip to False in configs/models.yaml when real inference is available.
        """
        self.model_name = model_name
        self.mock = mock
        if not mock:
            raise NotImplementedError(
                "Real HuggingFace inference is not yet implemented. "
                "Set mock: true in configs/models.yaml for Week 1."
            )

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Return a mock response (Week 1) or real model output (Week 2).

        Args:
            prompt: The attacker prompt.
            system_prompt: The system instruction (unused in mock mode).

        Returns:
            Model response string.
        """
        if self.mock:
            logger.debug("Mock inference for model=%s", self.model_name)
            return _MOCK_RESPONSE
        raise NotImplementedError("Real inference not implemented — set mock: true.")
