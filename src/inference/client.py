"""Abstract base class for all model inference clients."""

from abc import ABC, abstractmethod


class ModelClient(ABC):
    """Unified interface for model inference. Implementations: HuggingFaceClient, OpenAIClient."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a response to the given prompt.

        Args:
            prompt: The user-turn input to send to the model.
            system_prompt: The system-turn instruction. May be empty string.

        Returns:
            The model's response as a plain string.
        """
