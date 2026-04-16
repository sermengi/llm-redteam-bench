"""Modal cloud GPU inference client for HuggingFace models."""

import logging

import modal

from src.inference.client import ModelClient

logger = logging.getLogger(__name__)

_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers>=4.44",
    "torch>=2.0",
    "accelerate>=0.30",
    "huggingface_hub>=0.23",
)

app = modal.App("llm-redteam-bench-inference")


@app.cls(
    gpu="T4",
    image=_image,
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class _ModalInference:
    """GPU-resident HuggingFace inference class. Model is loaded once per container."""

    model_name: str = modal.parameter()

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import pipeline as hf_pipeline

        logger.info("Loading model %s on Modal T4 ...", self.model_name)
        self._pipe = hf_pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        logger.info("Model %s loaded.", self.model_name)

    @modal.method()
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Run inference and return the model's response string."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        outputs = self._pipe(messages, max_new_tokens=512, return_full_text=False)
        return outputs[0]["generated_text"]


class ModalHFClient(ModelClient):
    """Local wrapper that dispatches inference to a Modal cloud T4 GPU container.

    The remote container loads the model once and handles all .generate() calls.
    """

    def __init__(self, model_name: str) -> None:
        """Bind to a remote Modal container for the given model.

        Args:
            model_name: HuggingFace model identifier string.
        """
        self.model_name = model_name
        self._remote = _ModalInference(model_name=model_name)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """Dispatch a single inference request to the Modal container.

        Args:
            prompt: The user-turn input.
            system_prompt: The system instruction.

        Returns:
            Model response string.
        """
        return self._remote.generate.remote(prompt, system_prompt)
