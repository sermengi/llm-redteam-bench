from unittest.mock import MagicMock

from src.inference.client import ModelClient
from src.inference.modal_client import ModalHFClient


def _make_client_with_mock_remote(response: str) -> ModalHFClient:
    """Construct ModalHFClient bypassing __init__ and injecting a mock remote."""
    client = ModalHFClient.__new__(ModalHFClient)
    client.model_name = "meta-llama/Llama-3.1-8B-Instruct"
    mock_remote = MagicMock()
    mock_remote.generate.remote.return_value = response
    client._remote = mock_remote
    return client


def test_modal_hf_client_is_model_client():
    client = _make_client_with_mock_remote("test")
    assert isinstance(client, ModelClient)


def test_modal_hf_client_generate_returns_string():
    client = _make_client_with_mock_remote("Response from T4 GPU.")
    result = client.generate("test prompt", "test system")
    assert result == "Response from T4 GPU."


def test_modal_hf_client_passes_prompt_and_system_to_remote():
    client = _make_client_with_mock_remote("response")
    client.generate("my attack prompt", "my system prompt")
    client._remote.generate.remote.assert_called_once_with("my attack prompt", "my system prompt")


def test_modal_hf_client_stores_model_name():
    client = _make_client_with_mock_remote("response")
    assert client.model_name == "meta-llama/Llama-3.1-8B-Instruct"
