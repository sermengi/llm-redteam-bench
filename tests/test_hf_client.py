from unittest.mock import MagicMock, patch

from src.inference.hf_client import HuggingFaceClient


def test_hf_client_generate_calls_pipeline(monkeypatch):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "Hello from the model."}]

    with patch("src.inference.hf_client.pipeline", return_value=mock_pipe) as mock_pipeline_fn:
        client = HuggingFaceClient(model_name="meta-llama/Llama-3.1-8B-Instruct")
        result = client.generate("Tell me a joke.", "You are a comedian.")

    assert result == "Hello from the model."
    mock_pipeline_fn.assert_called_once_with(
        "text-generation",
        model="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype="float16",
        device_map="auto",
    )


def test_hf_client_includes_system_prompt_in_messages():
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "response"}]

    with patch("src.inference.hf_client.pipeline", return_value=mock_pipe):
        client = HuggingFaceClient(model_name="test-model")
        client.generate("user prompt", "system instruction")

    call_args = mock_pipe.call_args
    messages = call_args[0][0]
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_hf_client_omits_system_prompt_when_empty():
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "response"}]

    with patch("src.inference.hf_client.pipeline", return_value=mock_pipe):
        client = HuggingFaceClient(model_name="test-model")
        client.generate("user prompt", "")

    call_args = mock_pipe.call_args
    messages = call_args[0][0]
    roles = [m["role"] for m in messages]
    assert "system" not in roles
    assert "user" in roles
