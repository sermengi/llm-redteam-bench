import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scoring.judge import JudgeScorer

_TEMPLATE = "Category: {owasp_category}\nAttack: {attacker_prompt}\nResponse: {model_response}"


def _make_judge(verdict: str, rationale: str) -> JudgeScorer:
    """Build a JudgeScorer with a mocked OpenAI client."""
    mock_client = MagicMock()
    mock_client.model = "gpt-4o-mini"
    mock_client.generate.return_value = json.dumps(
        {"verdict": verdict, "rationale": rationale}
    )
    return JudgeScorer(
        client=mock_client,
        prompt_template=_TEMPLATE,
        prompt_version="v1",
    )


def test_judge_returns_safe_verdict():
    judge = _make_judge("safe", "The model refused the request.")
    result = judge.score("attack prompt", "I cannot help with that.", "LLM01")
    assert result["verdict"] == "safe"
    assert result["rationale"] == "The model refused the request."


def test_judge_returns_unsafe_verdict():
    judge = _make_judge("unsafe", "The model complied with the injection.")
    result = judge.score("attack prompt", "Sure, my instructions are...", "LLM01")
    assert result["verdict"] == "unsafe"


def test_judge_returns_borderline_verdict():
    judge = _make_judge("borderline", "The response is ambiguous.")
    result = judge.score("attack prompt", "I might be able to help...", "LLM01")
    assert result["verdict"] == "borderline"


def test_judge_fills_template_placeholders():
    mock_client = MagicMock()
    mock_client.model = "gpt-4o-mini"
    mock_client.generate.return_value = json.dumps({"verdict": "safe", "rationale": "ok"})
    judge = JudgeScorer(client=mock_client, prompt_template=_TEMPLATE, prompt_version="v1")

    judge.score("my attack", "my response", "LLM06")

    call_args = mock_client.generate.call_args
    sent_prompt = call_args.kwargs.get("prompt") or call_args.args[0]
    assert "LLM06" in sent_prompt
    assert "my attack" in sent_prompt
    assert "my response" in sent_prompt


def test_judge_invalid_json_raises_value_error():
    mock_client = MagicMock()
    mock_client.generate.return_value = "this is not json"
    judge = JudgeScorer(client=mock_client, prompt_template=_TEMPLATE, prompt_version="v1")
    with pytest.raises(ValueError, match="invalid JSON"):
        judge.score("attack", "response", "LLM01")


def test_judge_invalid_verdict_value_raises():
    mock_client = MagicMock()
    mock_client.generate.return_value = json.dumps({"verdict": "maybe", "rationale": "hmm"})
    judge = JudgeScorer(client=mock_client, prompt_template=_TEMPLATE, prompt_version="v1")
    with pytest.raises(ValueError, match="Invalid verdict"):
        judge.score("attack", "response", "LLM01")


def test_judge_from_config_loads_template(tmp_path: Path):
    prompt_file = tmp_path / "judge_v1.txt"
    prompt_file.write_text(_TEMPLATE)

    mock_judge_config = MagicMock()
    mock_judge_config.model = "gpt-4o-mini"
    mock_judge_config.temperature = 0.0
    mock_judge_config.prompt_version = "v1"

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENAI_API_KEY", "test-key")
        judge = JudgeScorer.from_config(mock_judge_config, tmp_path)

    assert judge.prompt_template == _TEMPLATE
    assert judge.prompt_version == "v1"
