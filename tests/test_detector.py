# Tests for local and API-based detector behavior.

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


class FakeScalar:
    """Small scalar wrapper that mimics the PyTorch item API used by detector tests."""

    def __init__(self, value: float):
        self.value = value

    def item(self) -> float:
        """Return the wrapped scalar value."""

        return self.value


class FakeTensor:
    """Tiny tensor substitute for the detector unit tests."""

    def __init__(self, values: list[float] | list[list[float]]):
        self.values = values

    def to(self, device: str) -> "FakeTensor":
        """Mimic moving a tensor to a device."""

        return self

    def squeeze(self) -> "FakeTensor":
        """Mimic removing singleton dimensions from a tensor."""

        if self.values and isinstance(self.values[0], list):
            return FakeTensor(self.values[0])
        return self

    def __getitem__(self, index: int) -> FakeScalar:
        return FakeScalar(self.values[index])


class FakeNoGrad:
    """Context manager replacement for torch.no_grad."""

    def __enter__(self) -> None:
        """Enter the no-grad context."""

        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        """Exit the no-grad context."""

        return False


def _fake_softmax(tensor: FakeTensor, dim: int) -> FakeTensor:
    """Return deterministic probabilities that favor the AI class."""

    return FakeTensor([0.05, 0.95])


def _fake_argmax(tensor: FakeTensor) -> FakeScalar:
    """Return index 1 to represent the AI class."""

    return FakeScalar(1)


fake_torch = types.ModuleType("torch")
fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
fake_torch.no_grad = lambda: FakeNoGrad()
fake_torch.softmax = _fake_softmax
fake_torch.argmax = _fake_argmax
fake_torch.tensor = lambda values: FakeTensor(values)

fake_transformers = types.ModuleType("transformers")
fake_transformers.DistilBertForSequenceClassification = object
fake_transformers.DistilBertTokenizer = object

fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = Mock()

sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("transformers", fake_transformers)
sys.modules.setdefault("openai", fake_openai)

from app import detector


def test_detect_raises_value_error_for_short_text() -> None:
    """detect should reject text shorter than the configured minimum length."""

    with pytest.raises(ValueError, match="Text too short"):
        detector.detect("short text")


def test_detect_local_returns_ai_label() -> None:
    """detect_local should return AI when mocked logits favor class 1."""

    mock_model = Mock()
    mock_model.return_value = SimpleNamespace(logits=fake_torch.tensor([[0.1, 3.0]]))

    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        "input_ids": fake_torch.tensor([[101, 2023, 102]]),
        "attention_mask": fake_torch.tensor([[1, 1, 1]]),
    }

    with patch("app.detector._load_local_model", return_value=(mock_model, mock_tokenizer)):
        result = detector.detect_local("This is a long enough sample text for local detection.")

    assert result["label"] == "AI"
    assert result["method"] == "local"
    assert 0.0 <= result["score"] <= 1.0


def test_detect_openai_returns_error_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_openai should return a friendly error when the API key is missing."""

    monkeypatch.setenv("OPENAI_API_KEY", "")

    result = detector.detect_openai("This is a sample text for OpenAI detection.")

    assert result["error"] == "OPENAI_API_KEY not set"
    assert result["method"] == "openai"


def test_detect_openai_parses_valid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_openai should parse a valid JSON response from the OpenAI client."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_create = Mock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"label": "Human", "score": 0.85, "reasoning": "test"}'
                    )
                )
            ]
        )
    )
    mock_client = Mock()
    mock_client.chat.completions.create = mock_create

    with patch("app.detector.OpenAI", return_value=mock_client):
        result = detector.detect_openai("This is a sample text for OpenAI detection.")

    assert result["label"] == "Human"
    assert result["score"] == 0.85
    assert result["reasoning"] == "test"
    assert result["method"] == "openai"


def test_detect_truncates_long_text() -> None:
    """detect should truncate text longer than the configured maximum length."""

    local_result = {"label": "Human", "score": 0.9, "method": "local"}
    openai_result = {"label": "Human", "score": 0.8, "reasoning": "test", "method": "openai"}
    long_text = "a" * (detector.config.max_text_length + 100)

    with patch("app.detector.detect_local", return_value=local_result):
        with patch("app.detector.detect_openai", return_value=openai_result):
            result = detector.detect(long_text)

    assert result["truncated"] is True
    assert result["text_length"] == detector.config.max_text_length
