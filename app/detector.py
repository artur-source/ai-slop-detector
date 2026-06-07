# Local and API-based AI text detection logic.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from config import AppConfig


load_dotenv()

config = AppConfig()
_model: DistilBertForSequenceClassification | None = None
_tokenizer: DistilBertTokenizer | None = None


def _load_local_model() -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
    """Load and cache the fine-tuned local DistilBERT classifier.

    Returns:
        A tuple containing the cached model and tokenizer.

    Raises:
        FileNotFoundError: If the saved model directory does not exist.
    """

    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    saved_model_path = Path(config.saved_model_path)
    if not saved_model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {saved_model_path}. "
            "Train the model first with: python model/train_classifier.py"
        )

    _tokenizer = DistilBertTokenizer.from_pretrained(saved_model_path)
    _model = DistilBertForSequenceClassification.from_pretrained(saved_model_path)
    _model.to(config.device)
    _model.eval()

    return _model, _tokenizer


def detect_local(text: str) -> dict[str, Any]:
    """Classify text with the fine-tuned local DistilBERT model.

    Args:
        text: Input text to classify.

    Returns:
        Dictionary containing label, confidence score, class probabilities, and method.
    """

    model, tokenizer = _load_local_model()
    encoded_text = tokenizer(
        text,
        max_length=config.max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    encoded_text = {key: value.to(config.device) for key, value in encoded_text.items()}

    with torch.no_grad():
        outputs = model(**encoded_text)
        probabilities = torch.softmax(outputs.logits, dim=1).squeeze()
        predicted_class = int(torch.argmax(probabilities).item())

    human_probability = float(probabilities[0].item())
    ai_probability = float(probabilities[1].item())
    predicted_score = ai_probability if predicted_class == 1 else human_probability

    return {
        "label": "AI" if predicted_class == 1 else "Human",
        "score": predicted_score,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
        "method": "local",
    }


def detect_openai(text: str) -> dict[str, Any]:
    """Classify text with the OpenAI API for side-by-side comparison.

    Args:
        text: Input text to classify.

    Returns:
        Dictionary containing OpenAI classification data, or a friendly error message.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set", "method": "openai"}

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at detecting AI-generated text. Analyze the text and respond "
                        "only with valid JSON in this exact format: "
                        '{"label": "AI" or "Human", "score": <float 0.0 to 1.0>, '
                        '"reasoning": "<one sentence explanation>"}'
                    ),
                },
                {"role": "user", "content": f"Classify this text:\n\n{text[:3000]}"},
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned an empty response")

        parsed = json.loads(content)
        return {
            "label": parsed["label"],
            "score": float(parsed["score"]),
            "reasoning": parsed["reasoning"],
            "method": "openai",
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"error": f"Could not parse OpenAI response: {exc}", "method": "openai"}
    except Exception as exc:
        return {"error": f"OpenAI detection failed: {exc}", "method": "openai"}


def detect(text: str) -> dict[str, Any]:
    """Run local and OpenAI detection on normalized text.

    Args:
        text: Input text to classify.

    Returns:
        Dictionary containing local result, OpenAI result, text length, and truncation status.

    Raises:
        ValueError: If the input text is shorter than the configured minimum length.
    """

    if len(text) < config.min_text_length:
        raise ValueError(f"Text too short. Minimum {config.min_text_length} characters required.")

    truncated = len(text) > config.max_text_length
    detection_text = text[: config.max_text_length] if truncated else text

    local_result = detect_local(detection_text)
    openai_result = detect_openai(detection_text)

    return {
        "local": local_result,
        "openai": openai_result,
        "text_length": len(detection_text),
        "truncated": truncated,
    }

