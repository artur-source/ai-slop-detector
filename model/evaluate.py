# Evaluation script for measuring classifier performance.

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from config import AppConfig


load_dotenv()

config = AppConfig()
SEED = 42


def _first_non_empty_answer(answers: list[str] | None) -> str | None:
    """Return the first non-empty answer from an HC3 answer list.

    Args:
        answers: List of candidate answers from one HC3 record.

    Returns:
        The first non-empty stripped answer, or None when no valid text exists.
    """

    if not answers:
        return None

    for answer in answers:
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    return None


def _load_test_samples() -> tuple[list[str], list[int]]:
    """Load a balanced 20% HC3 test split for evaluation.

    Returns:
        A tuple containing test texts and integer labels, where 0 is human text
        and 1 is AI-generated text.
    """

    raw_dataset = load_dataset("Hello-SimpleAI/HC3", "all")
    records = raw_dataset["train"] if isinstance(raw_dataset, DatasetDict) else raw_dataset

    human_texts: list[str] = []
    ai_texts: list[str] = []

    for item in records:
        human_answer = _first_non_empty_answer(item.get("human_answers"))
        ai_answer = _first_non_empty_answer(item.get("chatgpt_answers"))

        if human_answer and len(human_answer) >= 50:
            human_texts.append(human_answer)

        if ai_answer and len(ai_answer) >= 50:
            ai_texts.append(ai_answer)

    sample_count = min(len(human_texts), len(ai_texts))
    human_texts = human_texts[:sample_count]
    ai_texts = ai_texts[:sample_count]

    texts = human_texts + ai_texts
    labels = [0] * sample_count + [1] * sample_count
    paired_samples = list(zip(texts, labels, strict=True))
    random.Random(SEED).shuffle(paired_samples)

    split_start = int(len(paired_samples) * config.train_ratio)
    test_samples = paired_samples[split_start:]
    test_texts = [text for text, _ in test_samples]
    test_labels = [label for _, label in test_samples]

    return test_texts, test_labels


def evaluate_model() -> dict[str, float]:
    """Evaluate the saved DistilBERT classifier on a balanced HC3 test split.

    Returns:
        A dictionary with weighted accuracy, precision, recall, and F1 metrics.
    """

    saved_model_path = Path(config.saved_model_path)
    assets_path = Path("assets")
    assets_path.mkdir(parents=True, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained(saved_model_path)
    model = DistilBertForSequenceClassification.from_pretrained(saved_model_path)
    model.to(config.device)
    model.eval()

    test_texts, test_labels = _load_test_samples()
    tokenized = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=config.max_length,
    )

    test_dataset = Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": test_labels,
        }
    )
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {key: value.to(config.device) for key, value in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            batch_predictions = torch.argmax(outputs.logits, dim=1)

            predictions.extend(batch_predictions.cpu().tolist())
            references.extend(labels.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(references, predictions),
        "precision": precision_score(references, predictions, average="weighted", zero_division=0),
        "recall": recall_score(references, predictions, average="weighted", zero_division=0),
        "f1": f1_score(references, predictions, average="weighted", zero_division=0),
    }

    display = ConfusionMatrixDisplay.from_predictions(
        references,
        predictions,
        display_labels=["Human", "AI"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(assets_path / "confusion_matrix.png", dpi=150)
    plt.show()

    return metrics


if __name__ == "__main__":
    evaluation_metrics = evaluate_model()

    print("Evaluation metrics:")
    for metric_name, metric_value in evaluation_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
